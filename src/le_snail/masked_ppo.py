"""PPO that excludes scripted-prefix teacher transitions from learning (#6, handover curriculum).

Pinned to RSL-RL 3.0.1 feedforward PPO. Teacher-controlled steps are stored with the rollout but
carry zero weight in the surrogate, value, entropy and KL terms, in advantage normalization and in
observation-normalizer updates. With every step marked as a learner step the update matches the
stock PPO update. Recurrent policies, RND, symmetry and multi-GPU training are not supported.
"""
import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO


class MaskedPPO(PPO):
    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)
        self.learner_mask = torch.ones(num_transitions_per_env, num_envs, 1, device=self.device)

    def process_env_step(self, obs, rewards, dones, extras):
        if self.rnd or self.policy.is_recurrent:
            raise NotImplementedError('MaskedPPO supports feedforward policies without RND')
        mask = extras.get('learner_mask')
        mask = torch.ones_like(rewards, dtype=torch.float) if mask is None else mask.to(self.device).float()
        following = extras.get('next_observation_is_learner')
        following = torch.ones_like(mask) if following is None else following.to(self.device).float()
        # The normalizer sees only observations the learner will act on.
        rows = following > .5
        if bool(rows.all()):
            self.policy.update_normalization(obs)
        elif bool(rows.any()):
            self.policy.update_normalization(obs[rows])
        self.learner_mask[self.storage.step].copy_(mask.view(-1, 1))
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'time_outs' in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras['time_outs'].unsqueeze(1).to(self.device), 1)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam, normalize_advantage=False)
        if not self.normalize_advantage_per_mini_batch:
            self.storage.advantages = self._normalize(self.storage.advantages, self.learner_mask)

    @staticmethod
    def _normalize(advantages, mask):
        # Same unbiased statistics as the stock (advantages - mean) / (std + 1e-8), over learner steps only.
        count = mask.sum()
        mean = (advantages * mask).sum() / count
        variance = ((advantages - mean).square() * mask).sum() / (count - 1).clamp_min(1)
        return (advantages - mean) / (variance.sqrt() + 1e-8) * mask

    def _batches(self):
        storage = self.storage
        mini_batch_size = storage.num_envs * storage.num_transitions_per_env // self.num_mini_batches
        # Identical random draw to RolloutStorage.mini_batch_generator.
        indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        fields = (storage.observations.flatten(0, 1), storage.actions.flatten(0, 1), storage.values.flatten(0, 1),
                  storage.advantages.flatten(0, 1), storage.returns.flatten(0, 1),
                  storage.actions_log_prob.flatten(0, 1), storage.mu.flatten(0, 1), storage.sigma.flatten(0, 1),
                  self.learner_mask.flatten(0, 1))
        for _ in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                batch = indices[i * mini_batch_size:(i + 1) * mini_batch_size]
                yield tuple(field[batch] for field in fields)

    def update(self):
        if self.rnd or self.symmetry or self.is_multi_gpu or self.policy.is_recurrent:
            raise NotImplementedError('MaskedPPO supports single-GPU feedforward PPO only')
        totals = {'value_function': 0., 'surrogate': 0., 'entropy': 0.}
        learner_fraction, updates = 0., 0
        for (obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, mask_batch) in self._batches():
            weight = mask_batch.squeeze(-1)
            count = weight.sum()
            learner_fraction += float(count) / len(weight)
            if count < 1:
                continue

            def mean(values):
                return (values * weight).sum() / count

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = self._normalize(advantages_batch, mask_batch)
            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            mu_batch, sigma_batch = self.policy.action_mean, self.policy.action_std
            entropy_batch = self.policy.entropy
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                                   + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                                   / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif self.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group['lr'] = self.learning_rate
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            advantages = torch.squeeze(advantages_batch)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = mean(torch.max(surrogate, surrogate_clipped))
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = mean(torch.max(value_losses, value_losses_clipped).squeeze(-1))
            else:
                value_loss = mean((returns_batch - value_batch).pow(2).squeeze(-1))
            entropy = mean(entropy_batch)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals['value_function'] += value_loss.item()
            totals['surrogate'] += surrogate_loss.item()
            totals['entropy'] += entropy.item()
            updates += 1
        self.storage.clear()
        result = {name: value / max(updates, 1) for name, value in totals.items()}
        result['learner_fraction'] = learner_fraction / (self.num_learning_epochs * self.num_mini_batches)
        return result
