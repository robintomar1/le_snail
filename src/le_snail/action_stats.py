"""Read-only action-path counters, accumulated before any clipping or slew limit."""
import torch


class ActionDiagnostics:
    names = ('raw_out_of_range', 'joint_limit_clipped', 'slew_limited', 'controller_at_slew_bound', 'desired_at_joint_bound')

    def __init__(self, num_envs, joint_names, device, initial_steps=64):
        self.num_envs = num_envs
        self.joint_names = list(joint_names)
        self.initial_steps = initial_steps
        self.steps = 0
        self.counts = torch.zeros((len(self.names), len(joint_names)), dtype=torch.long, device=device)
        self.initial_counts = torch.zeros_like(self.counts)
        self.samples = torch.zeros((), dtype=torch.long, device=device)
        self.initial_samples = torch.zeros_like(self.samples)
        self.raw_peak = torch.zeros(len(joint_names), device=device)

    def update(self, raw_actions, unclipped_target, desired_target, old_target, new_target, slew, lower, upper, mask=None):
        """mask: optional per-env bool; False rows (teacher-controlled steps) are not counted."""
        # These comparisons do not modify the input actions or any controller state.
        masks = torch.stack((raw_actions.abs() > 1.,
                             unclipped_target != desired_target,
                             (desired_target - old_target).abs() > slew + 1e-7,
                             (new_target - old_target).abs() >= slew - 1e-7,
                             (desired_target <= lower + 1e-7) | (desired_target >= upper - 1e-7)))
        magnitude = raw_actions.abs()
        rows = self.num_envs
        if mask is not None:
            masks &= mask[None, :, None]
            magnitude = magnitude * mask[:, None]
            rows = mask.sum()
        counts = masks.sum(dim=1)
        self.counts += counts
        self.samples += rows
        if self.steps < self.initial_steps:
            self.initial_counts += counts
            self.initial_samples += rows
        self.raw_peak = torch.maximum(self.raw_peak, magnitude.amax(dim=0))
        self.steps += 1

    def report(self):
        def window(counts, samples, steps):
            denominator = int(samples)
            data = counts.cpu().tolist()
            return {'policy_steps': steps, 'samples_per_joint': denominator,
                    'counts_per_joint': dict(zip(self.names, data)),
                    'fractions_per_joint': {name: [v / denominator if denominator else 0. for v in row]
                                            for name, row in zip(self.names, data)},
                    'fractions_all_joints': {name: sum(row) / (denominator * len(self.joint_names)) if denominator else 0.
                                            for name, row in zip(self.names, data)}}
        return {'joint_names': self.joint_names,
                'all': window(self.counts, self.samples, self.steps),
                'initial': window(self.initial_counts, self.initial_samples, min(self.steps, self.initial_steps)),
                'initial_window_max_policy_steps': self.initial_steps,
                'peak_abs_raw_action_per_joint': self.raw_peak.cpu().tolist(),
                'scope': 'Every public policy step, including evaluation settling; counters persist across episode resets. '
                         'Teacher-controlled handover-prefix steps are excluded from counts and denominators.',
                'definitions': {'raw_out_of_range': 'Raw policy action strictly outside [-1,1], before any clamp.',
                                'joint_limit_clipped': 'home+clamped_action*scale differs from joint-limit-clipped desired target.',
                                'slew_limited': 'Desired minus previous controller target exceeds allowed step by >1e-7 rad.',
                                'controller_at_slew_bound': 'New controller target step reaches allowed slew step within 1e-7 rad; before optional command delay.',
                                'desired_at_joint_bound': 'Desired target within 1e-7 rad of either policy joint limit, including exact endpoints.'}}
