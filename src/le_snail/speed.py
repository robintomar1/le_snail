"""Read-only, device-resident joint-speed diagnostics; no dynamics corrections."""
import torch


class PhysicsSpeedDiagnostics:
    def __init__(self, num_envs, num_joints, dt, device):
        self.dt, self.num_envs, self.num_joints = dt, num_envs, num_joints
        self.samples = 0
        self.env_samples = torch.zeros((), dtype=torch.long, device=device)
        self.counts = torch.zeros(3, dtype=torch.long, device=device)
        self.peak = torch.zeros(2, device=device)
        self.streak = torch.zeros((num_envs, num_joints), dtype=torch.long, device=device)
        self.max_streak = torch.zeros((), dtype=torch.long, device=device)
        # Where the longest driven-proxy streak ended: env, joint, physics sample, optional context row.
        self.sample_index = torch.zeros((), dtype=torch.long, device=device)
        self.max_location = torch.full((3,), -1, dtype=torch.long, device=device)
        self.max_context = None

    def update(self, velocity, limits, pd_estimate, mask=None, context=None):
        """mask: optional per-env bool; False rows (teacher-controlled steps) are not counted.
        context: optional (envs, k) float values stored for the env holding the longest streak."""
        absolute = velocity.abs()
        if mask is not None:
            absolute = absolute * mask[:, None]
        over = absolute > limits + .001
        driven = over & (velocity * pd_estimate > 0)
        self.samples += 1
        self.env_samples += self.num_envs if mask is None else mask.sum()
        self.counts += torch.stack((over.any(dim=1).sum(), over.sum(), driven.sum()))
        self.peak = torch.maximum(self.peak, torch.stack((absolute.max(), (absolute / limits).max())))
        self.streak = torch.where(driven, self.streak + 1, 0)
        self.sample_index += 1
        current = self.streak.max()
        improved = current > self.max_streak
        flat = self.streak.argmax()
        env = flat // self.num_joints
        location = torch.stack((env, flat % self.num_joints, self.sample_index))
        self.max_location = torch.where(improved, location, self.max_location)
        if context is not None:
            row = context[env]
            self.max_context = row.clone() if self.max_context is None else torch.where(improved, row, self.max_context)
        self.max_streak = torch.maximum(self.max_streak, current)

    def reset_envs(self, ids):
        self.streak[ids] = 0

    def report(self):
        env_over, joint_over, driven = self.counts.cpu().tolist()
        peak, ratio = self.peak.cpu().tolist()
        env_samples = int(self.env_samples)
        joint_samples = env_samples * self.num_joints
        fraction = env_over / env_samples if env_samples else 0.
        duration = int(self.max_streak) * self.dt
        location = None
        if int(self.max_streak) > 0:
            env, joint, sample = self.max_location.cpu().tolist()
            location = {'env': env, 'joint_index': joint, 'physics_sample': sample,
                        'context': None if self.max_context is None else self.max_context.cpu().tolist()}
        return {'physics_dt_s': self.dt, 'physics_steps': self.samples,
                'environment_time_samples': env_samples, 'joint_time_samples': joint_samples,
                'environment_over_limit_samples': env_over, 'joint_over_limit_samples': joint_over,
                'over_limit_environment_time_fraction': fraction,
                'over_limit_joint_time_fraction': joint_over / joint_samples if joint_samples else 0.,
                'peak_abs_velocity_rad_s': peak, 'peak_limit_ratio': ratio,
                'driven_proxy_over_limit_joint_samples': driven,
                'longest_driven_proxy_over_limit_s': duration,
                'longest_driven_proxy_location': location,
                'flag_over_one_percent_time': fraction > .01,
                'flag_sustained_driven_proxy': duration >= .05,
                'tolerance_rad_s': .001,
                'scope': 'Every completed physics step before autoreset, including settling; limits are actual per-environment randomized solver caps. '
                         'Teacher-controlled handover-prefix steps are excluded from counts and denominators.',
                'interpretation': 'Joint velocity constraint may be exceeded by solver/contact interactions. PD sign is an unclipped estimate, not measured drive force. Flags require review, not proof of policy exploitation. Sustained proxy threshold is one policy period (0.05s).'}
