"""Read-only, device-resident joint-speed diagnostics; no dynamics corrections."""
import torch


class PhysicsSpeedDiagnostics:
    def __init__(self, num_envs, num_joints, dt, device):
        self.dt, self.num_envs, self.num_joints = dt, num_envs, num_joints
        self.samples = 0
        self.counts = torch.zeros(3, dtype=torch.long, device=device)
        self.peak = torch.zeros(2, device=device)
        self.streak = torch.zeros((num_envs, num_joints), dtype=torch.long, device=device)
        self.max_streak = torch.zeros((), dtype=torch.long, device=device)

    def update(self, velocity, limits, pd_estimate):
        absolute = velocity.abs()
        over = absolute > limits + .001
        driven = over & (velocity * pd_estimate > 0)
        self.samples += 1
        self.counts += torch.stack((over.any(dim=1).sum(), over.sum(), driven.sum()))
        self.peak = torch.maximum(self.peak, torch.stack((absolute.max(), (absolute / limits).max())))
        self.streak = torch.where(driven, self.streak + 1, 0)
        self.max_streak = torch.maximum(self.max_streak, self.streak.max())

    def reset_envs(self, ids):
        self.streak[ids] = 0

    def report(self):
        env_over, joint_over, driven = self.counts.cpu().tolist()
        peak, ratio = self.peak.cpu().tolist()
        env_samples = self.samples * self.num_envs
        joint_samples = env_samples * self.num_joints
        fraction = env_over / env_samples if env_samples else 0.
        duration = int(self.max_streak) * self.dt
        return {'physics_dt_s': self.dt, 'physics_steps': self.samples,
                'environment_time_samples': env_samples, 'joint_time_samples': joint_samples,
                'environment_over_limit_samples': env_over, 'joint_over_limit_samples': joint_over,
                'over_limit_environment_time_fraction': fraction,
                'over_limit_joint_time_fraction': joint_over / joint_samples if joint_samples else 0.,
                'peak_abs_velocity_rad_s': peak, 'peak_limit_ratio': ratio,
                'driven_proxy_over_limit_joint_samples': driven,
                'longest_driven_proxy_over_limit_s': duration,
                'flag_over_one_percent_time': fraction > .01,
                'flag_sustained_driven_proxy': duration >= .05,
                'tolerance_rad_s': .001,
                'scope': 'Every completed physics step before autoreset, including settling; limits are actual per-environment randomized solver caps.',
                'interpretation': 'Joint velocity constraint may be exceeded by solver/contact interactions. PD sign is an unclipped estimate, not measured drive force. Flags require review, not proof of policy exploitation. Sustained proxy threshold is one policy period (0.05s).'}
