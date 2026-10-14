"""Validation of tunable physics parameters before launching expensive simulation."""
import json
import math
from pathlib import Path


def load_physics(path):
    config = json.loads(Path(path).read_text())
    for key in ('parameters_measured', 'contact_geometry_validated', 'protected_points_validated'):
        if type(config[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    positive = ('physics_dt', 'stiffness_nm_rad', 'damping_nm_s_rad', 'effort_limit_nm',
                'velocity_limit_rad_s', 'initial_height_m', 'side_height_m',
                'drag_velocity_scale_m_s', 'max_fallen_time_s')
    for key in positive:
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('static_friction_range', 'mass_scale_range', 'gain_scale_range', 'drag_force_range_n'):
        values = config[key]
        if len(values) != 2 or not all(math.isfinite(v) for v in values) or not 0 <= values[0] <= values[1]:
            raise ValueError(f'{key} must be a nonnegative ordered interval')
        if key in ('mass_scale_range', 'gain_scale_range') and values[0] <= 0:
            raise ValueError(f'{key} must be positive')
    for key in ('position_noise_std_rad', 'gyro_noise_std_rad_s', 'gravity_noise_std'):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('dynamic_friction_ratio', 'recovery_start_fraction'):
        if not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
    if config['max_action_delay_steps'] not in (0, 1):
        raise ValueError('Only zero or one action delay step is supported')
    attachment = config['tether_attachment_base_m']
    if len(attachment) != 3 or not all(math.isfinite(v) for v in attachment):
        raise ValueError('Tether attachment requires three finite coordinates')
    for probe in config['protected_points']:
        if not probe['body'] or len(probe['position_m']) != 3 or not all(math.isfinite(v) for v in probe['position_m']):
            raise ValueError('Invalid protected-point location')
        if not math.isfinite(probe['radius_m']) or probe['radius_m'] <= 0:
            raise ValueError('Protected-point radius must be positive')
    if 'positive_progress_upright_range' in config:
        bounds = config['positive_progress_upright_range']
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or not -1 <= bounds[0] < bounds[1] <= 1:
            raise ValueError('Progress posture gate requires an increasing interval within [-1, 1]')
    if 'yaw_cost_weight' in config and (not math.isfinite(config['yaw_cost_weight']) or config['yaw_cost_weight'] < 0):
        raise ValueError('Yaw cost weight must be finite and nonnegative')
    if 'settled_starts' in config:
        if set(config['settled_starts']) != {'upright', 'left', 'right'}:
            raise ValueError('Settled starts require upright, left and right states')
        for state in config['settled_starts'].values():
            pose, joints = state['root_pose'], state['joint_rad']
            if len(pose) != 7 or len(joints) != 6 or not all(math.isfinite(x) for x in pose + joints):
                raise ValueError('Settled starts require finite root pose and six joint angles')
            if pose[2] <= 0 or not math.isclose(sum(x*x for x in pose[3:]), 1., abs_tol=1e-5):
                raise ValueError('Settled pose requires positive height and a unit quaternion')
        for key in ('settled_start_asset_sha256', 'settled_start_contract_fingerprint'):
            digest = config.get(key, '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Settled starts require asset and policy contract identity')
    return config
