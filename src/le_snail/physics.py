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
    return config
