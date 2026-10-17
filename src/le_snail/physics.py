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
    if 'command_delay_ms' in config:
        nominal = config['command_delay_ms']
        if type(nominal) not in (int, float) or not math.isfinite(nominal) or nominal < 0:
            raise ValueError('command_delay_ms must be finite and nonnegative')
        if 'command_delay_range_ms' in config:
            low, high = config['command_delay_range_ms']
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= nominal <= high):
                raise ValueError('command_delay_range_ms must be a nonnegative ordered interval containing command_delay_ms')
    if 'distal_mass_scale_range' in config:
        low, high = config['distal_mass_scale_range']
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high) or not config.get('distal_bodies'):
            raise ValueError('distal_mass_scale_range must be a positive ordered interval with distal_bodies')
    for key in ('static_friction_range', 'mass_scale_range', 'gain_scale_range', 'drag_force_range_n'):
        values = config[key]
        if len(values) != 2 or not all(math.isfinite(v) for v in values) or not 0 <= values[0] <= values[1]:
            raise ValueError(f'{key} must be a nonnegative ordered interval')
        if key in ('mass_scale_range', 'gain_scale_range') and values[0] <= 0:
            raise ValueError(f'{key} must be positive')
    for key, nominal in (('effort_limit_range_nm', 'effort_limit_nm'),
                         ('velocity_limit_range_rad_s', 'velocity_limit_rad_s')):
        if key in config:
            bounds = config[key]
            if (len(bounds) != 2 or not all(math.isfinite(v) and v > 0 for v in bounds)
                    or not bounds[0] <= config[nominal] <= bounds[1]):
                raise ValueError(f'{key} must be positive, ordered and contain its nominal value')
    if 'base_payload' in config:
        payload = config['base_payload']
        if type(payload.get('measured')) is not bool:
            raise ValueError('Payload requires explicit measured boolean')
        if not math.isfinite(payload['mass_kg']) or payload['mass_kg'] < 0:
            raise ValueError('Payload mass must be finite and nonnegative')
        for key in ('center_m', 'box_size_m'):
            values = payload[key]
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError(f'Payload {key} requires three finite values')
        if any(v <= 0 for v in payload['box_size_m']):
            raise ValueError('Payload box dimensions must be positive')
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
    cap = config.get('side_upright_progress_reward_cap', 0.)
    if not math.isfinite(cap) or not 0 <= cap <= 1:
        raise ValueError('Side upright progress reward cap must be finite and within [0, 1]')
    if cap > 0 and 'settled_starts' not in config:
        raise ValueError('Side upright progress shaping requires recorded settled starts')
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
        if cap > 0:
            for side in ('left', 'right'):
                _, x, y, _ = config['settled_starts'][side]['root_pose'][3:]
                if 1 - 2 * (x*x + y*y) >= .5:
                    raise ValueError('Side upright progress shaping requires genuinely fallen side starts')
    if 'reward' in config:
        reward = config['reward']
        if reward.get('type') != 'two_portion':
            raise ValueError('Only the two_portion reward block is supported')
        for key in ('standing_per_s', 'forward_per_m', 'getup_progress_weight', 'effort_weight'):
            value = reward.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Reward {key} must be finite and nonnegative')
        for key in ('standing_ramp_cos', 'forward_gate_cos'):
            bounds = reward.get(key, [])
            if len(bounds) != 2 or not all(math.isfinite(b) for b in bounds) or not -1 <= bounds[0] < bounds[1] <= 1:
                raise ValueError(f'Reward {key} must be an increasing cosine interval within [-1, 1]')
        for key in ('forward_requires_crawl_mode', 'heading_requires_crawl_mode'):
            if key in reward and type(reward[key]) is not bool:
                raise ValueError(f'Reward {key} must be a boolean')
        if 'forward_heading_floor' in reward:
            floor, band = reward['forward_heading_floor'], reward.get('forward_heading_band_deg', [])
            if type(floor) not in (int, float) or not 0 <= floor <= 1:
                raise ValueError('Reward forward_heading_floor must be within [0, 1]')
            if len(band) != 2 or not all(type(b) in (int, float) for b in band) or not 0 <= band[0] < band[1] <= 180:
                raise ValueError('Reward forward_heading_band_deg must be an increasing interval within [0, 180] degrees')
        digest = reward.get('getup_progress_bank_sha256', '')
        if not reward.get('getup_progress_bank') or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Reward needs a get-up progress bank path and its sha256')
    for key in ('side_start_heading_reset_at_recovery', 'side_start_progress_neutral_until_recovery'):
        if key in config and type(config[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    if 'handover' in config:
        validate_handover(config)
    return config


def validate_handover(config):
    """Scripted-prefix handover curriculum: teacher keyframes and audited prefix lengths."""
    handover = config['handover']
    if handover.get('teacher') == 'hybrid':
        # Track B (#15): the hybrid scripted controller drives a random prefix of upright and side starts.
        fraction = handover.get('fraction_of_starts')
        if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('Hybrid handover fraction_of_starts must be within (0, 1]')
        prefix = handover.get('max_prefix_s', {})
        if set(prefix) != {'upright', 'side'} or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in prefix.values()):
            raise ValueError('Hybrid handover max_prefix_s needs positive upright and side durations')
        if not handover.get('controller') or not handover.get('controller_sha256'):
            raise ValueError('Hybrid handover needs a controller config and its hash')
        return
    fraction = handover.get('fraction_of_side_starts')
    if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Handover fraction_of_side_starts must be within (0, 1]')
    if 'settled_starts' not in config or not config['recovery_start_fraction'] > 0:
        raise ValueError('Handover requires recorded settled starts and side starts')
    attempt = handover.get('attempt_seconds')
    if attempt is not None and (type(attempt) not in (int, float) or not math.isfinite(attempt) or attempt <= 0):
        raise ValueError('Handover attempt_seconds must be finite and positive when present')
    scripts = handover.get('scripts', {})
    if set(scripts) != {'left', 'right'}:
        raise ValueError('Handover requires left and right teacher scripts')
    for side, variants in scripts.items():
        if not variants:
            raise ValueError(f'Handover {side} requires at least one teacher script')
        for variant in variants:
            keys = variant['keyframes_rad']
            if len(keys) != 3 or any(len(k) != 6 or not all(math.isfinite(q) for q in k) for k in keys):
                raise ValueError('Each teacher script requires three finite six-joint keyframes')
            steps = variant['prefix_steps']
            if not steps or any(type(k) is not int or k < 1 for k in steps):
                raise ValueError('Teacher prefix steps must be positive integers; k=0 is an official start')
            if 'schedule' in handover and steps != sorted(set(steps)):
                raise ValueError('A scheduled curriculum needs strictly increasing prefix steps (earliest to latest)')
            upright = variant.get('handover_upright')
            if upright is not None and (len(upright) != len(steps) or not all(math.isfinite(c) for c in upright)):
                raise ValueError('handover_upright must give one finite cosine per prefix step')
    schedule = handover.get('schedule')
    if schedule is not None:
        if schedule.get('type') != 'reverse':
            raise ValueError('Only the reverse handover schedule is supported')
        for key in ('uniform_share', 'frontier_share', 'promote_success_rate', 'ema_alpha'):
            value = schedule.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Schedule {key} must be within [0, 1]')
        if not schedule['promote_success_rate'] > 0 or not schedule['ema_alpha'] > 0:
            raise ValueError('Schedule promotion rate and EMA alpha must be positive')
        if type(schedule.get('promote_min_attempts')) is not int or schedule['promote_min_attempts'] < 1:
            raise ValueError('Schedule promote_min_attempts must be a positive integer')


def combine_box_payload(mass, center, inertia, payload):
    """Combine a link and a uniform box using the parallel-axis theorem.

    Both centers and COM inertia matrices use the same link frame. The added
    box axes follow that frame. Returns combined mass, COM and COM inertia.
    This is a mass approximation, not added collision geometry.
    """
    import numpy as np
    center, inertia = np.asarray(center, dtype=float), np.asarray(inertia, dtype=float).reshape(3, 3)
    added = payload['mass_kg']
    total = mass + added
    location = np.asarray(payload['center_m'], dtype=float)
    combined = (mass * center + added * location) / total
    size = np.asarray(payload['box_size_m'], dtype=float)
    box_inertia = np.diag(added / 12. * (np.sum(size * size) - size * size))
    def shift(amount, offset):
        return amount * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    return total, combined, inertia + box_inertia + shift(mass, center-combined) + shift(added, location-combined)
