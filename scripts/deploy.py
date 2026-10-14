#!/usr/bin/env python3
"""Replay a policy offline or explicitly run it on a calibrated SO-101."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from le_snail.bundle import load_bundle
from le_snail.contract import VelocityEstimator
from le_snail.hardware import ImuSerial, validate_hardware_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--replay', type=Path, help='JSONL joint/IMU records; no hardware is opened')
    mode.add_argument('--live', action='store_true', help='Enable actual motor commands')
    parser.add_argument('--hardware', type=Path)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path to preserve earlier logs')
    if not 0 < args.seconds <= 120:
        parser.error('--seconds must be in (0, 120]')
    contract, manifest = load_bundle(args.bundle)
    import torch
    # CPU is sufficient for this small MLP and avoids adding GPU scheduling jitter.
    policy = torch.jit.load(str(args.bundle / 'policy.pt'), map_location='cpu').eval()
    torch.set_num_threads(1)
    estimator = VelocityEstimator(contract.velocity_filter_alpha)
    target = np.asarray(contract.home_rad, dtype=float)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.replay:
        with args.replay.open() as source, args.output.open('x') as sink, torch.inference_mode():
            previous_time = None
            for line in source:
                record = json.loads(line)
                now = float(record['time_s'])
                dt = contract.dt if previous_time is None else now - previous_time
                velocity = estimator.update(record['joint_rad'], now)
                observation = contract.observation(record['joint_rad'], velocity, record['gravity_base'], record['gyro_base_rad_s'], target)
                action = policy(torch.from_numpy(observation)[None])[0].numpy()
                target = contract.position_target(action, target, dt)
                sink.write(json.dumps({'time_s': now, 'target_rad': target.tolist()}) + '\n')
                previous_time = now
        return
    if args.hardware is None:
        parser.error('--live requires --hardware')
    config = json.loads(args.hardware.read_text())
    if manifest['physics_validated'] is not True or manifest['stage'] != 'robust':
        parser.error('Live operation requires a robust-stage policy with validated physics; provisional exports are for simulation/replay')
    mapping = validate_hardware_settings(config, contract)
    calibration_dir = Path(config['calibration_dir']).expanduser()
    if not (calibration_dir / f"{config['robot_id']}.json").is_file():
        parser.error('Existing LeRobot calibration file is missing; this runner never recalibrates automatically')
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower
    robot = SO101Follower(SO101FollowerConfig(
        port=config['robot_port'], id=config['robot_id'], calibration_dir=calibration_dir,
        use_degrees=True, cameras={}, disable_torque_on_disconnect=True,
    ))
    imu = None
    connected = False
    try:
        # Start sensors and warm up inference before enabling the robot connection.
        imu = ImuSerial(config['imu_port'], config['imu_rotation_base_from_sensor'], config['imu_baudrate'])
        with torch.inference_mode():
            policy(torch.zeros((1, 24)))
        time.sleep(0.2)
        imu.read(config['max_feedback_age_s'])
        robot.connect(calibrate=False)
        connected = True
        if not robot.is_calibrated:
            raise RuntimeError('Robot calibration differs from the recorded calibration')
        joint = mapping.to_radians(robot.get_observation())
        if np.max(np.abs(joint - contract.home_rad)) > config['start_tolerance_rad']:
            raise RuntimeError('Place the supported robot near the recorded start posture before enabling this policy')
        if np.any(joint < contract.lower_rad) or np.any(joint > contract.upper_rad):
            raise RuntimeError('Initial position is outside policy limits')
        # Initialize to measured position; the controller does not jump to home.
        target = joint.copy()
        estimator.update(joint, time.monotonic())
        start = next_tick = time.monotonic()
        previous_tick = None
        with args.output.open('x') as log, torch.inference_mode():
            step = 0
            while time.monotonic() - start < args.seconds:
                tick = time.monotonic()
                if previous_tick is not None and tick - previous_tick > contract.dt * 1.5:
                    raise RuntimeError('Control deadline missed')
                joint = mapping.to_radians(robot.get_observation())
                sampled_at = time.monotonic()
                if sampled_at - tick > config['max_feedback_age_s']:
                    raise RuntimeError('Joint feedback transaction was too slow')
                sample = imu.read(config['max_feedback_age_s'])
                if np.max(np.abs(target - joint)) > config['max_tracking_error_rad']:
                    raise RuntimeError('Joint tracking error exceeds the configured bound')
                if step % round(contract.policy_hz) == 0:
                    temperature = robot.bus.sync_read('Present_Temperature')
                    voltage = robot.bus.sync_read('Present_Voltage')
                    if set(temperature) != set(contract.joint_names) or set(voltage) != set(contract.joint_names):
                        raise RuntimeError('Incomplete servo telemetry')
                    if not all(np.isfinite(v) and v <= config['max_temperature_c'] for v in temperature.values()):
                        raise RuntimeError('Servo temperature limit or telemetry fault')
                    lo, hi = config['voltage_range_v']
                    volts_per_unit = config['voltage_v_per_register_unit']
                    if not all(np.isfinite(v) and lo <= v * volts_per_unit <= hi for v in voltage.values()):
                        raise RuntimeError('Servo voltage limit or telemetry fault')
                velocity = estimator.update(joint, sampled_at)
                observation = contract.observation(joint, velocity, sample['gravity'], sample['gyro'], target)
                action = policy(torch.from_numpy(observation)[None])[0].numpy()
                dt = contract.dt if previous_tick is None else tick - previous_tick
                candidate = contract.position_target(action, target, dt)
                if np.max(np.abs(candidate - joint)) > config['max_tracking_error_rad']:
                    raise RuntimeError('Proposed command exceeds joint tracking bound')
                if time.monotonic() - tick > contract.dt or time.monotonic() - sample['sample_at'] > config['max_feedback_age_s']:
                    raise RuntimeError('Feedback/inference deadline missed; no command sent')
                actual = robot.send_action(mapping.to_native(candidate))
                target = mapping.to_radians(actual)
                log.write(json.dumps({'time_s': sampled_at-start, 'joint_rad': joint.tolist(), 'gravity_base': sample['gravity'].tolist(), 'gyro_base_rad_s': sample['gyro'].tolist(), 'target_rad': target.tolist(), 'imu_seq': sample['seq']}) + '\n')
                log.flush()
                previous_tick = tick
                step += 1
                next_tick += contract.dt
                time.sleep(max(0., next_tick-time.monotonic()))
    finally:
        # Torque disable is best-effort; a physical power cutoff remains the operator's stop.
        try:
            if connected:
                robot.disconnect()
            elif robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=True)
        finally:
            if imu is not None:
                imu.close()


if __name__ == '__main__':
    main()
