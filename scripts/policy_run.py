#!/usr/bin/env python3
"""Run an exported policy closed-loop on the SO-101 through the servo bus directly (no LeRobot).

The loop reproduces the training environment's action path at the contract rate (50 Hz): observation =
[q - home, filtered velocity, base-frame gravity, base-frame gyro, (applied target - home) / scale], action ->
contract.position_target (clip, limits, slew), goal ticks via the measured joint map, clipped to the servos'
calibrated windows; the clipped target is what feeds back as the previous applied target. The velocity filter
is updated once per policy step, as in training. Joints the map does not list (a gripper without a servo)
are virtual: they read back as their last applied target.

--mode shadow (default): reads only, torque stays off. Each step uses the measured pose as the previous
target, so every logged action is the policy's one-step answer to the real sensors (compare with the sim's
observation at the same posture). Also writes a JSONL that scripts/deploy.py --replay can read.

--mode live: WRITES to the servos, only with the operator's explicit OK for that run, cutoff in reach.
Policies whose manifest has physics_validated false additionally need --allow-unvalidated-physics, and the
operator must type RUN POLICY. Start-up as in actuator_log.py (goal = present read back, Acceleration,
torque on), then a --ramp-s ramp to the contract home pose and a policy start from the sim's reset state
(previous target = home, filtered velocity 0). Every bus cycle between steps is checked against the #12
envelope; a stale IMU sample, a missed deadline or a contract error aborts. Abort, Ctrl-C or the end releases
torque (the end first holds the last goal until Enter).
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actuator_log import ACCELERATION, GOAL_POSITION, TORQUE_ENABLE, Recorder, motion_window, stdin_has_line  # noqa: E402
from le_snail.actuator_id import TICKS_PER_RAD, JointTickMap, SafetyLimits, ramp  # noqa: E402
from le_snail.bundle import load_bundle  # noqa: E402
from le_snail.contract import VelocityEstimator  # noqa: E402
from le_snail.hardware import Bno085Imu, ImuSerial  # noqa: E402
from le_snail.servo_bus import STATE_BLOCK, ServoBus, dump_servos, open_bus_port  # noqa: E402

MAX_SECONDS = 30.0
MAX_RAMP_SPEED_RAD_S = 0.5   # the start ramp is stretched to stay under this, whatever --ramp-s says


def proper_rotation(matrix):
    """Nearest proper rotation to a measured mounting matrix; refuse anything that is not close to one."""
    r = np.asarray(matrix, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or np.linalg.det(r) < 0.9 or not np.allclose(r @ r.T, np.eye(3), atol=1e-2):
        raise SystemExit(f'imu_rotation_base_from_sensor is not a proper rotation: {r.tolist()}')
    u, _, vt = np.linalg.svd(r)
    return u @ vt


def release_torque(recorder, ids, attempts=5):
    """Broadcast Torque_Enable = 0 until every servo reads it back as 0 (broadcasts get no reply and can be lost)."""
    for _ in range(attempts):
        try:
            recorder.write('Torque_Enable', {sid: 0 for sid in ids}, TORQUE_ENABLE)
            replies = [recorder.bus.read(sid, *TORQUE_ENABLE) for sid in ids]
            if all(r is not None and r[1][0] == 0 for r in replies):
                return 'confirmed'
        except Exception:  # keep trying; the caller still closes the ports
            pass
    return 'NOT CONFIRMED: cut the 12 V supply'


def first_imu_sample(imu, max_age_s, wait_s=2.0):
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return imu.read(max_age_s)
        except RuntimeError as error:
            if time.monotonic() > deadline:
                raise SystemExit(f'IMU not streaming fresh samples: {error}')
            time.sleep(0.02)


def open_imu(hardware, rotation):
    """hardware['imu_type']: 'xsens' (default; Xbus over USB serial) or 'bno085' (I2C, bus imu_i2c_bus)."""
    if hardware.get('imu_type', 'xsens') == 'bno085':
        return Bno085Imu(hardware['imu_i2c_bus'], rotation)
    return ImuSerial(hardware['imu_port'], rotation, hardware.get('imu_baudrate') or 115200)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Steps:
    """Per-policy-step log."""
    def __init__(self):
        self.rows = {k: [] for k in ('step_time_s', 'joint_rad', 'velocity', 'gravity', 'gyro', 'observation', 'action',
                                     'target_rad', 'goal_ticks', 'clipped', 'imu_seq', 'imu_age_s', 'state_age_s')}

    def add(self, **values):
        for key, value in values.items():
            self.rows[key].append(value)

    def arrays(self):
        return {k: np.array(v) for k, v in self.rows.items()}

    def jsonl(self, path):
        with path.open('w') as stream:
            for t, q, g, w in zip(*(self.rows[k] for k in ('step_time_s', 'joint_rad', 'gravity', 'gyro'))):
                stream.write(json.dumps({'time_s': t, 'joint_rad': list(map(float, q)), 'gravity_base': list(map(float, g)),
                                         'gyro_base_rad_s': list(map(float, w))}) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hardware', type=Path, required=True,
                        help='JSON: servo_port, joint_map, imu_rotation_base_from_sensor, imu_type (xsens: imu_port, imu_baudrate; bno085: imu_i2c_bus)')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--mode', choices=('shadow', 'live'), default='shadow')
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--ramp-s', type=float, default=5.0, help='live: ramp from the present pose to home')
    parser.add_argument('--margin-ticks', type=int, default=100)
    parser.add_argument('--acceleration', type=int, default=254)
    parser.add_argument('--settle-s', type=float, default=1.0, help='live: hold home before the first policy step')
    parser.add_argument('--max-feedback-age-s', type=float, default=0.04, help='servo/IMU sample age limit (2 policy steps)')
    parser.add_argument('--timeout-ms', type=float, default=10.0)
    parser.add_argument('--allow-unvalidated-physics', action='store_true',
                        help='live: run a policy whose physics is not validated against the real robot (operator decision)')
    parser.add_argument('--out', type=Path, default=Path('runs/policy-runs'))
    args = parser.parse_args()
    if not 0 < args.seconds <= MAX_SECONDS:
        parser.error(f'--seconds must be in (0, {MAX_SECONDS:g}]')
    live = args.mode == 'live'
    hardware = json.loads(args.hardware.read_text())
    if hardware.get('imu_rotation_base_from_sensor') is None:
        raise SystemExit(f'{args.hardware}: imu_rotation_base_from_sensor is not measured for this arm; measure the IMU mounting first')
    rotation = proper_rotation(hardware['imu_rotation_base_from_sensor'])
    joint_map_path = Path(hardware['joint_map'])
    joint_map = json.loads(joint_map_path.read_text())
    contract, manifest = load_bundle(args.bundle)
    if live and manifest.get('physics_validated') is not True and not args.allow_unvalidated_physics:
        raise SystemExit('This policy\'s physics is not validated against the real robot; live needs '
                         '--allow-unvalidated-physics (operator decision) or a validated export')
    jmap = JointTickMap(joint_map['joints'], contract.joint_names)
    home = np.asarray(contract.home_rad, dtype=float)
    dt, limits = contract.dt, SafetyLimits()

    import torch
    policy = torch.jit.load(str(args.bundle / 'policy.pt'), map_location='cpu').eval()
    torch.set_num_threads(1)
    with torch.inference_mode():
        policy(torch.zeros((1, 24)))

    imu = open_imu(hardware, rotation)
    port = open_bus_port(hardware['servo_port'])
    bus = ServoBus(port, timeout_s=args.timeout_ms / 1000, allow_writes=live)
    recorder = Recorder(bus, jmap.ids, limits)
    steps = Steps()
    servos_before = servos_after = None
    torque_on, finished, abort, interrupted, torque_release = False, False, None, False, None
    try:
        servos_before = dump_servos(bus, jmap.ids)
        expected = joint_map.get('valid_while_eeprom_homing_offset', {})
        found = {str(sid): servos_before[str(sid)]['registers']['Homing_Offset'] for sid in jmap.ids}
        wrong = {sid: (found[sid], expected[sid]) for sid in expected if sid in found and found[sid] != expected[sid]}
        if wrong:
            raise SystemExit(f'Servo Homing_Offset differs from the joint map {joint_map_path} (id: (servo, map)) {wrong}; '
                             'the arm was recalibrated or swapped, so the map does not apply')
        planned, lower, upper = motion_window(servos_before, jmap.ids, args.margin_ticks)
        lower, upper = np.array(lower), np.array(upper)
        # Hard stops the servo EEPROM does not know about (e.g. a full-turn roll with a cable stop).
        for k, name in enumerate(n for n in contract.joint_names if n in joint_map['joints']):
            if 'window_ticks' in joint_map['joints'][name]:
                lo, hi = joint_map['joints'][name]['window_ticks']
                lower[k], upper[k] = max(lower[k], lo), min(upper[k], hi)
        home_ticks = jmap.to_ticks(home)
        if np.any(home_ticks < lower) or np.any(home_ticks > upper):
            raise SystemExit(f'Home pose {home_ticks.tolist()} lies outside the servo windows {lower.tolist()}..{upper.tolist()}')
        sample = first_imu_sample(imu, args.max_feedback_age_s)
        print(f'IMU gravity in the base frame {np.round(sample["gravity"], 3).tolist()} (upright and level: [0, 0, -1])')
        print(f'policy {args.bundle} ({manifest["policy_sha256"][:12]}), {contract.policy_hz:g} Hz, mode {args.mode}; '
              f'servo ids {jmap.ids}, virtual joints {jmap.virtual}; home ticks {home_ticks.tolist()}')
        if live:
            print(f'writes: Goal_Position = present, Acceleration = {args.acceleration}, Torque_Enable = 1 on {jmap.ids}; '
                  f'{args.ramp_s:g} s ramp {planned.tolist()} -> home, then {args.seconds:g} s of policy'
                  + ('' if manifest.get('physics_validated') else '  [physics NOT validated: operator override]'))
            if input('Operator present, cutoff in reach, robot placed for this run? Type RUN POLICY to start: ') != 'RUN POLICY':
                raise SystemExit('not confirmed')
            present = recorder.positions()
            if np.any(np.abs(present - planned) > limits.max_start_drift_ticks):
                raise SystemExit(f'The robot moved since the plan ({(present - planned).tolist()} ticks); not starting')
            recorder.write_checked('Goal_Position', dict(zip(jmap.ids, present.tolist())), GOAL_POSITION)
            recorder.write_checked('Acceleration', {sid: args.acceleration for sid in jmap.ids}, ACCELERATION)
            torque_on = True
            recorder.write('Torque_Enable', {sid: 1 for sid in jmap.ids}, TORQUE_ENABLE)

        gc.disable()  # see actuator_log.py: cyclic GC pauses grow with the logs; the loop makes no cycles
        if live:
            distance_rad = np.abs(home_ticks - present).max() / TICKS_PER_RAD
            ramp_s = max(args.ramp_s, distance_rad / MAX_RAMP_SPEED_RAD_S, dt)
            goals = ramp(present, home_ticks, ramp_s, contract.policy_hz)
            goals = np.vstack([goals, np.repeat(home_ticks[None], round(args.settle_s * contract.policy_hz), axis=0)])
            t0 = time.monotonic()
            for k, goal in enumerate(goals):
                recorder.command(goal)
                while abort is None and time.monotonic() - t0 < (k + 1) * dt:
                    abort = recorder.cycle()
                if abort:
                    break
        target = home.copy() if live else None
        estimator = VelocityEstimator(contract.velocity_filter_alpha)
        start = time.monotonic()
        k, last_state_ns = 0, None
        cycle_budget = args.timeout_ms / 1000 + 0.004   # worst-case bus cycle: skip cycles that could overrun the slot
        while abort is None and (now := time.monotonic()) - start < args.seconds:
            slot = start + k * dt
            if now < slot:
                if slot - now > cycle_budget:
                    abort = recorder.cycle()
                else:
                    time.sleep(slot - now)   # never spin: the IMU reader thread needs the GIL
                continue
            abort = recorder.cycle()
            if abort:
                break
            if recorder.last_complete is None or recorder.last_complete[0] == last_state_ns:
                abort = f'no fresh complete servo state for step {k}'
                break
            state_ns, states = recorder.last_complete
            last_state_ns = state_ns
            state_age = time.monotonic() - state_ns / 1e9
            if state_age > args.max_feedback_age_s:
                abort = f'servo state {state_age * 1e3:.0f} ms old'
                break
            try:
                sample = imu.read(args.max_feedback_age_s)
            except RuntimeError as error:
                abort = f'IMU: {error}'
                break
            ticks = np.array([states[sid]['position'] for sid in jmap.ids])
            q = jmap.to_rad(ticks, virtual_rad=target if target is not None else home)
            velocity = estimator.update(q, state_ns / 1e9)
            previous = target if live else q          # shadow: one-step answer from the measured pose
            try:
                observation = contract.observation(q, velocity, sample['gravity'], sample['gyro'], previous)
                with torch.inference_mode():
                    action = policy(torch.from_numpy(observation)[None])[0].numpy().astype(float)
                # A limp robot can rest beyond the contract limits (e.g. lift on its stop); shadow clips the slew base.
                base = previous if live else np.clip(previous, contract.lower_rad, contract.upper_rad)
                candidate = contract.position_target(action, base, dt)
            except ValueError as error:
                abort = f'contract: {error}'
                break
            raw_goal = jmap.to_ticks(candidate)
            goal = np.clip(raw_goal, lower, upper)
            # Tick rounding can land a hair outside a contract limit; the next position_target would reject it.
            applied = np.clip(jmap.to_rad(goal, virtual_rad=candidate), contract.lower_rad, contract.upper_rad)
            if live:
                late = time.monotonic() - slot
                if late > 0.5 * dt:
                    abort = f'control deadline missed: step {k} command {late * 1e3:.1f} ms after its slot'
                    break
                recorder.command(goal)
                target = applied
            steps.add(step_time_s=now - start, joint_rad=q, velocity=velocity, gravity=sample['gravity'], gyro=sample['gyro'],
                      observation=observation, action=action, target_rad=applied, goal_ticks=goal, clipped=goal != raw_goal,
                      imu_seq=sample['seq'], imu_age_s=time.monotonic() - sample['sample_at'], state_age_s=state_age)
            k += 1
        if torque_on and abort is None:
            print('Policy finished, holding the last goal. Support the robot, then press Enter to release torque.')
            while not stdin_has_line():
                if abort := recorder.cycle():
                    break
            if abort is None:
                sys.stdin.readline()
        finished = abort is None
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:  # anything unexpected still releases torque and saves the log
        abort = abort or f'error: {error!r}'
    finally:
        gc.enable()
        try:
            if torque_on:
                torque_release = release_torque(recorder, jmap.ids)
                print(f'torque release {torque_release}')
        finally:
            try:
                servos_after = dump_servos(bus, jmap.ids)
            except Exception:
                servos_after = None
            finally:
                port.close()
                imu.close()

    utc = datetime.now(timezone.utc)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f'{args.mode}-{args.bundle.parent.name}-{utc.strftime("%Y%m%dT%H%M%SZ")}'
    summary = recorder.save(stem, steps.arrays())
    steps.jsonl(stem.with_suffix('.jsonl'))
    meta = {'utc': utc.isoformat(), 'mode': args.mode, 'bundle': str(args.bundle), 'policy_sha256': manifest['policy_sha256'],
            'contract_fingerprint': contract.fingerprint, 'physics_validated': manifest.get('physics_validated'),
            'physics_override': bool(live and args.allow_unvalidated_physics and not manifest.get('physics_validated')),
            'hardware': hardware, 'joint_map_sha256': sha256(joint_map_path), 'servo_ids': jmap.ids,
            'virtual_joints': jmap.virtual, 'seconds': args.seconds, 'ramp_s': args.ramp_s, 'margin_ticks': args.margin_ticks,
            'window_ticks': [lower.tolist(), upper.tolist()] if servos_before else None,
            'max_feedback_age_s': args.max_feedback_age_s, 'settle_s': args.settle_s, 'imu_rotation_used': rotation.tolist(), 'state_block': list(STATE_BLOCK), 'safety_limits': limits.__dict__,
            'policy_steps': len(steps.rows['step_time_s']), 'torque_release': torque_release, 'abort': abort, 'interrupted': interrupted, 'finished': finished,
            'summary': summary,
            'writes': [{'register': n, 'sent_ns': s, 'flushed_ns': f, 'values': v} for n, s, f, v in recorder.events],
            'register_dump_before': servos_before, 'register_dump_after': servos_after}
    stem.with_suffix('.json').write_text(json.dumps(meta, indent=1, default=list) + '\n')
    print(json.dumps({'abort': abort, 'interrupted': interrupted, 'policy_steps': meta['policy_steps'], **summary}, indent=1))
    print(f'wrote {stem}.npz, .json and .jsonl')


if __name__ == '__main__':
    main()
