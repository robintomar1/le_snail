#!/usr/bin/env python3
"""Stage 1 logger of #12: timestamped raw servo state, optionally while replaying a goal sequence.

Each cycle is one SYNC_READ of the Present_* block (addresses 56-70) on the listed ids, stamped with
time.monotonic_ns() immediately before the request and after the last reply. Raw bytes are logged;
convert offline. A register dump (Stage 0) is stored with every log.

Dry run (default, no --motion): reads only, to measure the achievable bus rate and loss.

Motion run (--motion FILE --allow-motion, then a typed confirmation): WRITES to the servos. Only with the
operator's explicit OK for that run, the 12 V cutoff in reach, the base clamped and the arm suspended.
The motion file holds goal offsets in ticks about a start pose (--start-ticks, default the present pose),
reached by a --ramp-s interpolation from the present pose; the start pose and every motion goal must lie
inside the servos' calibrated position limits minus --margin-ticks. Start-up writes, in order:
Goal_Position = present position, Acceleration (only if LeRobot configure() would set it), Torque_Enable = 1.
Any envelope violation, lost feedback or Ctrl-C releases torque at once; a normal end holds the last goal
until the operator presses Enter, then releases torque.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from le_snail.actuator_id import (
    SafetyLimits, abort_reason, closest_recent_goal, decode_state, joint_window, plan_motion, rate_summary,
)
from le_snail.servo_bus import STATE_BLOCK, STS3215_REGISTERS, ServoBus, configure_differences, open_bus_port

sys.path.insert(0, str(Path(__file__).resolve().parent))
from servo_dump import GRIPPER_ID, dump_arm  # noqa: E402

GOAL_POSITION, ACCELERATION, TORQUE_ENABLE = (STS3215_REGISTERS[name] for name in ('Goal_Position', 'Acceleration', 'Torque_Enable'))
SETTLE_S = 0.5  # window for the tracking-error reference (see closest_recent_goal)


def load_motion(path, ids):
    motion = np.load(path)
    offsets, rate_hz = motion['offsets_ticks'].astype(int), float(motion['rate_hz'])
    if motion['ids'].tolist() != list(ids):
        raise SystemExit(f'Motion ids {motion["ids"].tolist()} differ from --ids {list(ids)}')
    return offsets, rate_hz, str(motion['name'])


def check_start(servos, ids, offsets, margin_ticks, start_ticks, ramp_s, rate_hz):
    present, lower, upper = [], [], []
    for servo_id in ids:
        servo = servos[str(servo_id)]
        if not servo.get('answered') or servo['failed_ranges']:
            raise SystemExit(f'id {servo_id}: incomplete register dump; not starting')
        unexpected = set(servo['lerobot_configure_would_change']) - {'Acceleration'}
        if unexpected:
            raise SystemExit(f'id {servo_id}: EEPROM differs from the LeRobot deployment state: {sorted(unexpected)}')
        registers = servo['registers']
        if registers['Torque_Enable']:
            raise SystemExit(f'id {servo_id}: torque already enabled; not starting')
        present.append(registers['Present_Position'])
        lo, hi = joint_window(registers, margin_ticks)
        lower.append(lo)
        upper.append(hi)
    start = present if start_ticks is None else start_ticks
    if len(start) != len(ids):
        raise SystemExit('--start-ticks needs one value per id')
    try:
        goals = plan_motion(np.array(present), np.array(start), offsets, lower, upper, ramp_s, rate_hz)
    except ValueError as error:
        raise SystemExit(f'{error} (window = calibrated limits minus {margin_ticks} ticks: '
                         f'lower {lower}, upper {upper}; present {present})')
    return np.array(present), goals


def release(bus, ids):
    try:
        bus.sync_write({servo_id: 0 for servo_id in ids}, *TORQUE_ENABLE)
    finally:
        print('torque released')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('port')
    parser.add_argument('--ids', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--seconds', type=float, default=10.0, help='dry-run duration')
    parser.add_argument('--read-rate-hz', type=float, default=0.0, help='pace the reads; 0 = as fast as the bus allows')
    parser.add_argument('--timeout-ms', type=float, default=10.0, help='per-transaction reply timeout')
    parser.add_argument('--out', type=Path, default=Path('runs/actuator-id/stage1'))
    parser.add_argument('--motion', type=Path, help='.npz with offsets_ticks (samples, ids), rate_hz, ids, name')
    parser.add_argument('--allow-motion', action='store_true', help='permit servo writes for --motion')
    parser.add_argument('--margin-ticks', type=int, default=100)
    parser.add_argument('--start-ticks', type=int, nargs='+', help='pose to ramp to before the motion (default: present)')
    parser.add_argument('--ramp-s', type=float, default=3.0)
    parser.add_argument('--acceleration', type=int, default=254, help='LeRobot configure() Acceleration value')
    args = parser.parse_args()
    if args.motion and not args.allow_motion:
        raise SystemExit('--motion writes to the servos; add --allow-motion once the operator has approved this run')
    limits = SafetyLimits()

    port = open_bus_port(args.port)
    bus = ServoBus(port, timeout_s=args.timeout_ms / 1000, allow_writes=bool(args.motion))
    servos, torque_on, finished, abort, interrupted = None, False, False, None, False
    sent, received, lost, raw, valid, cmd_ns, cmd_goals = [], [], [], [], [], [], []
    try:
        servos = dump_arm(bus, args.ids)
        if args.motion:
            offsets, rate_hz, name = load_motion(args.motion, args.ids)
            present, goals = check_start(servos, args.ids, offsets, args.margin_ticks, args.start_ticks, args.ramp_s, rate_hz)
            duration = len(goals) / rate_hz
            print(f'motion {name}: {args.ramp_s:g} s ramp {present.tolist()} -> {goals[len(goals) - len(offsets)].tolist()}, '
                  f'then {len(offsets)} goals at {rate_hz:g} Hz ({duration:.1f} s total) on ids {args.ids}; '
                  f'max |offset| per id {np.abs(offsets).max(axis=0).tolist()} ticks')
            if input('Operator present, cutoff in reach, base clamped, arm suspended? Type MOVE to start: ') != 'MOVE':
                raise SystemExit('not confirmed')
            bus.sync_write(dict(zip(args.ids, present.tolist())), *GOAL_POSITION)  # Goal_Position reads 0 while torque is off
            needs_acceleration = {sid: args.acceleration for sid in args.ids
                                  if 'Acceleration' in servos[str(sid)]['lerobot_configure_would_change']}
            if needs_acceleration:
                bus.sync_write(needs_acceleration, *ACCELERATION)
            torque_on = True
            bus.sync_write({servo_id: 1 for servo_id in args.ids}, *TORQUE_ENABLE)
        else:
            goals, rate_hz, duration, name = None, None, args.seconds, 'dry-run'

        recent = deque()
        next_goal, consecutive_lost = 0, 0
        t0 = time.monotonic_ns()
        while True:
            now = time.monotonic_ns()
            elapsed = (now - t0) / 1e9
            if elapsed >= duration:
                break
            if goals is not None and next_goal < len(goals) and elapsed >= next_goal / rate_hz:
                goal = goals[next_goal]
                cmd_ns.append(bus.sync_write(dict(zip(args.ids, goal.tolist())), *GOAL_POSITION))
                cmd_goals.append(goal)
                recent.append((cmd_ns[-1], goal))
                next_goal += 1
            result = bus.sync_read(args.ids, *STATE_BLOCK)
            sent.append(result.sent_ns)
            received.append(result.received_ns)
            lost.append(len(result.lost))
            raw.append([result.replies.get(sid, (0, bytes(STATE_BLOCK[1])))[1] for sid in args.ids])
            valid.append([sid in result.replies for sid in args.ids])
            consecutive_lost = consecutive_lost + 1 if result.lost else 0
            states = {sid: decode_state(params) for sid, (_, params) in result.replies.items()}
            reference = None
            if recent:
                while len(recent) > 1 and recent[1][0] <= result.received_ns - SETTLE_S * 1e9:
                    recent.popleft()
                if result.received_ns - recent[0][0] >= SETTLE_S * 1e9 and len(states) == len(args.ids):
                    positions = [states[sid]['position'] for sid in args.ids]
                    reference = dict(zip(args.ids, closest_recent_goal([g for _, g in recent], positions).tolist()))
            abort = abort_reason(states, limits, reference, consecutive_lost)
            if abort:
                break
            if args.read_rate_hz:
                target = t0 + len(sent) * 1e9 / args.read_rate_hz
                while time.monotonic_ns() < target:
                    pass
        finished = abort is None
        if torque_on and abort:
            release(bus, args.ids)
            torque_on = False
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if torque_on:
            if finished:
                try:
                    input('Motion finished, holding the last goal. Support the arm, then press Enter to release torque: ')
                except (KeyboardInterrupt, EOFError):
                    pass
            release(bus, args.ids)
        port.close()

    utc = datetime.now(timezone.utc)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f'{name}-{utc.strftime("%Y%m%dT%H%M%SZ")}'
    np.savez_compressed(
        stem.with_suffix('.npz'), ids=np.array(args.ids), sent_ns=np.array(sent, dtype=np.int64),
        received_ns=np.array(received, dtype=np.int64), lost=np.array(lost),
        state_raw=np.frombuffer(b''.join(b''.join(row) for row in raw), dtype=np.uint8).reshape(len(raw), len(args.ids), STATE_BLOCK[1]),
        valid=np.array(valid, dtype=bool).reshape(len(valid), len(args.ids)),
        command_sent_ns=np.array(cmd_ns, dtype=np.int64), command_goal_ticks=np.array(cmd_goals, dtype=int).reshape(len(cmd_goals), len(args.ids)))
    summary = rate_summary(sent, received, lost) if sent else {'cycles': 0}
    meta = {'utc': utc.isoformat(), 'mode': 'motion' if args.motion else 'dry-run', 'motion': name,
            'motion_file': str(args.motion) if args.motion else None, 'port': args.port, 'ids': args.ids,
            'timeout_ms': args.timeout_ms, 'read_rate_hz': args.read_rate_hz, 'margin_ticks': args.margin_ticks,
            'start_ticks': args.start_ticks, 'ramp_s': args.ramp_s,
            'state_block': list(STATE_BLOCK), 'safety_limits': limits.__dict__, 'abort': abort,
            'interrupted': interrupted, 'summary': summary, 'gripper_id': GRIPPER_ID,
            'register_dump_at_start': servos}
    stem.with_suffix('.json').write_text(json.dumps(meta, indent=1, default=list) + '\n')
    print(json.dumps({'abort': abort, 'interrupted': interrupted, **summary}, indent=1))
    print(f'wrote {stem}.npz and .json')


if __name__ == '__main__':
    main()
