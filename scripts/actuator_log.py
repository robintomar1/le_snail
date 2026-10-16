#!/usr/bin/env python3
"""Stage 1 logger of #12: timestamped raw servo state, optionally while replaying a goal sequence.

Each cycle is one SYNC_READ of the Present_* block (addresses 56-70) on the listed ids, stamped with
time.monotonic_ns() immediately before the request and after the last reply. Raw bytes are logged;
convert offline. A register dump (Stage 0) is stored with every log, taken before and after the run.

Dry run (default, no --motion): reads only, to measure the achievable bus rate and loss.

Motion run (--motion FILE --allow-motion, then a typed confirmation): WRITES to the servos. Only with the
operator's explicit OK for that run, the 12 V cutoff in reach, the base clamped and the arm suspended.
The motion file holds goal offsets in ticks about a start pose (--start-ticks, default the present pose),
reached by a --ramp-s interpolation from the present pose; the start pose and every motion goal must lie
inside the servos' calibrated position limits minus --margin-ticks. After the confirmation the pose is
re-read (it must not have moved more than the start tolerance since the plan) and the plan is rebuilt
from it. Start-up writes, in order, each read back before the next: Goal_Position = present position,
Acceleration (only where LeRobot configure() would set it; announced before the confirmation),
Torque_Enable = 1. Any envelope violation, lost feedback or Ctrl-C releases torque at once. A normal end
holds the last goal, still checking the envelope, until the operator presses Enter, then releases torque.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import select
import sys
import time

import numpy as np

from le_snail.actuator_id import (
    SafetyLimits, abort_reason, closest_recent_goal, decode_state, joint_window, plan_motion, rate_summary,
)
from le_snail.servo_bus import STATE_BLOCK, STS3215_REGISTERS, ServoBus, dump_servos, open_bus_port

GOAL_POSITION, ACCELERATION, TORQUE_ENABLE = (STS3215_REGISTERS[name] for name in ('Goal_Position', 'Acceleration', 'Torque_Enable'))
SETTLE_S = 0.5  # window for the tracking-error reference (see closest_recent_goal)


def load_motion(path, ids):
    motion = np.load(path)
    offsets, rate_hz = motion['offsets_ticks'].astype(int), float(motion['rate_hz'])
    if motion['ids'].tolist() != list(ids):
        raise SystemExit(f'Motion ids {motion["ids"].tolist()} differ from --ids {list(ids)}')
    return offsets, rate_hz, str(motion['name'])


def motion_window(servos, ids, margin_ticks):
    """Check the dump allows a run; return the present pose and the goal window."""
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
    return np.array(present), lower, upper


def plan(present, start_ticks, offsets, lower, upper, ramp_s, rate_hz):
    start = present if start_ticks is None else np.array(start_ticks)
    if len(start) != len(present):
        raise SystemExit('--start-ticks needs one value per id')
    try:
        return plan_motion(present, start, offsets, lower, upper, ramp_s, rate_hz)
    except ValueError as error:
        raise SystemExit(f'{error} (window = calibrated limits minus margin: lower {lower}, upper {upper}; '
                         f'present {present.tolist()})')


class Recorder:
    """Bus cycles, commands and start-up writes of one run, with the envelope check per cycle."""
    def __init__(self, bus, ids, limits):
        self.bus, self.ids, self.limits = bus, list(ids), limits
        self.sent, self.received, self.lost, self.raw, self.valid = [], [], [], [], []
        self.commands, self.events = [], []  # (sent_ns, flushed_ns, goal) / (name, sent_ns, flushed_ns, values)
        self.recent, self.consecutive_lost = deque(), 0

    def write(self, name, values, register):
        sent, flushed = self.bus.sync_write(values, *register)
        self.events.append((name, sent, flushed, values))

    def write_checked(self, name, values, register):
        """Write, then READ the register back on every id; refuse to continue on any mismatch."""
        self.write(name, values, register)
        for servo_id, value in values.items():
            reply = self.bus.read(servo_id, *register)
            if reply is None or int.from_bytes(reply[1], 'little') != value:
                raise SystemExit(f'id {servo_id}: {name} read back {reply} instead of {value}')

    def command(self, goal):
        sent, flushed = self.bus.sync_write(dict(zip(self.ids, goal.tolist())), *GOAL_POSITION)
        self.commands.append((sent, flushed, goal))
        self.recent.append((sent, goal))

    def cycle(self):
        """One state read; returns an abort message or None."""
        result = self.bus.sync_read(self.ids, *STATE_BLOCK)
        self.sent.append(result.sent_ns)
        self.received.append(result.received_ns)
        self.lost.append(len(result.lost))
        self.raw.append(b''.join(result.replies.get(sid, (0, bytes(STATE_BLOCK[1])))[1] for sid in self.ids))
        self.valid.append([sid in result.replies for sid in self.ids])
        self.consecutive_lost = self.consecutive_lost + 1 if result.lost else 0
        states = {sid: decode_state(params) for sid, (_, params) in result.replies.items()}
        reference = None
        if self.recent:
            while len(self.recent) > 1 and self.recent[1][0] <= result.received_ns - SETTLE_S * 1e9:
                self.recent.popleft()
            if result.received_ns - self.recent[0][0] >= SETTLE_S * 1e9 and len(states) == len(self.ids):
                positions = [states[sid]['position'] for sid in self.ids]
                reference = dict(zip(self.ids, closest_recent_goal([g for _, g in self.recent], positions).tolist()))
        return abort_reason(states, self.limits, reference, self.consecutive_lost)

    def positions(self):
        result = self.bus.sync_read(self.ids, *STATE_BLOCK)
        if result.lost:
            raise SystemExit(f'No state from ids {result.lost}; not starting')
        return np.array([decode_state(result.replies[sid][1])['position'] for sid in self.ids])

    def save(self, stem):
        n, j = len(self.raw), len(self.ids)
        np.savez_compressed(
            stem.with_suffix('.npz'), ids=np.array(self.ids), sent_ns=np.array(self.sent, dtype=np.int64),
            received_ns=np.array(self.received, dtype=np.int64), lost=np.array(self.lost),
            state_raw=np.frombuffer(b''.join(self.raw), dtype=np.uint8).reshape(n, j, STATE_BLOCK[1]),
            valid=np.array(self.valid, dtype=bool).reshape(n, j),
            command_sent_ns=np.array([c[0] for c in self.commands], dtype=np.int64),
            command_flushed_ns=np.array([c[1] for c in self.commands], dtype=np.int64),
            command_goal_ticks=np.array([c[2] for c in self.commands], dtype=int).reshape(len(self.commands), j))
        return rate_summary(self.sent, self.received, self.lost) if self.sent else {'cycles': 0}


def stdin_has_line():
    return bool(select.select([sys.stdin], [], [], 0)[0])


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
    name = args.motion.stem if args.motion else 'dry-run'

    port = open_bus_port(args.port)
    bus = ServoBus(port, timeout_s=args.timeout_ms / 1000, allow_writes=bool(args.motion))
    recorder = Recorder(bus, args.ids, limits)
    servos_before = servos_after = None
    torque_on, finished, abort, interrupted = False, False, None, False
    try:
        servos_before = dump_servos(bus, args.ids)
        goals, rate_hz, duration = None, None, args.seconds
        if args.motion:
            offsets, rate_hz, name = load_motion(args.motion, args.ids)
            planned_present, lower, upper = motion_window(servos_before, args.ids, args.margin_ticks)
            goals = plan(planned_present, args.start_ticks, offsets, lower, upper, args.ramp_s, rate_hz)
            duration = len(goals) / rate_hz
            acceleration = {sid: args.acceleration for sid in args.ids
                            if 'Acceleration' in servos_before[str(sid)]['lerobot_configure_would_change']}
            print(f'motion {name}: {args.ramp_s:g} s ramp {planned_present.tolist()} -> {goals[len(goals) - len(offsets)].tolist()}, '
                  f'then {len(offsets)} goals at {rate_hz:g} Hz ({duration:.1f} s total) on ids {args.ids}; '
                  f'max |offset| per id {np.abs(offsets).max(axis=0).tolist()} ticks')
            print(f'writes: Goal_Position = present on {args.ids}; Acceleration = {args.acceleration} on '
                  f'{sorted(acceleration) or "none"}; Torque_Enable = 1 on {args.ids}')
            if input('Operator present, cutoff in reach, base clamped, arm suspended? Type MOVE to start: ') != 'MOVE':
                raise SystemExit('not confirmed')
            present = recorder.positions()
            drift = np.abs(present - planned_present)
            if np.any(drift > limits.max_start_drift_ticks):
                raise SystemExit(f'The arm moved since the plan ({drift.tolist()} ticks); not starting')
            goals = plan(present, args.start_ticks, offsets, lower, upper, args.ramp_s, rate_hz)
            recorder.write_checked('Goal_Position', dict(zip(args.ids, present.tolist())), GOAL_POSITION)  # reads 0 while torque is off
            if acceleration:
                recorder.write_checked('Acceleration', acceleration, ACCELERATION)
            torque_on = True
            recorder.write('Torque_Enable', {sid: 1 for sid in args.ids}, TORQUE_ENABLE)

        next_goal = 0
        gc.disable()  # cyclic GC pauses grow with the log lists (tens of ms on long runs); the loop makes no cycles
        t0 = time.monotonic_ns()
        while (elapsed := (time.monotonic_ns() - t0) / 1e9) < duration:
            if goals is not None and next_goal < len(goals) and elapsed >= next_goal / rate_hz:
                recorder.command(goals[next_goal])
                next_goal += 1
            if abort := recorder.cycle():
                break
            if args.read_rate_hz:
                wait = (t0 + len(recorder.sent) * 1e9 / args.read_rate_hz - time.monotonic_ns()) / 1e9
                if wait > 0:
                    time.sleep(wait)
        if torque_on and abort is None:
            print('Motion finished, holding the last goal. Support the arm, then press Enter to release torque.')
            while not stdin_has_line():
                if abort := recorder.cycle():
                    break
            if abort is None:
                sys.stdin.readline()
        finished = abort is None
    except KeyboardInterrupt:
        interrupted = True
    finally:
        gc.enable()
        if torque_on:
            try:
                recorder.write('Torque_Enable', {sid: 0 for sid in args.ids}, TORQUE_ENABLE)
            finally:
                print('torque released')
        try:
            servos_after = dump_servos(bus, args.ids)
        finally:
            port.close()

    utc = datetime.now(timezone.utc)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f'{name}-{utc.strftime("%Y%m%dT%H%M%SZ")}'
    summary = recorder.save(stem)
    meta = {'utc': utc.isoformat(), 'mode': 'motion' if args.motion else 'dry-run', 'motion': name,
            'motion_file': str(args.motion) if args.motion else None, 'port': args.port, 'ids': args.ids,
            'timeout_ms': args.timeout_ms, 'read_rate_hz': args.read_rate_hz, 'margin_ticks': args.margin_ticks,
            'start_ticks': args.start_ticks, 'ramp_s': args.ramp_s, 'state_block': list(STATE_BLOCK),
            'safety_limits': limits.__dict__, 'abort': abort, 'interrupted': interrupted, 'finished': finished,
            'summary': summary,
            'writes': [{'register': n, 'sent_ns': s, 'flushed_ns': f, 'values': v} for n, s, f, v in recorder.events],
            'register_dump_before': servos_before, 'register_dump_after': servos_after}
    stem.with_suffix('.json').write_text(json.dumps(meta, indent=1, default=list) + '\n')
    print(json.dumps({'abort': abort, 'interrupted': interrupted, **summary}, indent=1))
    print(f'wrote {stem}.npz and .json')


if __name__ == '__main__':
    main()
