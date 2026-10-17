#!/usr/bin/env python3
"""Summarize Stage 1 step runs of #12: per-step servo response and, with the IMU on the forearm, link play.

For each log (stem without extension) it finds the excited joint and every goal step after the start ramp,
then reports dead time, 10-90 % rise, overshoot, final error, peak speed and peak load. When the log has
IMU data, it also reports the forearm pitch residual: pitch + sign * (shoulder_lift + elbow_flex), which
is constant for a rigid arm. Its change between holds at the same goal, approached from opposite sides,
is play or flex that the servo encoders cannot see. The residual sign is fitted per run.
"""
import argparse
import json
import math

import numpy as np

from le_snail.actuator_id import TICKS_PER_RAD, decode_state, frequency_response, step_metrics

DEG_PER_TICK = 360 / 4096
BANDS = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.2), (1.2, 1.5), (1.5, 1.8), (1.8, 2.0)]
NAMES = {1: 'shoulder_pan', 2: 'shoulder_lift', 3: 'elbow_flex', 4: 'wrist_flex', 5: 'wrist_roll'}


def load(stem):
    log = dict(np.load(f'{stem}.npz'))
    meta = json.load(open(f'{stem}.json'))
    states = [[decode_state(bytes(row)) for row in cycle] for cycle in log['state_raw']]
    field = lambda key: np.array([[s[key] for s in cycle] for cycle in states], dtype=float)
    t0 = log['sent_ns'][0]
    return {
        'meta': meta, 'ids': log['ids'].tolist(),
        't': (log['sent_ns'] + log['received_ns']) / 2e9 - t0 / 1e9,  # sample ~ mid-transaction
        'pos': field('position'), 'vel': field('velocity'), 'load': field('load'), 'temp': field('temperature_c'),
        'cmd_t': (log['command_sent_ns'] - t0) / 1e9, 'cmd': log['command_goal_ticks'].astype(float),
        'imu': imu_pitch(log, t0) if 'imu_fields' in log and len(log['imu_fields']) else None,
    }


def imu_pitch(log, t0):
    """(time, pitch of the sensor x axis in degrees) on the servo clock; start-up packets dropped."""
    fields, received = log['imu_fields'], log['imu_received_ns'] / 1e9
    ok = ~np.isnan(fields[:, 2])
    ok &= np.r_[False, np.diff(fields[:, 0]) % 65536 == 1] & np.r_[False, np.diff(fields[:, 1]) > 0]
    fields, received = fields[ok], received[ok]
    sensor = fields[:, 1] / 1e4  # SampleTimeFine, 10 kHz
    offset = received - sensor
    keep = np.abs(offset - np.median(offset)) < 0.1
    fields, sensor, offset = fields[keep], sensor[keep], offset[keep]
    t = sensor + np.percentile(offset, 1) - t0 / 1e9  # minimum transport delay as the clock offset
    w, x, y, z = fields[:, 2:6].T
    return t, np.degrees(np.arcsin(np.clip(2 * (x * z - w * y), -1, 1)))


def steps(run):
    meta, cmd, cmd_t = run['meta'], run['cmd'], run['cmd_t']
    period = float(np.median(np.diff(cmd_t)))
    first = round(meta['ramp_s'] / period) + 1  # commands after the start ramp
    joint = int(np.argmax(np.ptp(cmd[first:], axis=0)))
    changes = [i for i in range(first + 1, len(cmd)) if abs(cmd[i, joint] - cmd[i - 1, joint]) >= 5]
    return joint, [(i, changes[n + 1] if n + 1 < len(changes) else len(cmd)) for n, i in enumerate(changes)], first


def chirp_report(run, stem):
    """Command -> encoder and (for joints the forearm IMU sees) encoder -> IMU, per frequency band."""
    meta, cmd, cmd_t, t, pos = run['meta'], run['cmd'], run['cmd_t'], run['t'], run['pos']
    period = float(np.median(np.diff(cmd_t)))
    first = round(meta['ramp_s'] / period) + 1
    joint = int(np.argmax(np.ptp(cmd[first:], axis=0)))
    moving = np.flatnonzero(np.diff(cmd[first:, joint]) != 0) + first
    t_lo, t_hi = cmd_t[moving[0]], cmd_t[moving[-1]]
    c = (cmd_t >= t_lo) & (cmd_t <= t_hi)
    w = (t >= t_lo) & (t <= t_hi)
    sid = run['ids'][joint]
    print(f'\n## {stem.split("/")[-1]}: {NAMES.get(sid, sid)} (id {sid}) chirp; abort {meta["abort"]}, '
          f'{meta["summary"]["rate_hz"]:.0f} Hz, lost {meta["summary"]["lost_replies"]}, command +/-{np.ptp(cmd[c, joint]) / 2 * DEG_PER_TICK:.1f} deg, '
          f'peak |load| {np.abs(run["load"][w, joint]).max():.0f}, temperature {np.median(run["temp"][-50:], 0).astype(int).tolist()} C')
    # Goals are held between writes; shifting the samples by half a period makes linear interpolation match that hold.
    servo = frequency_response(cmd_t[c] + period / 2, cmd[c, joint], t[w], pos[w, joint], BANDS)
    link = None
    if run['imu'] is not None and sid in (2, 3) and {2, 3} <= set(run['ids']):
        ti, pitch = run['imu']
        chain = (pos[:, run['ids'].index(2)] + pos[:, run['ids'].index(3)]) * DEG_PER_TICK
        wi = (ti >= t_lo) & (ti <= t_hi)
        link = frequency_response(t[w], -chain[w], ti[wi], pitch[wi], BANDS)
    print(' f(Hz)  enc/cmd gain  phase(deg)  delay(ms)' + ('  |  imu/enc gain  phase(deg)' if link else ''))
    for n, b in enumerate(servo):
        extra = f'  |  {link[n]["gain"]:12.2f}  {link[n]["phase_deg"]:10.1f}' if link else ''
        print(f' {b["f_hz"]:5.2f}  {b["gain"]:12.2f}  {b["phase_deg"]:10.1f}  {b["delay_s"] * 1e3:9.0f}{extra}')


def report(stem):
    run = load(stem)
    if run['meta']['motion'].startswith('chirp'):
        return chirp_report(run, stem)
    joint, segments, first = steps(run)
    sid = run['ids'][joint]
    t, pos = run['t'], run['pos']
    meta = run['meta']
    print(f'\n## {stem.split("/")[-1]}: {NAMES.get(sid, sid)} (id {sid}); abort {meta["abort"]}, '
          f'{meta["summary"]["rate_hz"]:.0f} Hz, lost {meta["summary"]["lost_replies"]}, '
          f'temperature {np.median(run["temp"][:50], 0).astype(int).tolist()} -> {np.median(run["temp"][-50:], 0).astype(int).tolist()} C')
    residual = None
    if run['imu'] is not None and {2, 3} <= set(run['ids']):
        ti, pitch = run['imu']
        chain = (np.interp(ti, t, pos[:, run['ids'].index(2)]) + np.interp(ti, t, pos[:, run['ids'].index(3)])) * DEG_PER_TICK
        sign = -1.0 if np.corrcoef(pitch, chain)[0, 1] < 0 else 1.0
        residual = (ti, pitch - sign * chain)
        print(f'IMU forearm pitch = {sign:+.0f} x (lift + elbow) + c; residual spread over the run {np.ptp(residual[1]):.2f} deg')
    print('step(deg)  from  dead(ms)  rise(ms)  t90(ms)  overshoot  final_err(deg)  peak_vel(rad/s)  peak|load|  imu_resid(deg)')
    rows = []
    for i, end in segments:
        t_cmd = run['cmd_t'][i]
        t_end = run['cmd_t'][end] if end < len(run['cmd_t']) else t_cmd + 1.0
        goal, previous = run['cmd'][i, joint], run['cmd'][i - 1, joint]
        m = step_metrics(t, pos[:, joint], t_cmd, goal, t_end)
        w = (t >= t_cmd) & (t < t_end)
        hold = None
        if residual is not None:
            r = (residual[0] >= t_end - 0.3) & (residual[0] < t_end)
            hold = float(np.mean(residual[1][r])) if r.any() else None
        rows.append((goal, previous, hold))
        print(f'{(goal - previous) * DEG_PER_TICK:+8.1f} {"above" if previous > goal else "below":>5} {m["dead_time_s"] * 1e3:8.1f} '
              f'{m["rise_10_90_s"] * 1e3:9.1f} {m["time_to_90_s"] * 1e3:8.1f} {m["overshoot"] * 100:8.1f}% '
              f'{m["final_error_ticks"] * DEG_PER_TICK:+14.2f} {np.abs(run["vel"][w, joint]).max() / TICKS_PER_RAD:15.2f} '
              f'{np.abs(run["load"][w, joint]).max():10.0f}  {"" if hold is None else f"{hold:+.2f}"}')
    if residual is not None:
        centre = run['cmd'][first, joint]
        above = [h for g, p, h in rows if g == centre and p > g and h is not None]
        below = [h for g, p, h in rows if g == centre and p < g and h is not None]
        if above and below:
            print(f'hysteresis at centre (IMU, encoder-invisible play): returning from above {np.mean(above):+.2f}, '
                  f'from below {np.mean(below):+.2f} -> {np.mean(above) - np.mean(below):+.2f} deg')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stems', nargs='+', help='log paths without .npz/.json')
    for stem in parser.parse_args().stems:
        report(stem)


if __name__ == '__main__':
    main()
