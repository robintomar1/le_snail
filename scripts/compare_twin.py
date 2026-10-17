#!/usr/bin/env python3
"""Compare a Stage 1 servo log with its sim-twin replay (#12); no simulation.

Both are put on the twin's clock (seconds from the Torque_Enable 1 write) and in servo ticks through the twin's
joint map, so the real side is exactly what scripts/actuator_report.py reads. Reports, for the joints the twin
drives: the RMS real-minus-sim position difference over the torque-on window, the hold offset (final 1 s, position
minus goal) for each, per-step response (dead time, 10-90 % rise, time to 90 %, overshoot, final error) for step
runs, and the band-averaged command -> encoder gain/phase/delay for chirp runs. Real data: measured via the ESP32
pass-through on the current arm with the PROVISIONAL map; the arm is heavier than the URDF (760 g vs 632 g) and
carries the IMU board on chirp/step runs of ids 2-4, which the twin does not model.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from le_snail.actuator_id import decode_state, frequency_response, step_metrics

BANDS = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.2), (1.2, 1.5), (1.5, 1.8), (1.8, 2.0)]
DEG_PER_TICK = 360 / 4096

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('twin', type=Path, help='twin output directory (twin.npz, meta.json) made with --log')
p.add_argument('--output', type=Path, help='write the comparison as JSON here')
args = p.parse_args()
meta = json.loads((args.twin / 'meta.json').read_text())
stem = meta['log']
log = np.load(f'{stem}.npz')
log_meta = json.loads(Path(f'{stem}.json').read_text())
twin = np.load(args.twin / 'twin.npz')
t0, (on, off) = meta['t0_ns'], meta['torque_window_ns']
ids = log['ids'].tolist()
names = [str(n) for n in twin['joint_names']]
columns = [names.index(j) for j in meta['joints']]
sign, offset = np.array(meta['tick_sign']), np.array(meta['tick_offset'])

real_t = ((log['sent_ns'] + log['received_ns']) / 2 - t0) / 1e9
real = np.array([[decode_state(bytes(row))['position'] for row in cycle] for cycle in log['state_raw']], dtype=float)
cmd_t = (log['command_sent_ns'] - t0) / 1e9
cmd = log['command_goal_ticks'].astype(float)
sim_t = twin['t_s']
sim = twin['joint_pos_rad'][:, columns] / sign * 4096 / (2 * math.pi) + offset   # radians -> ticks
window = (real_t >= 0) & (real_t <= (off - t0) / 1e9)
sim_at_real = np.stack([np.interp(real_t[window], sim_t, sim[:, j]) for j in range(len(ids))], axis=1)
result = {'log': stem, 'twin': str(args.twin), 'wrist_offset': meta['wrist_offset'], 'motion': log_meta['motion'], 'joints': {}}
for j, sid in enumerate(ids):
    goal_end = cmd[cmd_t <= (off - t0) / 1e9][-1, j]
    hold_real = real_t[window][-1] - 1.
    last = real_t[window] >= hold_real
    result['joints'][meta['joints'][j]] = {
        'rms_real_minus_sim_deg': float(np.sqrt(np.mean((real[window, j] - sim_at_real[:, j]) ** 2)) * DEG_PER_TICK),
        'final_offset_from_goal_deg': {'real': float(np.median(real[window][last, j]) - goal_end) * DEG_PER_TICK,
                                       'sim': float(np.median(sim_at_real[last, j]) - goal_end) * DEG_PER_TICK}}

# Excited joint and its steps or chirp, as in scripts/actuator_report.py.
period = float(np.median(np.diff(cmd_t)))
first = round(log_meta['ramp_s'] / period) + 1
print(f"## {Path(stem).name} vs {args.twin} (wrist offset {meta['wrist_offset']})")
for name, r in result['joints'].items():
    o = r['final_offset_from_goal_deg']
    print(f"  {name:13s} RMS real-sim {r['rms_real_minus_sim_deg']:6.2f} deg   final offset from goal: real {o['real']:+6.2f}  sim {o['sim']:+6.2f} deg")
if len(cmd) > first + 1 and np.ptp(cmd[first:], axis=0).max() > 0:
    joint = int(np.argmax(np.ptp(cmd[first:], axis=0)))
    jname = meta['joints'][joint]
    result['excited'] = jname
    if log_meta['motion'].startswith('chirp'):
        moving = np.flatnonzero(np.diff(cmd[first:, joint]) != 0) + first
        lo, hi = cmd_t[moving[0]], cmd_t[moving[-1]]
        c = (cmd_t >= lo) & (cmd_t <= hi)
        rw, sw = (real_t >= lo) & (real_t <= hi), (sim_t >= lo) & (sim_t <= hi)
        fr = {'real': frequency_response(cmd_t[c] + period / 2, cmd[c, joint], real_t[rw], real[rw, joint], BANDS),
              'sim': frequency_response(cmd_t[c] + period / 2, cmd[c, joint], sim_t[sw], sim[sw, joint], BANDS)}
        result['chirp'] = fr
        print(f'  chirp on {jname}:  f(Hz)  gain real/sim   phase real/sim (deg)   delay real/sim (ms)')
        for a, b in zip(fr['real'], fr['sim']):
            print(f"   {a['f_hz']:5.2f}   {a['gain']:5.2f} / {b['gain']:5.2f}   {a['phase_deg']:7.1f} / {b['phase_deg']:7.1f}   {a['delay_s'] * 1e3:6.0f} / {b['delay_s'] * 1e3:6.0f}")
    else:
        changes = [i for i in range(first + 1, len(cmd)) if abs(cmd[i, joint] - cmd[i - 1, joint]) >= 5]
        result['steps'] = []
        print(f'  steps on {jname}: step(deg)  dead(ms) r/s   t90(ms) r/s   overshoot(%) r/s   final err(deg) r/s')
        for n, i in enumerate(changes):
            end = changes[n + 1] if n + 1 < len(changes) else len(cmd)
            t_cmd = cmd_t[i]
            t_end = min(cmd_t[end] if end < len(cmd) else t_cmd + 1., (off - t0) / 1e9)
            goal = cmd[i, joint]
            m = {'real': step_metrics(real_t, real[:, joint], t_cmd, goal, t_end), 'sim': step_metrics(sim_t, sim[:, joint], t_cmd, goal, t_end)}
            m['step_deg'] = (goal - cmd[i - 1, joint]) * DEG_PER_TICK
            result['steps'].append(m)
            r, s = m['real'], m['sim']
            print(f"   {m['step_deg']:+7.1f}  {r['dead_time_s'] * 1e3:5.0f} / {s['dead_time_s'] * 1e3:4.0f}   {r['time_to_90_s'] * 1e3:5.0f} / {s['time_to_90_s'] * 1e3:4.0f}"
                  f"   {r['overshoot'] * 100:5.1f} / {s['overshoot'] * 100:5.1f}   {r['final_error_ticks'] * DEG_PER_TICK:+5.2f} / {s['final_error_ticks'] * DEG_PER_TICK:+5.2f}")
if args.output:
    args.output.write_text(json.dumps(result, indent=1) + '\n')
