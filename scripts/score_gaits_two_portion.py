#!/usr/bin/env python3
"""Score the laptop's recorded scripted gaits (#13) under the two-portion reward; offline, no simulation (#6).

Uses the reward block of --config on 20 Hz traces of world x, upright cosine and joint targets. The standing
and forward portions and the effort cost are reproduced exactly as in src/le_snail/rewards.py (crawl mode on
after cos >= 0.8 for 0.5 s, off below 0.5; new ground against the running max of x). The get-up potential
shaping is approximated with Phi = u^2 (a lower bound of max(progress, u^2)); for upright gaits it mostly
telescopes. Reports per-second totals, the share of net forward distance that earns credit, crawl-mode time
and the time below each uprightness threshold, against standing still at the settled upright pose.
"""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, default=Path('configs/experiments/two-portion-002.json'))
p.add_argument('--output', type=Path, default=Path('runs/gait-scores-two-portion-001'))
args = p.parse_args()
cfg = json.loads(args.config.read_text())
reward = cfg['reward']
DT = .05
GAMMA = .995 ** (DT / .05)
settled = cfg['settled_starts']['upright']['root_pose']
w, x_, y_, z_ = settled[3:7]
REST_COS = 1 - 2 * (x_ * x_ + y_ * y_)


def ramp(up, bounds):
    low, high = bounds
    return np.clip((up - low) / (high - low), 0., 1.)


def crawl_mode(up):
    need, streak, mode, out = round(.5 / DT) + 1, 0, False, np.zeros(len(up), dtype=bool)
    for t, u in enumerate(up):
        streak = streak + 1 if u >= .8 else 0
        mode = True if streak >= need else (False if u < .5 else mode)
        out[t] = mode
    return out


def score(x, up, target, alive=None):
    n = len(x) if alive is None else int(alive.sum())
    x, up, target = x[:n] - x[0], up[:n], target[:n]
    u2 = ramp(up, reward['standing_ramp_cos']) ** 2
    gate = ramp(up, reward['forward_gate_cos'])
    mode = crawl_mode(up)
    record = np.maximum.accumulate(np.concatenate(([0.], x)))[:-1]
    ground = np.maximum(x - record, 0.)
    forward = reward['forward_per_m'] * gate * mode * ground
    rate = np.diff(np.vstack((target[:1], target)), axis=0) / DT
    effort = -reward['effort_weight'] * DT * (rate ** 2).sum(1) * gate
    standing = reward['standing_per_s'] * DT * u2
    shaping = reward['getup_progress_weight'] * (GAMMA * u2[1:] - u2[:-1])
    seconds = n * DT
    return {'seconds': seconds, 'fell': bool((up < .5).any()), 'net_forward_m': float(x[-1]),
            'per_s': {'standing': standing.sum() / seconds, 'forward': forward.sum() / seconds, 'effort': effort.sum() / seconds,
                      'shaping_u2_approx': shaping.sum() / seconds},
            'total_per_s': (standing.sum() + forward.sum() + effort.sum() + shaping.sum()) / seconds,
            'forward_credit_share': float((gate * mode * ground).sum() / max(ground.sum(), 1e-9)),
            'new_ground_m': float(ground.sum()),
            'crawl_mode_time_share': float(mode.mean()),
            'time_share_cos_below': {f'{c:g}': float((up < c).mean()) for c in (.8, .7, .5)},
            'up_p5': float(np.percentile(up, 5))}


groups = {}
d = np.load('runs/gait-study-rowing/eval/nominal_traces.npz')
for i, name in enumerate(d['names']):
    groups.setdefault(f'rowing/{name}', []).append(score(d['x'][i], d['up'][i], d['tgt'][i]))
d = np.load('runs/gait-study-inchworm/eval-nominal/traces_round00.npz')
sets = json.loads(Path('runs/gait-study-inchworm/eval_sets.json').read_text()) if Path('runs/gait-study-inchworm/eval_sets.json').exists() else None
for j in range(d['x'].shape[1]):
    groups.setdefault(f'inchworm/env{int(d["envs"][j])}', []).append(score(d['x'][:, j], d['up'][:, j], d['target'][:, j], d['alive'][:, j]))
cands = json.loads(Path('runs/gait-study-rock/final_cands.json').read_text())
z = np.load('runs/gait-study-rock/final1/trace-batch0.npz')['steps']
per = z.shape[0] // len(cands)
for ci, c in enumerate(cands):
    for e in range(ci * per, (ci + 1) * per):
        tr = z[e]
        groups.setdefault(f"rock/{c.get('name', ci)}", []).append(score(tr[:, 0], tr[:, 3], tr[:, 13:19], tr[:, 19] > 0))

still = {'seconds': 120., 'per_s': {'standing': reward['standing_per_s'] * ramp(np.array([REST_COS]), reward['standing_ramp_cos'])[0] ** 2,
         'forward': 0., 'effort': 0., 'shaping_u2_approx': -(1 - GAMMA) * reward['getup_progress_weight'] * ramp(np.array([REST_COS]), reward['standing_ramp_cos'])[0] ** 2 / DT}}
still['total_per_s'] = sum(still['per_s'].values())
summary = {}
for name, rows in groups.items():
    keys = ('total_per_s', 'net_forward_m', 'forward_credit_share', 'crawl_mode_time_share', 'up_p5', 'seconds')
    summary[name] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary[name].update(trials=len(rows), fell=sum(r['fell'] for r in rows),
                         per_s={k: float(np.mean([r['per_s'][k] for r in rows])) for k in rows[0]['per_s']},
                         cos_below={k: float(np.mean([r['time_share_cos_below'][k] for r in rows])) for k in rows[0]['time_share_cos_below']})
args.output.mkdir(parents=True, exist_ok=True)
(args.output / 'scores.json').write_text(json.dumps({'config': str(args.config), 'rest_upright_cos': REST_COS, 'stand_still': still,
                                                    'gaits': summary}, indent=1) + '\n')
print(f"settled upright rest cos {REST_COS:.3f}; stand still: total {still['total_per_s']:+.3f}/s")
print(f"{'gait':34s} {'n':>3s} {'fell':>4s} {'cm/s':>6s} {'tot/s':>7s} {'stand':>6s} {'fwd':>6s} {'eff':>7s} {'credit':>6s} {'crawl':>6s} {'<0.8':>5s} {'<0.5':>5s} {'up_p5':>6s}")
for name, s in sorted(summary.items(), key=lambda kv: -kv[1]['total_per_s']):
    print(f"{name[:34]:34s} {s['trials']:3d} {s['fell']:4d} {100 * s['net_forward_m'] / s['seconds']:6.2f} {s['total_per_s']:+7.3f} {s['per_s']['standing']:6.3f} "
          f"{s['per_s']['forward']:6.3f} {s['per_s']['effort']:+7.4f} {s['forward_credit_share']:6.2f} {s['crawl_mode_time_share']:6.2f} "
          f"{s['cos_below']['0.8']:5.2f} {s['cos_below']['0.5']:5.2f} {s['up_p5']:6.2f}")
