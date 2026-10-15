#!/usr/bin/env python3
"""Offline check of a two-portion reward on recorded trajectories; no simulation or training (#6).

Proposed reward per 20 Hz policy step, identical in every episode and after every fall:
  standing = w_stand * dt * u(cos)^2,  u = clamp((cos + 0.1) / 0.9, 0, 1)   (convex: half-up earns little)
  forward  = w_fwd * g(cos) * max(0, x - x_record),  g = clamp((cos - 0.5) / 0.3, 0, 1);
             x_record = running maximum of world x in the episode, updated every step (new ground only)
  effort   = existing command-rate cost * g(cos)   (smoothness while upright; moving while fallen is free)
Terminations unchanged and absorbing. Measured parts use recorded traces; 120 s totals extrapolate
measured per-second rates and say so.
"""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, default=Path('runs/two-portion-reward-analysis-001'))
p.add_argument('--w-stand', type=float, nargs='+', default=[.25, .5, 1.])
p.add_argument('--w-fwd', type=float, default=30.)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
DT, GAMMA = .05, .995


def u2(up):
    return np.clip((up + .1) / .9, 0., 1.) ** 2


def gate(up):
    return np.clip((up - .5) / .3, 0., 1.)


def new_ground(x):
    record = np.maximum.accumulate(np.concatenate(([0.], x)))
    return np.maximum(x - record[:-1], 0.)


def crawl_mode(up):
    """On after a stable upright (cos >= 0.8 for 0.5 s), off at the next fall below cos 0.5 (as in the env)."""
    need, streak, mode, out = round(.5 / DT) + 1, 0, False, np.zeros(len(up), dtype=bool)
    for t, u in enumerate(up):
        streak = streak + 1 if u >= .8 else 0
        mode = True if streak >= need else (False if u < .5 else mode)
        out[t] = mode
    return out


def parts(x, up, effort, w_stand):
    return {'standing': w_stand * DT * u2(up), 'forward': args.w_fwd * gate(up) * crawl_mode(up) * new_ground(x),
            'effort': effort * gate(up)}


def horizons(per_step, live):
    per_step = per_step * live
    out = {f'{h:g}s': float(per_step[:round(h / DT)].sum()) for h in (2, 4, 6, 10, 30)}
    weights = GAMMA ** np.arange(round(30 / DT))
    out['disc30'] = float((per_step[:round(30 / DT)] * weights[:len(per_step[:round(30 / DT)])]).sum())
    return out


audit = Path('runs/reward-study-gpu-returns-001')
report = json.loads((audit / 'report.json').read_text())
traces = {}
for side in ('left', 'right'):
    for mode in ('hold', 'script', 'deterministic', 'stochastic'):
        group = next(g for g in report['groups'] if g['side'] == side and g['mode'] == mode)
        data = np.load(audit / f'{side}-{mode}.npz')
        traces[side, mode] = (data['state'], data['components'][:, :, group['term_names'].index('command_rate')],
                              data['active'].astype(float), group)
crawls = []
for run in sorted(Path('runs').glob('evaluation-handover-*-050-development-5100')):
    for trace in sorted(run.glob('crawl-*.jsonl')):
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        crawls.append((f"{run.name.split('-')[2]}/{trace.stem}", np.array([r['forward_m'] for r in rows]),
                       np.array([r['upright_cos'] for r in rows]), bool(rows[-1].get('terminated'))))
good = [c for c in crawls if (c[2] >= .5).all() and not c[3]]
crawl_speed = float(np.mean([c[1][-1] / (len(c[1]) * DT) for c in good]))
crawl_stand = float(np.mean([u2(c[2]).mean() for c in good]))
crawl_effort_per_s = .0055 / DT   # training-log average of the command-rate term while crawling (Reward/command_rate)

result = {'reward': {'standing': 'w_stand * dt * clamp((cos+0.1)/0.9,0,1)^2', 'forward': 'w_fwd * clamp((cos-0.5)/0.3,0,1) * new ground (running max x)',
                     'effort': 'existing command-rate cost * the same uprightness gate', 'w_fwd_per_m': args.w_fwd},
          'measured_crawl': {'successful_crawls': len(good), 'speed_m_s': crawl_speed, 'mean_standing_u2': crawl_stand},
          'by_w_stand': {}}
for w in args.w_stand:
    entry = result['by_w_stand'][f'{w:g}'] = {'side_starts': {}, 'upright_crawls': {}, 'episode_120s': {}, 'checks': {}}
    for side in ('left', 'right'):
        table = {}
        for mode in ('hold', 'script', 'deterministic', 'stochastic'):
            state, effort, live, group = traces[side, mode]
            rows = [parts(state[:, i, 0], state[:, i, 3], effort[:, i], w) for i in range(state.shape[1])]
            table[mode] = {h: float(np.mean([horizons(sum(r.values()), live[:, i])[h] for i, r in enumerate(rows)]))
                           for h in ('2s', '4s', '6s', '10s', '30s', 'disc30')}
            # Pre-recovery phase of the scripted get-ups versus lying still for the same time.
            if mode == 'script':
                gaps = []
                for i, trial in enumerate(group['trials']):
                    if trial['recovery_time_s'] is not None:
                        k = round(trial['recovery_time_s'] / DT)
                        gaps.append(float(sum(v[:k].sum() for v in rows[i].values())))
                table['script_before_stable_upright'] = float(np.mean(gaps))
        entry['side_starts'][side] = table
    for name, x, up, terminated in crawls:
        values = parts(x, up, np.full(len(x), -crawl_effort_per_s * DT), w)
        entry['upright_crawls'][name] = {'seconds': len(x) * DT, 'forward_m': float(x[-1]), 'fell': bool((up < .5).any()),
                                         'ended': 'fault/lane' if terminated else 'success',
                                         'total': float(sum(v.sum() for v in values.values()))}
    # 120 s episodes from measured rates (extrapolated): crawling speed/uprightness from successful crawls,
    # the measured scripted get-up time (~3.3 s) and standing at the settled upright pose (u^2 = 1).
    stand, fwd, eff = w, args.w_fwd * crawl_speed, crawl_effort_per_s
    crawl_rate = stand * crawl_stand + fwd - eff
    t50 = .5 / crawl_speed
    entry['episode_120s'] = {
        'crawl continuously': 120 * crawl_rate,
        'crawl, fall once, get up in 3.3 s, keep crawling': (120 - 3.3) * crawl_rate,
        'crawl 50 cm, then stand still': t50 * crawl_rate + (120 - t50) * stand,
        'stand still upright': 120 * stand,
        'crawl, fall once, lie still (30 s limit ends it)': 30 * crawl_rate,
        'half-up resting pose (cos 0.35)': 120 * stand * .25,
        'lie still': 0.,
    }
    side = entry['side_starts']
    e120 = entry['episode_120s']
    entry['checks'] = {
        'get_up_beats_lying_at_4s_both_sides': all(side[s]['script']['4s'] > side[s]['hold']['4s'] for s in ('left', 'right')),
        'get_up_beats_lying_at_30s_both_sides': all(side[s]['script']['30s'] > side[s]['hold']['30s'] for s in ('left', 'right')),
        'trying_not_worse_than_lying_both_sides': all(min(side[s]['deterministic']['30s'], side[s]['stochastic']['30s']) >= side[s]['hold']['30s'] - 1e-9
                                                      for s in ('left', 'right')),
        'crawl_50cm_beats_stand_still_by_ratio': e120['crawl 50 cm, then stand still'] / e120['stand still upright'],
        'ordering_120s_holds': (e120['crawl continuously'] > e120['crawl, fall once, get up in 3.3 s, keep crawling'] > e120['stand still upright']
                                > e120['half-up resting pose (cos 0.35)'] >= e120['lie still']),
        'fall_then_recover_minus_fall_and_lie_within_10s': 6.7 * crawl_rate,
    }
(args.output / 'analysis.json').write_text(json.dumps(result, indent=1) + '\n')

print(f"Measured successful crawls: {len(good)}, speed {crawl_speed * 100:.1f} cm/s, mean standing u^2 {crawl_stand:.2f}")
labels = {'hold': 'lie still', 'script': 'scripted get-up + crawl', 'deterministic': 'learned, deterministic', 'stochastic': 'learned, exploring'}
for w, entry in result['by_w_stand'].items():
    print(f"\n######## w_stand = {w}/s, w_fwd = {args.w_fwd:g}/m")
    for side, table in entry['side_starts'].items():
        print(f"  {side.upper()} side start: {'2 s':>7s} {'4 s':>7s} {'6 s':>7s} {'10 s':>7s} {'30 s':>7s} {'disc30':>7s}")
        for mode in ('hold', 'script', 'deterministic', 'stochastic'):
            print(f"    {labels[mode]:24s}" + ' '.join(f"{table[mode][h]:7.2f}" for h in ('2s', '4s', '6s', '10s', '30s', 'disc30')))
        print(f"    scripted get-up before stable upright: {table['script_before_stable_upright']:+.3f}  (lying still: 0)")
    print('  120 s episodes (measured rates, extrapolated):')
    for name, value in entry['episode_120s'].items():
        print(f"    {name:52s} {value:8.1f}")
    print('  checks:', json.dumps(entry['checks']))
