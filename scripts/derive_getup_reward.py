#!/usr/bin/env python3
"""Derive a get-up progress reward from recorded scripted get-ups and test it offline; no simulation (#6).

Progress potential: Phi(s) = tau_nn * exp(-d_nn^2 / (2 sigma^2)), where tau_nn is the phase (time / time to
stable upright) of the nearest state in a bank of successful scripted get-ups and d_nn its distance in
standardized [base-frame gravity, joint angles]. The shaping reward is w_phase * (gamma Phi(s') - Phi(s)),
so a closed loop earns nothing. Generalization is tested leave-one-variant-out: every successful
perturbation is scored against a bank that excludes it (and replicas of the same keyframes).
Full reward checked: shaping + standing (w_stand dt u^2) + forward (new ground while upright) - effort
(only while upright; omitted here because these trajectories have no crawl phase).
"""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data', type=Path, default=Path('runs/getup-trajectories-50hz-001'))
p.add_argument('--output', type=Path, default=Path('runs/getup-reward-derivation-001'))
p.add_argument('--w-phase', type=float, default=2.)
p.add_argument('--w-stand', type=float, default=.25)
p.add_argument('--bank-stride', type=int, default=4)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
report = json.loads((args.data / 'report.json').read_text())
dt = report['policy_dt']
gamma = .995 ** (dt / .05)


def load(side, kind):
    data = np.load(args.data / f'{side}-{kind}.npz')
    batch = next(b for b in report['batches'] if b['side'] == side and b['kind'] == kind)
    return data, batch['labels']


trajectories = []   # (side, kind, variant, success, stable_step, features[T, 9], upright[T])
for side in ('left', 'right'):
    for kind in ('perturbed', 'confirmed', 'hold', 'random'):
        data, labels = load(side, kind)
        features = np.concatenate((data['gravity_b'], data['joint_q']), axis=2)   # (T, N, 9)
        for i, label in enumerate(labels):
            t_up = label['stable_upright_time_s']
            trajectories.append({'side': side, 'kind': kind, 'variant': label['variant'], 'success': t_up is not None,
                                 'stable_step': None if t_up is None else round(t_up / dt) - 1,
                                 'features': features[:, i], 'upright': data['upright'][:, i]})
success = [t for t in trajectories if t['success'] and t['kind'] in ('perturbed', 'confirmed')]
stack = np.concatenate([t['features'][:t['stable_step'] + 1] for t in success])
mean, std = stack.mean(0), stack.std(0) + 1e-6


def bank_from(selected):
    states, phases = [], []
    for t in selected:
        k = t['stable_step']
        idx = np.arange(0, k + 1, args.bank_stride)
        states.append((t['features'][idx] - mean) / std)
        phases.append(idx / k)
    return np.concatenate(states), np.concatenate(phases)


K_NEIGHBOURS, SHRINK = 16, 1.


def nearest(bank, phases, queries, chunk=4096, k=1):
    """Distances and phases of the k nearest bank states for each query (sorted by distance)."""
    best_d, best_tau = np.empty((len(queries), k)), np.empty((len(queries), k))
    norms = (bank ** 2).sum(1)
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk]
        d2 = np.maximum((q ** 2).sum(1)[:, None] + norms[None, :] - 2 * q @ bank.T, 0.)
        j = np.argpartition(d2, k - 1, axis=1)[:, :k] if k < len(bank) else np.argsort(d2, axis=1)
        rows = np.arange(len(q))[:, None]
        order = np.argsort(d2[rows, j], axis=1)
        j = j[rows, order]
        best_d[start:start + chunk] = np.sqrt(d2[rows, j])
        best_tau[start:start + chunk] = phases[j]
    return (best_d[:, 0], best_tau[:, 0]) if k == 1 else (best_d, best_tau)


# Kernel width: typical distance from a successful state to the nearest state of a DIFFERENT successful trajectory.
full_bank, full_phase = bank_from(success)
probe = []
for t in success[::7]:
    others = [o for o in success if o is not t]
    b, ph = bank_from(others)
    d, _ = nearest(b, ph, (t['features'][:t['stable_step'] + 1:2] - mean) / std)
    probe.append(d)
sigma = float(np.percentile(np.concatenate(probe), 90))


def potential(t, bank, phases):
    """Kernel regression of phase over the K nearest successful states, shrunk to 0 far from the bank,
    and never below the convex uprightness term, so any upright state scores ~1 whatever the arm does."""
    d, tau = nearest(bank, phases, (t['features'] - mean) / std, k=K_NEIGHBOURS)
    w = np.exp(-d ** 2 / (2 * sigma ** 2))
    progress = (w * tau).sum(1) / (w.sum(1) + SHRINK)
    upright = np.clip((t['upright'] + .1) / .9, 0., 1.) ** 2
    return np.maximum(progress, upright)


def rewards(t, phi):
    shaping = args.w_phase * (gamma * phi[1:] - phi[:-1])
    shaping = np.concatenate(([0.], shaping))
    standing = args.w_stand * dt * np.clip((t['upright'] + .1) / .9, 0., 1.) ** 2
    return shaping, standing


def same_keys(a, b):
    return a['side'] == b['side'] and a['variant'] is not None and a['variant'] == b['variant']


checkpoints = (1., 2., 3., 4., 8.)
rows = []
for t in trajectories:
    if t['success'] and t['kind'] == 'perturbed':
        bank, phases = bank_from([o for o in success if not same_keys(o, t)])   # leave this variant out
        held_out = True
    else:
        bank, phases, held_out = full_bank, full_phase, t['kind'] not in ('perturbed', 'confirmed') or not t['success']
    phi = potential(t, bank, phases)
    shaping, standing = rewards(t, phi)
    cumulative = np.cumsum(shaping + standing)
    entry = {'side': t['side'], 'kind': t['kind'], 'variant': t['variant'], 'success': t['success'], 'held_out': held_out,
             'phi_max': float(phi.max()), 'phi_end': float(phi[-1]),
             'reward_at': {f'{c:g}s': float(cumulative[min(round(c / dt), len(cumulative)) - 1]) for c in checkpoints}}
    if t['success']:
        k = t['stable_step']
        quarter = [float(phi[round(f * k)]) for f in (.25, .5, .75, 1.)]
        drops = np.diff(phi[:k + 1])
        entry.update(phi_at_quarters=quarter, phi_drop_fraction=float((drops < -.02).mean()),
                     reward_before_stable=float(cumulative[k]))
    rows.append(entry)


def group(side, predicate):
    return [r for r in rows if r['side'] == side and predicate(r)]


summary = {'sigma': sigma, 'gamma_per_step': gamma, 'w_phase': args.w_phase, 'w_stand_per_s': args.w_stand, 'bank_states': len(full_bank), 'sides': {}}
for side in ('left', 'right'):
    classes = {
        'held-out successful perturbations': group(side, lambda r: r['kind'] == 'perturbed' and r['success']),
        'failed perturbations': group(side, lambda r: r['kind'] == 'perturbed' and not r['success']),
        'lying still': group(side, lambda r: r['kind'] == 'hold'),
        'random flailing': group(side, lambda r: r['kind'] == 'random'),
    }
    out = summary['sides'][side] = {}
    for name, members in classes.items():
        if not members:
            out[name] = {'n': 0}
            continue
        out[name] = {'n': len(members), 'phi_max_median': float(np.median([m['phi_max'] for m in members])),
                     'reward_median': {c: float(np.median([m['reward_at'][c] for m in members])) for c in members[0]['reward_at']},
                     'reward_min': {c: float(np.min([m['reward_at'][c] for m in members])) for c in members[0]['reward_at']}}
        if name.startswith('held-out'):
            out[name]['phi_at_quarters_median'] = np.median([m['phi_at_quarters'] for m in members], axis=0).tolist()
            out[name]['phi_drop_fraction_median'] = float(np.median([m['phi_drop_fraction'] for m in members]))
            out[name]['reward_before_stable_min'] = float(np.min([m['reward_before_stable'] for m in members]))
    # Early discrimination: does reward at 1 s / 2 s already rank eventual successes above failures (ROC AUC)?
    succ, fail = classes['held-out successful perturbations'], classes['failed perturbations']
    for c in ('1s', '2s'):
        if succ and fail:
            pairs = [(s['reward_at'][c] > f['reward_at'][c]) + .5 * (s['reward_at'][c] == f['reward_at'][c]) for s in succ for f in fail]
            out[f'auc_success_vs_failure_at_{c}'] = float(np.mean(pairs))
(args.output / 'derivation.json').write_text(json.dumps({'summary': summary, 'trajectories': rows}, indent=1) + '\n')
print(json.dumps(summary, indent=1))
