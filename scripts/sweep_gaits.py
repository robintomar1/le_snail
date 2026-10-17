#!/usr/bin/env python3
"""Compare phase-locked crawl gaits side by side in sim (#15); no training.

Each candidate (a gait block like configs/controllers/hybrid-001.json's, optionally with a period scale) drives an
equal share of the environments through PhaseGait from upright starts for --seconds. Per candidate: steady forward
speed, time to 50 cm, terminations (lane exits or falls), max |lateral|, final |yaw|, fraction of time with upright
cosine >= 0.8 and the minimum cosine. Writes only --output (JSON).
"""
import argparse
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('candidates', type=Path, help='JSON list of {"name", "gait": {...}, "period_scale"?}')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--contract', type=Path, default=Path('configs/policy-50hz-001.json'))
p.add_argument('--physics', type=Path, default=Path('configs/experiments/physics-v2-50hz-001.json'))
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--stage', choices=('robust', 'crawl'), default='robust')
p.add_argument('--envs-per-candidate', type=int, default=128)
p.add_argument('--seconds', type=float, default=40.)
p.add_argument('--seed', type=int, default=11)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new')
app = AppLauncher(args).app
env = None


def main():
    global env
    import math
    import torch
    from le_snail.gait_teacher import PhaseGait
    from le_snail.sim.env import SnailEnv, make_config

    candidates = json.loads(args.candidates.read_text())
    contract_raw = json.loads(args.contract.read_text())
    k, device = args.envs_per_candidate, args.device or 'cuda:0'
    n = k * len(candidates)
    env = SnailEnv(make_config(args.asset, args.contract, args.physics, num_envs=n, stage=args.stage, device=device,
                               evaluation=True, initial_side='upright', seed=args.seed))
    teachers = []
    for c in candidates:
        gait = dict(c['gait'])
        gait['T'] = gait['T'] * c.get('period_scale', 1.)
        teachers.append(PhaseGait({'gait': gait}, contract_raw).to(device))
    obs, _ = env.reset(seed=args.seed)
    start = env.robot.data.root_pos_w.clone()
    alive = torch.ones(n, dtype=torch.bool, device=device)
    t50 = torch.full((n,), float('nan'), device=device)
    max_lateral = torch.zeros(n, device=device)
    upright_steps = torch.zeros(n, device=device)
    min_cos = torch.ones(n, device=device)
    final_x, final_y, final_yaw = (torch.zeros(n, device=device) for _ in range(3))
    x_at_4s = torch.zeros(n, device=device)
    steps = round(args.seconds / env.step_dt)
    yaw0 = None
    for s in range(steps):
        o = obs['policy']
        action = torch.cat([t(o[i * k:(i + 1) * k]) for i, t in enumerate(teachers)])
        obs, _, term, trunc, _ = env.step(action)
        pos, gravity, _ = env.last_transition
        q = env.last_transition_quat
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        if yaw0 is None:
            yaw0 = yaw.clone()
        d = pos - start
        live = alive & ~(term | trunc)
        cos = -gravity[:, 2]
        upright_steps += (alive & (cos >= .8)).float()
        min_cos = torch.where(alive, torch.minimum(min_cos, cos), min_cos)
        max_lateral = torch.where(alive, torch.maximum(max_lateral, d[:, 1].abs()), max_lateral)
        t50 = torch.where(alive & torch.isnan(t50) & (d[:, 0] >= .5), torch.full_like(t50, (s + 1) * env.step_dt), t50)
        final_x = torch.where(alive, d[:, 0], final_x)
        final_y = torch.where(alive, d[:, 1], final_y)
        final_yaw = torch.where(alive, (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi, final_yaw)
        if s + 1 == round(4. / env.step_dt):
            x_at_4s = d[:, 0].clone()
        alive = live
    results = []
    for i, c in enumerate(candidates):
        sl = slice(i * k, (i + 1) * k)
        ok = alive[sl]
        speed = (final_x[sl] - x_at_4s[sl]) / (args.seconds - 4.) * 100
        results.append({
            'name': c['name'], 'period_s': teachers[i].period, 'envs': k,
            'survived_full_run': float(ok.float().mean()),
            'steady_speed_cm_s_survivors': float(speed[ok].mean()) if bool(ok.any()) else None,
            'reached_50cm': float((~torch.isnan(t50[sl])).float().mean()),
            'time_to_50cm_s_median': float(t50[sl][~torch.isnan(t50[sl])].median()) if bool((~torch.isnan(t50[sl])).any()) else None,
            'max_lateral_cm_p90': float(torch.quantile(max_lateral[sl], .9)) * 100,
            'lateral_cm_at_end_mean': float(final_y[sl].mean()) * 100,
            'final_abs_yaw_deg_median_survivors': float(final_yaw[sl][ok].abs().median()) * 180 / math.pi if bool(ok.any()) else None,
            'upright_fraction': float(upright_steps[sl].mean()) / steps,
            'min_cos_p10': float(torch.quantile(min_cos[sl], .1)),
        })
        print(json.dumps(results[-1]), flush=True)
    args.output.write_text(json.dumps({'args': {k2: str(v) for k2, v in vars(args).items()}, 'results': results}, indent=1) + '\n')


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(1)
finally:
    if env is not None:
        env.close()
    sys.stdout.flush()
    app.close()
