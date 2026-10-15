#!/usr/bin/env python3
"""Record scripted get-up attempts with full state for reward design; zero training (#6).

Per side, from the settled side start: all 64 seed-4242 +-0.1 rad perturbations of the confirmed
keyframe plan (successes and failures), the two confirmed variants x 64 replicas, lying still, and
random actions (flailing). Keyframes run through the normal action path with holds in seconds; after
the last keyframe the targets stay at home. Saves per step: base-frame gravity, joint positions and
targets, base position/orientation, whole-body centre of mass and upright cosine, plus per-trajectory
labels. Writes only under --output.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--contract', default='configs/policy-50hz-001.json')
p.add_argument('--physics', default='configs/experiments/physics-v2-50hz-001.json')
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--confirmation', default='runs/feasibility-side-recovery-20261014T111801Z/results.json')
p.add_argument('--seconds', type=float, default=8.)
p.add_argument('--max-process-seconds', type=float, default=1200.)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new; evidence is never overwritten')
started = time.monotonic()
args.output.mkdir(parents=True)
app = AppLauncher(args).app
env = None

try:
    import numpy as np
    import torch
    from le_snail.bundle import file_hash
    from le_snail.evaluation import Trial
    from le_snail.sim.env import SnailEnv, make_config

    N, SEED = 64, 4242
    cfg = make_config(args.asset, args.contract, args.physics, num_envs=N, stage='recover',
                      evaluation=True, initial_side='left', device=args.device or 'cuda:0', seed=SEED)
    env = SnailEnv(cfg)
    device, dt = env.device, env.step_dt
    source = json.loads(Path(args.confirmation).read_text())
    perturb = np.random.default_rng(SEED).uniform(-.1, .1, (64, 3, 6))
    masses = env.robot.root_physx_view.get_masses().to(device)
    ar = torch.arange(N, device=device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    report = {'schema_version': 1, 'training_performed': False, 'num_envs': N, 'policy_dt': dt, 'seconds': args.seconds,
              'contract_policy_hz': env.contract.policy_hz,
              'provenance': {str(f): file_hash(f) for f in (args.asset, args.contract, args.physics, args.confirmation, __file__, 'src/le_snail/sim/env.py')},
              'batches': []}

    def confirmed_ids(side):
        variants = {'left': (9, 34), 'right': (48, 62)}[side]
        ids = [variants[i % 2] for i in range(N)]
        for v in variants:
            match = [r for r in source['results'] if r['stage'] == 'recover' and r['side'] == side and r['variant'] == f'perturbed{v}'
                     and r['effort_nm'] == .8 and r['vel_limit'] == 2. and r['success']]
            ids[match[0]['env']] = v
        return ids

    def run(side, kind, variants):
        env.cfg.initial_side = side
        env.reset(seed=SEED)
        zero = torch.zeros((N, 6), device=device)
        for _ in range(round(2 / dt)):
            env.step(zero)
        start = env.robot.data.root_pos_w.clone()
        keys = None
        if variants is not None:
            base = np.asarray(source['keys'][side])
            keys = env.tensor(np.array([np.vstack((base + perturb[v], env.contract.home_rad)) for v in variants]))
        key_i = torch.zeros(N, dtype=torch.long, device=device)
        held = torch.zeros_like(key_i)
        trials = [Trial(kind=side) for _ in range(N)]
        live = torch.ones(N, dtype=torch.bool, device=device)
        names = ('gravity_b', 'joint_q', 'target', 'root_pos', 'root_quat', 'com', 'upright', 'key', 'live')
        rows = {name: [] for name in names}
        for step in range(1, round(args.seconds / dt) + 1):
            assert time.monotonic() - started < args.max_process_seconds, 'Process budget exhausted'
            if kind == 'random':
                actions = torch.randn((N, 6), device=device, generator=generator)
            elif keys is None:
                actions = zero
            else:
                goal = keys[ar, key_i.clamp(max=3)]
                actions = ((goal - env.home) / env.scale).clamp(-1, 1)
            alive = live.clone()
            _, _, terminated, truncated, _ = env.step(actions)
            if keys is not None:
                clipped = torch.clamp(env.home + actions * env.scale, env.lower, env.upper)
                reached = (env.target - clipped).abs().amax(dim=1) < 1e-3
                running = key_i <= 3
                held = torch.where(reached & running, held + 1, 0)
                advance = running & (held >= torch.where(key_i == 3, round(1. / dt), round(.1 / dt)))
                key_i += advance.long()
                held = torch.where(advance, 0, held)
            pos, gravity, _ = env.last_transition
            com = (masses[:, :, None] * env.robot.data.body_com_pos_w).sum(1) / masses.sum(1, keepdim=True)
            up = -gravity[:, 2]
            ended = terminated | truncated
            disp, upc, endc, livec = (pos - start).cpu().numpy(), up.cpu().numpy(), ended.cpu().numpy(), alive.cpu().numpy()
            for i, trial in enumerate(trials):
                if livec[i]:
                    trial.update(step * dt, float(disp[i, 0]), float(disp[i, 1]), float(upc[i]), fault=bool(endc[i]))
            live &= ~ended
            values = (gravity, env.robot.data.joint_pos[:, env.joint_ids], env.target, pos - start,
                      env.last_transition_quat, com - start, up, key_i, alive)
            for name, value in zip(names, values):
                rows[name].append(value.float().cpu().numpy())
        stem = f'{side}-{kind}'
        np.savez_compressed(args.output / f'{stem}.npz', **{k: np.asarray(v) for k, v in rows.items()})
        labels = [{'env': i, 'variant': None if variants is None else int(variants[i]), 'stable_upright_time_s': t.recovery_time_s,
                   'reason': t.reason, 'max_upright': float(np.asarray(rows['upright'])[:, i].max())} for i, t in enumerate(trials)]
        batch = {'side': side, 'kind': kind, 'stem': stem, 'labels': labels,
                 'stable_upright': sum(l['stable_upright_time_s'] is not None for l in labels)}
        report['batches'].append(batch)
        (args.output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
        print(json.dumps({'side': side, 'kind': kind, 'stable_upright': batch['stable_upright'], 'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)

    with torch.no_grad():
        for side in ('left', 'right'):
            run(side, 'perturbed', list(range(N)))
            run(side, 'confirmed', confirmed_ids(side))
            run(side, 'hold', None)
            run(side, 'random', None)
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print('Get-up trajectory collection complete', flush=True)
except BaseException:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    if env is not None:
        env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
