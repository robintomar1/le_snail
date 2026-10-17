#!/usr/bin/env python3
"""Long-run crawl endurance in sim (#14, #15): how far does each policy keep moving? No training.

Every bundle (same contract) drives an equal share of the environments from upright starts for --seconds, with the
lane limit switched off (turning is allowed); falls do not end a run, but lying down for the physics' fallen limit
does. Per bundle: path length (every physics-policy step, and on 1 s samples, which ignores the back-and-forth rocking within a
stroke) and straight-line distance from the start, mean path speed, the share of time upright
(cos >= 0.5), falls (cos < 0.5 for 0.5 s) and recoveries (stable upright again), runs ended fallen, total turning,
and path length per 30 s window, so a policy that stops moving shows up. Physics: the first bundle's sim.json unless
--physics. Writes only --output (JSON).
"""
import argparse
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('bundles', type=Path, nargs='+')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--physics', type=Path)
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--stage', choices=('robust', 'crawl'), default='robust')
p.add_argument('--envs-per-bundle', type=int, default=32)
p.add_argument('--seconds', type=float, default=180.)
p.add_argument('--seed', type=int, default=21)
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
    from le_snail.bundle import load_bundle
    from le_snail.sim.env import SnailEnv, make_config

    contracts = [load_bundle(b)[0] for b in args.bundles]
    if len({c.fingerprint for c in contracts}) != 1:
        raise SystemExit('All bundles must share one policy contract')

    class NoLaneEnv(SnailEnv):
        def _get_dones(self):
            self.start_pos[:, 1] = self.robot.data.root_pos_w[:, 1]   # lateral displacement is always 0: no lane exit
            return super()._get_dones()

    device = args.device or 'cuda:0'
    k = args.envs_per_bundle
    n = k * len(args.bundles)
    physics = args.physics or args.bundles[0] / 'sim.json'
    cfg = make_config(args.asset, args.bundles[0] / 'policy.json', physics, num_envs=n, stage=args.stage, device=device,
                      evaluation=True, initial_side='upright', seed=args.seed)
    cfg.episode_length_s = args.seconds + 60.
    env = NoLaneEnv(cfg)
    policies = [torch.jit.load(str(b / 'policy.pt'), map_location=device).eval() for b in args.bundles]
    dt = env.step_dt
    obs, _ = env.reset(seed=args.seed)
    for pol in policies:
        if hasattr(pol, 'reset'):
            pol.reset()
    window = round(30. / dt)
    steps = round(args.seconds / dt)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    start = env.robot.data.root_pos_w[:, :2].clone()
    prev = start.clone()
    path = torch.zeros(n, device=device)
    windows = torch.zeros((n, math.ceil(steps / window)), device=device)
    upright_steps = torch.zeros(n, device=device)
    down_steps = torch.zeros(n, device=device)
    down = torch.zeros(n, dtype=torch.bool, device=device)
    falls = torch.zeros(n, device=device)
    recoveries = torch.zeros(n, device=device)
    stable_steps = torch.zeros(n, device=device)
    turning = torch.zeros(n, device=device)
    end_xy = start.clone()
    every = round(1. / dt)
    samples = [start.clone()]   # base xy once per second; path on these ignores the rocking within each stroke
    ended_fallen = torch.zeros(n, dtype=torch.bool, device=device)

    def yaw_of(q):
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    prev_yaw = yaw_of(env.robot.data.root_quat_w)
    with torch.no_grad():
        for s in range(steps):
            o = obs['policy']
            action = torch.cat([pol(o[i * k:(i + 1) * k]) for i, pol in enumerate(policies)])
            obs, _, term, trunc, _ = env.step(action)
            pos, gravity, _ = env.last_transition
            xy = pos[:, :2]
            yaw = yaw_of(env.last_transition_quat)
            step_len = torch.linalg.vector_norm(xy - prev, dim=1)
            path += torch.where(alive, step_len, torch.zeros_like(step_len))
            windows[:, s // window] += torch.where(alive, step_len, torch.zeros_like(step_len))
            turning += torch.where(alive, ((yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi).abs(), torch.zeros_like(yaw))
            cos = -gravity[:, 2]
            upright_steps += (alive & (cos >= .5)).float()
            down_steps = torch.where(cos < .5, down_steps + 1, torch.zeros_like(down_steps))
            fell = alive & ~down & (down_steps >= round(.5 / dt))
            falls += fell.float()
            down |= fell
            stable_steps = torch.where(cos >= .8, stable_steps + 1, torch.zeros_like(stable_steps))
            recovered = alive & down & (stable_steps >= round(.5 / dt) + 1)
            recoveries += recovered.float()
            down &= ~recovered
            end_xy = torch.where(alive[:, None], xy, end_xy)
            if (s + 1) % every == 0:
                samples.append(end_xy.clone())
            ended = alive & (term | trunc)
            ended_fallen |= ended & (cos < .5)
            alive &= ~(term | trunc)
            prev = env.robot.data.root_pos_w[:, :2].clone()
            prev_yaw = yaw_of(env.robot.data.root_quat_w)
    track = torch.stack(samples, dim=1)   # (n, seconds + 1, 2); frozen after a run ends
    path_1s = torch.linalg.vector_norm(track.diff(dim=1), dim=2).sum(dim=1)
    results = []
    for i, b in enumerate(args.bundles):
        sl = slice(i * k, (i + 1) * k)
        net = torch.linalg.vector_norm(end_xy[sl] - start[sl], dim=1)
        results.append({
            'bundle': str(b), 'envs': k, 'seconds': args.seconds,
            'path_m_mean': float(path[sl].mean()), 'path_m_min': float(path[sl].min()), 'path_m_max': float(path[sl].max()),
            'path_1s_m_mean': float(path_1s[sl].mean()), 'path_1s_m_min': float(path_1s[sl].min()), 'path_1s_m_max': float(path_1s[sl].max()),
            'net_distance_m_mean': float(net.mean()), 'net_distance_m_min': float(net.min()), 'net_distance_m_max': float(net.max()),
            'path_speed_cm_s': float(path[sl].mean()) / args.seconds * 100,
            'path_cm_per_30s_window_mean': [round(float(v) * 100, 1) for v in windows[sl].mean(0)],
            'upright_time_fraction': float(upright_steps[sl].mean()) / steps,
            'falls_per_run': float(falls[sl].mean()), 'recoveries_per_run': float(recoveries[sl].mean()),
            'runs_ended_early': float((~alive[sl]).float().mean()), 'runs_ended_fallen': float(ended_fallen[sl].float().mean()),
            'turning_deg_per_min': float(turning[sl].mean()) * 180 / math.pi / (args.seconds / 60),
        })
        print(json.dumps(results[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez_compressed(args.output.with_suffix('.npz'), track_xy_m=(track - start[:, None, :]).cpu().numpy(),
                        bundles=np.array([str(b) for b in args.bundles]), envs_per_bundle=k)
    args.output.write_text(json.dumps({'args': {k2: str(v) for k2, v in vars(args).items()}, 'physics': str(physics),
                                       'results': results}, indent=1) + '\n')


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
