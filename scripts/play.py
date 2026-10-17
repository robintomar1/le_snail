#!/usr/bin/env python3
"""Watch an exported policy bundle in the Isaac Sim GUI (#15). No training, no scoring, writes nothing.

A few robots side by side, each with its own start (default: two upright, two left, two right), driven by the
bundle's policy deterministically (no exploration noise), paced to real time. Like evaluate.py: robust-stage
physics from the bundle unless --stage crawl (nominal) or --physics-override, and every (re)start holds zero
action for 2 s to settle. Robots whose episode ends (lane exit, fallen limit, timeout) restart on their own; all
robots restart together every --round-s. Close the window or press Ctrl+C to stop.

Needs a display: run through scripts/container.sh with LE_SNAIL_DISPLAY=1 (and DISPLAY set on the host).
"""
import argparse
from pathlib import Path
import sys
import time
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle', type=Path, required=True)
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--physics-override', type=Path)
p.add_argument('--stage', choices=('robust', 'crawl'), default='robust', help="'crawl' = nominal physics")
p.add_argument('--starts', nargs='+', choices=('upright', 'left', 'right'), default=['upright', 'left', 'right'])
p.add_argument('--per-start', type=int, default=2)
p.add_argument('--spacing', type=float, default=1.0, help='metres between robots')
p.add_argument('--round-s', type=float, default=40., help='restart all robots every this many simulated seconds (0 = never)')
p.add_argument('--speed', type=float, default=1., help='playback speed relative to real time (0 = as fast as possible)')
p.add_argument('--seed', type=int, default=0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
env = None
try:
    import torch
    from le_snail.bundle import load_bundle
    from le_snail.sim.env import SnailEnv, make_config

    contract, manifest = load_bundle(args.bundle)
    device = args.device or 'cuda:0'
    sides = [side for side in args.starts for _ in range(args.per_start)]
    n = len(sides)

    class ViewerEnv(SnailEnv):
        """Evaluation resets, but each robot keeps its own start side."""

        def _reset_idx(self, env_ids):
            if env_ids is None:
                env_ids = self.robot._ALL_INDICES
            for side in set(sides):
                mask = torch.tensor([sides[i] == side for i in env_ids.tolist()], device=env_ids.device)
                if bool(mask.any()):
                    self.cfg.initial_side = side
                    super()._reset_idx(env_ids[mask])

    cfg = make_config(args.asset, args.bundle / 'policy.json', args.physics_override or args.bundle / 'sim.json', num_envs=n,
                      stage=args.stage, device=device, evaluation=True, seed=args.seed)
    cfg.scene.env_spacing = args.spacing
    env = ViewerEnv(cfg)
    policy = torch.jit.load(str(args.bundle / 'policy.pt'), map_location=device).eval()
    origins = env.scene.env_origins
    centre = origins.mean(dim=0).cpu()
    span = float((origins.max(dim=0).values - origins.min(dim=0).values)[:2].max()) + 1.
    env.sim.set_camera_view(eye=(centre + torch.tensor([-0.4 * span, -0.9 * span, 0.6 * span])).tolist(), target=(centre + torch.tensor([0.25, 0., 0.])).tolist())
    print(f"bundle {args.bundle} ({manifest.get('stage', '?')}), physics stage {args.stage}; +x is forward", flush=True)
    for i, side in enumerate(sides):
        x, y, _ = origins[i].tolist()
        print(f'  robot {i}: {side:7s} start at x={x:+.1f} y={y:+.1f}', flush=True)

    settle = round(2. / contract.dt)
    hold = torch.full((n,), settle, device=device)
    zeros = torch.zeros((n, 6), device=device)
    obs, _ = env.reset(seed=args.seed)
    if hasattr(policy, 'reset'):
        policy.reset()
    round_steps = round(args.round_s / contract.dt)
    step = rounds = 0
    with torch.no_grad():
        while app.is_running():
            tick = time.monotonic()
            actions = policy(obs['policy'])
            actions = torch.where((hold > 0)[:, None], zeros, actions)
            obs, _, term, trunc, _ = env.step(actions)
            hold = (hold - 1).clamp_min(0)
            ended = term | trunc
            if bool(ended.any()):
                hold[ended] = settle
                if hasattr(policy, 'reset_envs'):
                    policy.reset_envs(ended.nonzero().flatten())
            step += 1
            if round_steps > 0 and step % round_steps == 0:
                obs, _ = env.reset(seed=args.seed + rounds + 1)
                if hasattr(policy, 'reset'):
                    policy.reset()
                hold[:] = settle
                rounds += 1
            if args.speed > 0:
                time.sleep(max(0., contract.dt / args.speed - (time.monotonic() - tick)))
except KeyboardInterrupt:
    pass
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
