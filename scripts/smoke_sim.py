#!/usr/bin/env python3
"""Exercise floating-base import, all reset modes, drag and finite observations."""
import argparse
import sys
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', default='assets/so101/snail.urdf')
parser.add_argument('--contract', default='configs/policy.json')
parser.add_argument('--physics', default='configs/sim.json')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

try:
    import torch
    from le_snail.sim.env import SnailEnv, make_config

    cfg = make_config(args.asset, args.contract, args.physics, num_envs=4, stage='robust', evaluation=True, device=args.device or 'cuda:0')
    env = SnailEnv(cfg)
    for side in ('upright', 'left', 'right'):
        env.cfg.initial_side = side
        obs, _ = env.reset(seed=7)
        for _ in range(100):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros((4, 6), device=env.device))
            assert obs['policy'].shape == (4, 24)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        print(f'{side}: finite state; inspect geometry and poses before training', flush=True)
    env.close()
    print('Smoke test passed: upright, left, right', flush=True)
except BaseException:
    # Kit shutdown can terminate the process before Python displays a pending error.
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
