#!/usr/bin/env python3
"""Diagnostic (#12): when does a new policy command reach the joint drives? No training.

One upright env per physics config (nominal stage, so the nominal delay applies). After 1 s at home, a single
step command on shoulder_lift is issued; the target applied to the drives is recorded at every physics substep,
and the time from the command to its first appearance is reported (expected: the configured delay rounded up to
the 5 ms substep; 0 for legacy configs).
"""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--physics', required=True)
p.add_argument('--contract', default='configs/policy-50hz-001.json')
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
import torch
from le_snail.sim.env import SnailEnv, make_config


class LoggingEnv(SnailEnv):
    log = []

    def _apply_action(self):
        super()._apply_action()
        self.log.append((self._sim_step_counter, float(self.applied_target[0, 1])))


cfg = make_config(args.asset, args.contract, args.physics, num_envs=1, stage='recover', evaluation=True, device=args.device or 'cuda:0', seed=0)
env = LoggingEnv(cfg)
env.reset(seed=0)
with torch.no_grad():
    for _ in range(round(1. / env.step_dt)):
        env.step(torch.zeros((1, 6), device=env.device))
    issued = env._sim_step_counter          # the next substep is the first one after the command is issued
    before = float(env.applied_target[0, 1])
    action = torch.zeros((1, 6), device=env.device)
    action[0, 1] = 1.
    for _ in range(10):
        env.step(action)
first = next(counter for counter, value in env.log if counter > issued and abs(value - before) > 1e-6)
result = {'physics': args.physics, 'command_delay_ms': env.physics.get('command_delay_ms'), 'legacy_delay_steps': int(env.delay_steps[0]),
          'physics_dt_ms': env.physics_dt * 1000, 'measured_delay_ms': (first - issued - 1) * env.physics_dt * 1000,
          'queue_slots': env.command_queue.shape[1]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=1) + '\n')
print(json.dumps(result), flush=True)
env.close()
app.close()
