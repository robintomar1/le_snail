#!/usr/bin/env python3
"""Diagnostic: is the servo effort limit enforced in the simulation? No training (#14).

For each physics config: 4 upright envs, stage robust (ranges applied at reset as in evaluation), hold home for
--seconds, then command shoulder_lift and elbow far from home for --seconds. Prints the PhysX max joint forces
and the joint positions reached, so a tiny cap (which cannot hold the arm against gravity) must differ visibly.
"""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--physics', nargs='+', required=True)
p.add_argument('--contract', default='configs/policy-50hz-001.json')
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--seconds', type=float, default=2.)
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
import torch
from le_snail.sim.env import SnailEnv, make_config

results = {}
for path in args.physics:
    cfg = make_config(args.asset, args.contract, path, num_envs=4, stage='robust', device=args.device or 'cuda:0', evaluation=True, seed=1)
    env = SnailEnv(cfg)
    env.reset(seed=1)
    dt = env.step_dt
    row = {'physx_max_forces': env.robot.root_physx_view.get_dof_max_forces()[0].tolist(),
           'data_effort_limits': env.robot.data.joint_effort_limits[0].tolist(),
           'actuator_effort_limit': env.robot.actuators['servos'].effort_limit[0].tolist()}
    with torch.no_grad():
        for _ in range(round(args.seconds / dt)):
            env.step(torch.zeros((4, 6), device=env.device))
        row['q_after_hold'] = env.robot.data.joint_pos[0, env.joint_ids].tolist()
        push = torch.zeros((4, 6), device=env.device)
        push[:, 1], push[:, 2] = 1., -1.
        for _ in range(round(args.seconds / dt)):
            env.step(push)
        row['q_after_push'] = env.robot.data.joint_pos[0, env.joint_ids].tolist()
        row['applied_torque'] = env.robot.data.applied_torque[0, env.joint_ids].tolist()
        row['computed_torque'] = env.robot.data.computed_torque[0, env.joint_ids].tolist()
    results[path] = row
    print(json.dumps({path: row}), flush=True)
    env.close()
args.output.write_text(json.dumps(results, indent=1) + '\n')
app.close()
