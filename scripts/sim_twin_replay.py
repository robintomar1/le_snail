#!/usr/bin/env python3
"""Sim twin of the suspended-arm actuator tests (#12): replay a logged servo command sequence; no training.

Reads the laptop logger's commands (command_sent_ns, command_goal_ticks; .npz or .csv) and replays them on the
simulated arm with the base FIXED in the air (no ground contact), physics at --physics-hz (default 200 Hz) and a
zero-order hold: at physics time t the target is the last command sent at or before t. The policy contract's
clamp, slew filter and action delay are NOT used. Ticks -> radians per joint: rad = sign * (ticks - offset) * 2 pi /
4096 (defaults offset 2048, sign +1; pass the #8 mapping when it exists). Joints missing from the log hold home.

Logs at every physics step: t_s, target_rad, joint_pos_rad, joint_vel_rad_s, the PhysX projected joint forces
(solver torque along each joint DOF; get_dof_projected_joint_forces) and Isaac Lab's applied_torque (the explicit PD
estimate, kept only for comparison). Physics (drive gains, effort/velocity caps, masses) come from --physics at its
nominal values unless overridden with --stiffness / --damping / --effort-limit.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--commands', type=Path, required=True, help='.npz with command_sent_ns (n,), command_goal_ticks (n, k) or .csv with those columns')
p.add_argument('--joints', nargs='+', required=True, help='joint name for each command column, e.g. shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll')
p.add_argument('--tick-offset', type=float, nargs='+', help='per joint, default 2048')
p.add_argument('--tick-sign', type=float, nargs='+', help='per joint, default +1')
p.add_argument('--contract', default='configs/policy-50hz-001.json')
p.add_argument('--physics', default='configs/experiments/physics-v2-50hz-001.json')
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--physics-hz', type=float, default=200.)
p.add_argument('--settle-s', type=float, default=1., help='hold the first command before the log clock starts')
p.add_argument('--stiffness', type=float)
p.add_argument('--damping', type=float)
p.add_argument('--effort-limit', type=float)
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new')
app = AppLauncher(args).app
try:
    import numpy as np
    import torch
    from le_snail.bundle import file_hash
    from le_snail.contract import PolicyContract
    from le_snail.sim.env import SnailEnv, make_config

    if args.commands.suffix == '.npz':
        log = np.load(args.commands)
        sent_ns, ticks = np.asarray(log['command_sent_ns'], dtype=np.int64), np.asarray(log['command_goal_ticks'], dtype=np.float64)
    else:
        with args.commands.open() as stream:
            rows = list(csv.DictReader(stream))
        sent_ns = np.array([int(r['command_sent_ns']) for r in rows], dtype=np.int64)
        tick_cols = [k for k in rows[0] if k.startswith('command_goal_ticks')]
        ticks = np.array([[float(r[k]) for k in tick_cols] for r in rows])
    ticks = ticks.reshape(len(sent_ns), -1)
    if ticks.shape[1] != len(args.joints):
        raise SystemExit(f'{ticks.shape[1]} command columns but {len(args.joints)} joint names')
    if np.any(np.diff(sent_ns) < 0):
        raise SystemExit('command_sent_ns must be monotonic')
    offset = np.array(args.tick_offset or [2048.] * len(args.joints))
    sign = np.array(args.tick_sign or [1.] * len(args.joints))
    goal_rad = sign * (ticks - offset) * 2 * math.pi / 4096
    send_s = (sent_ns - sent_ns[0]) * 1e-9

    contract = PolicyContract.load(args.contract)
    physics = json.loads(Path(args.physics).read_text())
    # Integer physics steps per policy period are required by make_config; the twin steps physics directly.
    physics_dt = 1. / args.physics_hz
    cfg = make_config(args.asset, args.contract, args.physics, num_envs=1, stage='recover', evaluation=True,
                      device=args.device or 'cuda:0', seed=0)
    cfg.sim.dt = physics_dt
    cfg.robot_cfg.spawn.fix_base = True
    cfg.robot_cfg.init_state.pos = (0., 0., 0.4)   # suspended: no ground contact
    if args.stiffness is not None:
        cfg.robot_cfg.actuators['servos'].stiffness = args.stiffness
    if args.damping is not None:
        cfg.robot_cfg.actuators['servos'].damping = args.damping
    if args.effort_limit is not None:
        cfg.robot_cfg.actuators['servos'].effort_limit_sim = args.effort_limit
    env = SnailEnv(cfg)
    robot, sim = env.robot, env.sim
    device = env.device
    names = list(contract.joint_names)
    columns = [names.index(j) for j in args.joints]
    joint_ids = env.joint_ids
    target = torch.tensor(contract.home_rad, device=device)
    first = torch.tensor(goal_rad[0], dtype=torch.float32, device=device)
    target[columns] = first
    # Start at the first commanded pose, at rest.
    q = robot.data.default_joint_pos.clone()
    q[0, joint_ids] = target
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    root = robot.data.default_root_state.clone()
    root[0, 2] = 0.4
    robot.write_root_pose_to_sim(root[:, :7])
    physx_view = robot.root_physx_view
    rows = []
    total = round((args.settle_s + send_s[-1] + 1.) / physics_dt)
    cursor = 0
    for step in range(total):
        t = step * physics_dt - args.settle_s
        while cursor + 1 < len(send_s) and send_s[cursor + 1] <= t:
            cursor += 1
        if t >= 0:
            target[columns] = torch.tensor(goal_rad[cursor], dtype=torch.float32, device=device)
        robot.set_joint_position_target(target[None], joint_ids=joint_ids)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(physics_dt)
        projected = physx_view.get_dof_projected_joint_forces()[0, joint_ids]
        rows.append([t, *target.tolist(), *robot.data.joint_pos[0, joint_ids].tolist(), *robot.data.joint_vel[0, joint_ids].tolist(),
                     *projected.tolist(), *robot.data.applied_torque[0, joint_ids].tolist()])
    data = np.array(rows)
    args.output.mkdir(parents=True)
    n = len(names)
    np.savez_compressed(args.output / 'twin.npz', t_s=data[:, 0], target_rad=data[:, 1:1 + n], joint_pos_rad=data[:, 1 + n:1 + 2 * n],
                        joint_vel_rad_s=data[:, 1 + 2 * n:1 + 3 * n], solver_torque_nm=data[:, 1 + 3 * n:1 + 4 * n],
                        pd_estimate_torque_nm=data[:, 1 + 4 * n:1 + 5 * n], joint_names=np.array(names))
    meta = {'commands': str(args.commands), 'commands_sha256': file_hash(args.commands), 'joints': args.joints, 'tick_offset': offset.tolist(),
            'tick_sign': sign.tolist(), 'physics': args.physics, 'physics_sha256': file_hash(args.physics), 'physics_hz': args.physics_hz,
            'stiffness': args.stiffness or physics['stiffness_nm_rad'], 'damping': args.damping or physics['damping_nm_s_rad'],
            'effort_limit': args.effort_limit or physics['effort_limit_nm'], 'fixed_base_height_m': 0.4, 'settle_s': args.settle_s,
            'slew_filter': False, 'action_delay': False, 'zero_order_hold': True, 'script_sha256': file_hash(__file__),
            'solver_torque': 'PhysX get_dof_projected_joint_forces (solver force along each DOF)', 'steps': len(rows)}
    (args.output / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({'steps': len(rows), 'output': str(args.output)}), flush=True)
except BaseException:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    if 'env' in globals():
        env.close()
    sys.stdout.flush()
    app.close()
