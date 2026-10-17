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
import os
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--commands', type=Path, help='.npz with command_sent_ns (n,), command_goal_ticks (n, k) or .csv with those columns')
p.add_argument('--joints', nargs='+', help='joint name for each command column, e.g. shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll')
p.add_argument('--log', help='Stage 1 actuator log stem (runs/actuator-id/stage1/NAME, .npz + .json); replaces --commands/--joints/--tick-*')
p.add_argument('--joint-map', type=Path, default=Path('runs/actuator-id/twin-joint-map.json'), help='with --log: ticks -> URDF radians')
p.add_argument('--wrist-offset', choices=('primary', 'alternative'), default='primary', help='with --log: wrist_flex offset_ticks or alternative_offset_ticks')
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
if (args.log is None) == (args.commands is None) or (args.commands is not None and not args.joints):
    p.error('give either --log, or --commands with --joints')
if args.output.exists():
    p.error('--output must be new')
app = AppLauncher(args).app
try:
    import numpy as np
    import torch
    from le_snail.bundle import file_hash
    from le_snail.contract import PolicyContract
    from le_snail.sim.env import SnailEnv, make_config

    initial_rad = None
    window_ns = None
    if args.log:
        # Stage 1 log (#12): drive active from the Torque_Enable 1 write to the Torque_Enable 0 write (after that
        # robin's hand holds the arm); goals = the start-up Goal_Position write plus the 50 Hz commands; initial
        # state = the last Present_Position sampled before torque-on (the limp rest pose).
        from le_snail.actuator_id import decode_state
        log = np.load(f'{args.log}.npz')
        meta_log = json.loads(Path(f'{args.log}.json').read_text())
        joint_map = json.loads(args.joint_map.read_text())['joints']
        by_id = {spec['id']: (name, spec) for name, spec in joint_map.items() if 'sign' in spec}
        ids = log['ids'].tolist()
        args.joints = [by_id[i][0] for i in ids]
        args.tick_sign = [float(by_id[i][1]['sign']) for i in ids]
        args.tick_offset = [float(by_id[i][1]['alternative_offset_ticks' if by_id[i][0] == 'wrist_flex' and args.wrist_offset == 'alternative'
                                              else 'offset_ticks']) for i in ids]
        enables = [w for w in meta_log['writes'] if w['register'] == 'Torque_Enable']
        on = next(w['sent_ns'] for w in enables if all(v == 1 for v in w['values'].values()))
        off = next((w['sent_ns'] for w in enables if w['sent_ns'] > on and all(v == 0 for v in w['values'].values())), int(log['sent_ns'][-1]))
        window_ns = (on, off)
        goals = [(w['sent_ns'], [w['values'][str(i)] for i in ids]) for w in meta_log['writes'] if w['register'] == 'Goal_Position']
        goals += list(zip(log['command_sent_ns'].tolist(), log['command_goal_ticks'].tolist()))
        goals = sorted(g for g in goals if g[0] <= off)
        before = [g for g in goals if g[0] <= on]
        goals = ([(on, before[-1][1])] if before else []) + [g for g in goals if g[0] > on]
        sent_ns = np.array([g[0] for g in goals], dtype=np.int64)
        ticks = np.array([g[1] for g in goals], dtype=np.float64)
        sample_ns = (log['sent_ns'] + log['received_ns']) // 2
        rest = max(int(np.searchsorted(sample_ns, on, side='right')) - 1, 0)
        rest_ticks = np.array([decode_state(bytes(row))['position'] for row in log['state_raw'][rest]], dtype=np.float64)
        initial_rad = np.array(args.tick_sign) * (rest_ticks - np.array(args.tick_offset)) * 2 * math.pi / 4096
        args.settle_s = 0.
    elif args.commands.suffix == '.npz':
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
    t0_ns = window_ns[0] if window_ns else int(sent_ns[0])
    send_s = (sent_ns - t0_ns) * 1e-9

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

    class TwinEnv(SnailEnv):
        allow_fixed_base = True

    env = TwinEnv(cfg)
    robot, sim = env.robot, env.sim
    device = env.device
    names = list(contract.joint_names)
    columns = [names.index(j) for j in args.joints]
    joint_ids = env.joint_ids
    target = torch.tensor(contract.home_rad, device=device)
    first = torch.tensor(goal_rad[0], dtype=torch.float32, device=device)
    target[columns] = first
    # Start at rest: at the logged rest pose (--log) or at the first commanded pose.
    q = robot.data.default_joint_pos.clone()
    q[0, joint_ids] = target
    if initial_rad is not None:
        q[0, [joint_ids[c] for c in columns]] = torch.tensor(initial_rad, dtype=torch.float32, device=device)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    root = robot.data.default_root_state.clone()
    root[0, 2] = 0.4
    robot.write_root_pose_to_sim(root[:, :7])
    physx_view = robot.root_physx_view
    rows = []
    end_s = (window_ns[1] - t0_ns) * 1e-9 if window_ns else send_s[-1] + 1.
    total = round((args.settle_s + end_s) / physics_dt)
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
    source = Path(f'{args.log}.npz') if args.log else args.commands
    meta = {'commands': str(source), 'commands_sha256': file_hash(source), 'log': args.log, 't0_ns': t0_ns,
            'torque_window_ns': list(window_ns) if window_ns else None, 'joint_map': str(args.joint_map) if args.log else None,
            'joint_map_sha256': file_hash(args.joint_map) if args.log else None, 'wrist_offset': args.wrist_offset if args.log else None,
            'initial_rad': None if initial_rad is None else initial_rad.tolist(),
            'base_height_measured_m': float(robot.data.root_pos_w[0, 2]),
            'joints': args.joints, 'tick_offset': offset.tolist(),
            'tick_sign': sign.tolist(), 'physics': args.physics, 'physics_sha256': file_hash(args.physics), 'physics_hz': args.physics_hz,
            'stiffness': args.stiffness or physics['stiffness_nm_rad'], 'damping': args.damping or physics['damping_nm_s_rad'],
            'effort_limit': args.effort_limit or physics['effort_limit_nm'], 'fixed_base_height_m': 0.4, 'settle_s': args.settle_s,
            'slew_filter': False, 'action_delay': False, 'zero_order_hold': True, 'script_sha256': file_hash(__file__),
            'solver_torque': 'PhysX get_dof_projected_joint_forces (solver force along each DOF)', 'steps': len(rows)}
    (args.output / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({'steps': len(rows), 'output': str(args.output)}), flush=True)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    # app.close() after an exception here spun at 100% CPU for hours (first self-test); exit hard instead.
    os._exit(1)
finally:
    if 'env' in globals():
        env.close()
    sys.stdout.flush()
    app.close()
