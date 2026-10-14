#!/usr/bin/env python3
"""Scripted (non-learned) side-recovery feasibility probe.

Drives open-loop joint keyframes from a quasi-static analysis through the normal
SnailEnv action path (action envelope, 1 rad/s target slew, implicit PD), from the
recorded settled LEFT/RIGHT starts, then hands control to an existing crawl policy
and scores each env with the acceptance Trial (stable upright >=0.8 for 0.5 s,
then +5 cm within 30 s). Per-env effort/velocity limits bound the real servo.
Independent feasibility check; writes only under --output.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', default='assets/so101/snail.urdf')
parser.add_argument('--contract', default='configs/policy.json')
parser.add_argument('--physics', default='configs/experiments/recover-t100-001.json')
parser.add_argument('--policy', default='runs/scale-treatment-051-100/exported/policy.pt')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=64)
parser.add_argument('--seed', type=int, default=4242)
parser.add_argument('--mode', choices=('sweep', 'confirm'), default='sweep',
                    help='sweep: nominal/holds/random +-0.1 rad perturbations (run 1); confirm: re-test sweep successes at every effort level')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error('--output must be new')
if args.num_envs > 64:
    parser.error('at most 64 envs')
app = AppLauncher(args).app

env = None
try:
    import numpy as np
    import torch
    from le_snail.evaluation import Trial
    from le_snail.sim.env import SnailEnv, make_config

    args.output.mkdir(parents=True)
    device = args.device or 'cuda:0'
    N = args.num_envs
    # Quasi-static keyframes (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper), rad.
    KEYS = {
        'left': [[-0.61, -1.61, 0.258, -1.106, -0.287, 0.439],
                 [-0.086, -1.545, 0.421, -1.205, -0.914, 0.901],
                 [0.356, 0.559, 0.776, -0.4, -1.458, 0.706]],
        'right': [[0.559, -1.121, 0.237, -1.517, 1.178, 0.95],
                  [-0.649, 0.434, 0.616, -0.054, 1.473, 0.677],
                  [0.231, 0.125, 0.843, -0.627, 0.622, 0.699]],
    }
    EFFORTS = [(0.49, 2.0), (0.8, 2.0), (1.9, 5.45)]   # (N m, rad/s): real rated, sim default, real stall/no-load
    if args.mode == 'confirm':
        EFFORTS.append((1.9, 2.0))   # stall torque at the sim speed cap: isolates torque from speed

    cfg = make_config(args.asset, args.contract, args.physics, num_envs=N, stage='recover', device=device, evaluation=True, initial_side='left', seed=args.seed)
    env = SnailEnv(cfg)
    policy = torch.jit.load(args.policy, map_location=device).eval()
    home, scale, lower, upper = env.home, env.scale, env.lower, env.upper
    jid = env.joint_ids
    rng = np.random.default_rng(args.seed)
    perturb = rng.uniform(-0.1, 0.1, (64, 3, 6))   # identical to the sweep run (seed 4242)
    # Perturbation indices that passed in the sweep run (runs/feasibility-side-recovery-20261014T111208Z).
    CONFIRM = {'left': [9, 14, 29, 34, 35, 44], 'right': [14, 26, 48, 56, 62]}

    def plan(side):
        """Per env: (effort index, label, 3x6 keyframes)."""
        out = []
        for i in range(N):
            if args.mode == 'sweep':
                e, k = i % 3, i // 3
                kind = 'nominal' if k == 0 else ('holds' if k == 1 else 'perturbed')
                ks = np.array(KEYS[side]) + (perturb[i] if kind == 'perturbed' else 0.)
                out.append((e, kind if kind != 'perturbed' else f'perturbed{i}', ks))
            else:
                cands = CONFIRM[side]
                c = cands[(i // len(EFFORTS)) % len(cands)]
                out.append((i % len(EFFORTS), f'perturbed{c}', np.array(KEYS[side]) + perturb[c]))
        return out
    k_nom, d_nom = env.physics['stiffness_nm_rad'], env.physics['damping_nm_s_rad']

    def apply_limits(effort, vlimit):
        n_j = env.robot.num_joints
        eff = env.robot.data.joint_effort_limits.clone()
        vel = env.robot.data.joint_vel_limits.clone()
        for c in jid:
            eff[:, c] = effort
            vel[:, c] = vlimit
        env.robot.write_joint_effort_limit_to_sim(eff)
        env.robot.write_joint_velocity_limit_to_sim(vel)
        got = env.robot.root_physx_view.get_dof_max_forces()
        return got[:, jid[0]].tolist()

    results = []
    traces = {}
    for stage in ('recover', 'robust'):
        for side in ('left', 'right'):
            env.cfg.stage, env.cfg.initial_side = stage, side
            variant = plan(side)
            effort = torch.tensor([EFFORTS[v[0]][0] for v in variant], device=device)
            vlimit = torch.tensor([EFFORTS[v[0]][1] for v in variant], device=device)
            obs, _ = env.reset(seed=args.seed)
            applied = apply_limits(effort, vlimit)
            dt = env.step_dt
            with torch.no_grad():
                zero = torch.zeros((N, 6), device=device)
                for _ in range(round(2 / dt)):
                    obs, *_ = env.step(zero)
                start = env.robot.data.root_pos_w.clone()
                env.start_pos[:] = start
                env.episode_length_buf[:] = 0
                env.fallen_time.zero_()
                keys = []
                for i in range(N):
                    ks = variant[i][2]
                    seq = [k for k in ks] + [home.cpu().numpy()]
                    keys.append(torch.tensor(np.array(seq), dtype=torch.float32, device=device))
                keys = torch.stack(keys)  # N, 4, 6
                hold = torch.tensor([10 if v[1] == 'holds' else 2 for v in variant], device=device)
                key_i = torch.zeros(N, dtype=torch.long, device=device)
                held = torch.zeros(N, dtype=torch.long, device=device)
                scripted = torch.ones(N, dtype=torch.bool, device=device)
                script_end_t = torch.full((N,), float('nan'), device=device)
                trials = [Trial(kind=side) for _ in range(N)]
                up_trace = []
                peak_tau = torch.zeros((N, 6), device=device)
                sat_steps = torch.zeros((N, 6), device=device)
                script_steps = torch.zeros(N, device=device)
                fault_kind = [None] * N
                max_up = torch.full((N,), -2., device=device)
                steps = round(30 / dt)
                ar = torch.arange(N, device=device)
                for step in range(1, steps + 1):
                    goal = keys[ar, key_i.clamp(max=3)]
                    act_script = ((goal - home) / scale).clamp(-1, 1)
                    act_pol = policy(obs['policy'])
                    actions = torch.where(scripted[:, None], act_script, act_pol)
                    obs, _, term, trunc, _ = env.step(actions)
                    # keyframe sequencing: advance once the slewed target reached the (envelope-clipped) goal and held
                    goal_clipped = torch.clamp(home + act_script * scale, lower, upper)
                    at_goal = (env.target - goal_clipped).abs().max(dim=1).values < 1e-3
                    held = torch.where(at_goal & scripted, held + 1, torch.zeros_like(held))
                    adv = scripted & (held >= torch.where(key_i == 3, torch.full_like(hold, 20), hold))
                    key_i = torch.where(adv, key_i + 1, key_i)
                    held = torch.where(adv, torch.zeros_like(held), held)
                    done_script = scripted & (key_i > 3)
                    script_end_t = torch.where(done_script, torch.full_like(script_end_t, step * dt), script_end_t)
                    scripted &= ~done_script
                    # torque estimate of the implicit PD (nominal gains; robust stage scales gains +-20%)
                    q = env.robot.data.joint_pos[:, jid]
                    qd = env.robot.data.joint_vel[:, jid]
                    tau = k_nom * (env.applied_target - q) - d_nom * qd
                    live = torch.tensor([not t.finished for t in trials], device=device)
                    phase = (live & (scripted | done_script))[:, None].float()   # scripted righting phase only, before trial end/reset
                    clipped = torch.minimum(tau.abs(), effort[:, None])
                    peak_tau = torch.maximum(peak_tau, clipped * phase)
                    sat_steps += (tau.abs() >= effort[:, None]).float() * phase
                    script_steps += phase[:, 0]
                    pos, gravity, _ = env.last_transition
                    up = -gravity[:, 2]
                    max_up = torch.maximum(max_up, torch.where(torch.tensor([not t.finished for t in trials], device=device), up, max_up))
                    disp = (pos - start).cpu().numpy()
                    upc = up.cpu().numpy()
                    termc = term.cpu().numpy()
                    if bool(term.any()):
                        lat = (pos[:, 1] - env.start_pos[:, 1]).abs() > 0.15
                        low = pos[:, 2] < -0.03
                        for i in torch.nonzero(term).flatten().tolist():
                            if fault_kind[i] is None and not trials[i].finished:
                                fault_kind[i] = 'lateral>0.15m' if bool(lat[i]) else ('base_z<-0.03' if bool(low[i]) else 'other(protected/invalid)')
                    for i, t in enumerate(trials):
                        t.update(step * dt, float(disp[i, 0]), float(disp[i, 1]), float(upc[i]), fault=bool(termc[i]))
                    up_trace.append(upc.copy())
                    if all(t.finished for t in trials):
                        break
            traces[f'{stage}-{side}'] = np.array(up_trace)
            for i, t in enumerate(trials):
                results.append(dict(stage=stage, side=side, env=i, effort_nm=EFFORTS[variant[i][0]][0], vel_limit=EFFORTS[variant[i][0]][1],
                                    variant=variant[i][1], success=t.success, reason=t.reason, recovery_time_s=t.recovery_time_s,
                                    script_end_s=float(script_end_t[i]), max_upright=float(max_up[i]), last_time_s=t.last_time_s,
                                    keys=np.round(variant[i][2], 4).tolist(), fault_kind=fault_kind[i],
                                    script_peak_clipped_pd_torque_nm=peak_tau[i].tolist(),
                                    script_saturation_fraction=(sat_steps[i] / script_steps[i].clamp(min=1)).tolist(),
                                    applied_effort_limit=applied[i]))
            sel = [r for r in results if r['stage'] == stage and r['side'] == side]
            for e, (lim, vel) in enumerate(EFFORTS):
                g = [r for r in sel if r['effort_nm'] == lim and r['vel_limit'] == vel]
                rec = [r for r in g if r['recovery_time_s'] is not None]
                print(f'{stage} {side} effort {lim} vel {vel}: recovered {len(rec)}/{len(g)}, passed(5cm) {sum(r["success"] for r in g)}/{len(g)}, '
                      f'median recovery {np.median([r["recovery_time_s"] for r in rec]) if rec else None}', flush=True)
    (args.output / 'results.json').write_text(json.dumps(dict(torque_note='PD torque = nominal k*(target-q)-d*qd (robust stage randomizes gains +-20%), clipped at the per-env effort limit; counted only during the scripted phase of unfinished trials', args={k: str(v) for k, v in vars(args).items()}, keys=KEYS, efforts=EFFORTS, results=results), indent=1))
    np.savez_compressed(args.output / 'upright_traces.npz', **traces)
    print('Probe complete', flush=True)
except BaseException:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    if env is not None:
        env.close()   # close the env before Kit, or the container can hang on shutdown
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
