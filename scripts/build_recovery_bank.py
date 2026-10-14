#!/usr/bin/env python3
"""Capture and audit a bounded intermediate recovery bank; never train a policy.

Replay confirmed paths through the normal action path. Candidate velocities are
removed before settling, then removed again for an independent two-second hold.
An audit completing successfully does not imply bank_valid: inspect report.json.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', default='assets/so101/snail.urdf')
parser.add_argument('--contract', default='configs/policy.json')
parser.add_argument('--physics', default='configs/experiments/recover-upright-progress-001.json')
parser.add_argument('--confirmation', default='runs/feasibility-side-recovery-20261014T111801Z/results.json')
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error('--output must be new; evidence is never overwritten')
args.output.mkdir(parents=True)
app = AppLauncher(args).app
env = None

try:
    import numpy as np
    import torch
    from isaaclab.utils.math import quat_apply_inverse
    from le_snail.bundle import file_hash
    from le_snail.sim.env import SnailEnv, make_config

    confirmation = json.loads(Path(args.confirmation).read_text())
    seed = int(confirmation['args']['seed'])
    perturbations = np.random.default_rng(seed).uniform(-.1, .1, (64, 3, 6))
    selected = {'left': (9, 34), 'right': (48, 62)}
    cfg = make_config(args.asset, args.contract, args.physics, num_envs=64,
                      stage='recover', evaluation=True, initial_side='left',
                      seed=seed, device=args.device or 'cuda:0')
    env = SnailEnv(cfg)
    assert env.physics['effort_limit_nm'] == .8
    assert env.physics['velocity_limit_rad_s'] == 2.
    assert all(v == 1. for v in env.contract.max_target_speed_rad_s)
    assert env.physics['side_upright_progress_reward_cap'] == 1.
    device, jid = env.device, env.joint_ids
    report = {
        'bank_valid': False, 'physics_validated': False,
        'seed': seed, 'num_envs': 64,
        'provenance': {str(p): file_hash(p) for p in
                       (args.asset, args.contract, args.physics, args.confirmation, __file__)},
        'contract_fingerprint': env.contract.fingerprint,
        'limits': {'effort_nm': .8, 'velocity_rad_s': 2., 'target_slew_rad_s': 1.},
        'gates': {'initial_cosine': [.5, .7], 'hold_s': 2.,
                  'max_planar_excursion_m': 1e-4, 'max_shaping_credit': 1e-4,
                  'max_total_return': 0., 'no_fault_or_passive_get_up': True},
        'paths': [], 'candidates': [], 'valid_counts': {'left': 0, 'right': 0},
    }
    candidates = report['candidates']

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))

    def snapshot(i):
        pose = env.robot.data.root_link_pose_w[i].clone()
        pose[:3] -= env.scene.env_origins[i]
        # Translation has no role in bank pose identity; recenter the lane.
        pose[:2] = 0.
        return {'root_pose': pose.cpu().tolist(),
                'joint_rad': env.robot.data.joint_pos[i, jid].cpu().tolist(),
                'holding_target_rad': env.applied_target[i].cpu().tolist(),
                'upright_cos': float(-env.robot.data.projected_gravity_b[i, 2])}

    with torch.no_grad():
        for side, variants in selected.items():
            env.cfg.initial_side = side
            env.reset(seed=seed)
            zero = torch.zeros((64, 6), device=device)
            for _ in range(round(2 / env.step_dt)):
                env.step(zero)
            env.start_pos.copy_(env.robot.data.root_pos_w)
            env.episode_length_buf.zero_()
            env.fallen_time.zero_()
            source = {}
            for variant in variants:
                matches = [r for r in confirmation['results']
                           if r['stage'] == 'recover' and r['side'] == side
                           and r['variant'] == f'perturbed{variant}'
                           and r['effort_nm'] == .8 and r['vel_limit'] == 2.
                           and r['success']]
                if not matches:
                    raise ValueError(f'No nominal confirmation for {side} p{variant}')
                index = matches[0]['env']
                keys = np.asarray(confirmation['keys'][side]) + perturbations[variant]
                source[index] = (variant, keys)
                report['paths'].append({'side': side, 'variant': variant,
                                        'source_env': index, 'exact_keys': keys.tolist()})
            default = next(iter(source.values()))[1]
            keys = env.tensor(np.stack([np.vstack((source.get(i, (None, default))[1],
                                                   env.contract.home_rad)) for i in range(64)]))
            ar = torch.arange(64, device=device)
            key_i = torch.zeros(64, device=device, dtype=torch.long)
            held = torch.zeros_like(key_i)
            faulted = torch.zeros(64, device=device, dtype=torch.bool)
            captured = {i: 0 for i in source}
            for step in range(1, round(15 / env.step_dt) + 1):
                goal = keys[ar, key_i.clamp(max=3)]
                actions = ((goal - env.home) / env.scale).clamp(-1, 1)
                _, _, term, trunc, _ = env.step(actions)
                faulted |= term | trunc
                up = -env.robot.data.projected_gravity_b[:, 2]
                for i, (variant, _) in source.items():
                    if not faulted[i] and .5 <= float(up[i]) <= .7 and captured[i] < 16:
                        candidates.append({'id': f'{side}-p{variant}-step{step}',
                                           'side': side, 'source_env': i,
                                           'script_step': step, 'keyframe': int(key_i[i]),
                                           'captured': snapshot(i)})
                        captured[i] += 1
                clipped = torch.clamp(env.home + actions * env.scale, env.lower, env.upper)
                at_goal = (env.target - clipped).abs().amax(dim=1) < 1e-3
                held = torch.where(at_goal, held + 1, 0)
                advance = (key_i <= 3) & (held >= torch.where(key_i == 3, 20, 2))
                key_i += advance.long()
                held = torch.where(advance, 0, held)
                if all(bool(key_i[i] > 3 or faulted[i]) for i in source):
                    break
            print(f'Captured {side}: {sum(captured.values())} candidates', flush=True)
            save()

        def install(states, sides):
            """Reset pose and every controller/history baseline without home snap."""
            env.reset(seed=seed)
            pose = env.tensor([s['root_pose'] for s in states])
            pose[:, :3] += env.scene.env_origins
            actual = env.tensor([s['joint_rad'] for s in states])
            holding = env.tensor([s['holding_target_rad'] for s in states])
            q = env.robot.data.default_joint_pos.clone()
            q[:, jid] = actual
            env.robot.write_root_pose_to_sim(pose)
            env.robot.write_root_velocity_to_sim(torch.zeros((64, 6), device=device))
            env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            env.robot.set_joint_position_target(holding, joint_ids=jid)
            env.scene.write_data_to_sim()
            for name in ('target', 'old_target', 'applied_target'):
                getattr(env, name).copy_(holding)
            env.command_queue[:] = holding[:, None, :]
            env.previous_q.copy_(actual)
            env.filtered_velocity.zero_()
            env.start_pos.copy_(pose[:, :3])
            env.start_yaw.copy_(env.yaw(pose[:, 3:7]))
            env.previous_x.copy_(pose[:, 0])
            env.fallen_time.zero_()
            env.episode_length_buf.zero_()
            env.initial_side[:] = torch.tensor([1 if s == 'left' else 2 for s in sides], device=device)
            env.reset_upright.copy_(-quat_apply_inverse(pose[:, 3:7], env.gravity_w)[:, 2])
            env.best_upright_progress.zero_()
            env.stable_upright_steps.zero_()
            env.reached_upright.zero_()
            env.observation_step = -1
            obs = env._get_observations()['policy']
            history_ok = (obs[:, 6:12].abs().amax(dim=1) <= 1e-6)
            history_ok &= ((obs[:, 18:] - (holding - env.home) / env.scale).abs().amax(dim=1) <= 1e-6)
            history_ok &= ((env.robot.data.joint_pos_target[:, jid] - holding).abs().amax(dim=1) <= 1e-6)
            history_ok &= env.robot.data.root_vel_w.abs().amax(dim=1) <= 1e-6
            history_ok &= env.robot.data.joint_vel.abs().amax(dim=1) <= 1e-6
            return holding, history_ok

        def hold(states, sides):
            holding, history_ok = install(states, sides)
            actions = (holding - env.home) / env.scale
            start = env.start_pos.clone()
            maximum = torch.zeros(64, device=device)
            credit = torch.zeros_like(maximum)
            returns = torch.zeros_like(maximum)
            faults = torch.zeros(64, dtype=torch.bool, device=device)
            stable_steps = torch.zeros(64, dtype=torch.long, device=device)
            recovered = torch.zeros_like(faults)
            finite = torch.ones_like(faults)
            limits = ((actions.abs() <= 1 + 1e-6).all(dim=1)
                      & (holding >= env.lower).all(dim=1) & (holding <= env.upper).all(dim=1))
            for _ in range(round(2 / env.step_dt)):
                obs, reward, term, trunc, _ = env.step(actions)
                pos, gravity, _ = env.last_transition
                up = -gravity[:, 2]
                maximum = torch.maximum(maximum, torch.linalg.vector_norm((pos - start)[:, :2], dim=1))
                credit += env.reward_components['upright_progress']
                returns += reward
                faults |= term | trunc
                stable_steps = torch.where(up >= .8, stable_steps + 1, 0)
                recovered |= stable_steps >= round(.5 / env.step_dt) + 1
                finite &= torch.isfinite(obs['policy']).all(dim=1) & torch.isfinite(reward)
                finite &= torch.isfinite(env.robot.data.root_state_w).all(dim=1)
                q = env.robot.data.joint_pos[:, jid]
                limits &= (q >= env.lower).all(dim=1) & (q <= env.upper).all(dim=1)
                history_ok &= (env.applied_target - holding).abs().amax(dim=1) <= 1e-6
            evidence = []
            for i in range(len(candidates)):
                evidence.append({'history_and_zero_velocity_ok': bool(history_ok[i]),
                                 'within_limits': bool(limits[i]), 'finite': bool(finite[i]),
                                 'fault': bool(faults[i]), 'passive_recovery': bool(recovered[i]),
                                 'max_planar_excursion_m': float(maximum[i]),
                                 'shaping_credit': float(credit[i]), 'total_return': float(returns[i])})
            return evidence, [snapshot(i) for i in range(len(candidates))]

        if candidates:
            assert len(candidates) <= 64
            def padded(values):
                return values + [values[0]] * (64 - len(values))
            sides = padded([c['side'] for c in candidates])
            settling, settled = hold(padded([c['captured'] for c in candidates]), sides)
            validation, final = hold(padded(settled), sides)
            for c, state, first, check, end in zip(candidates, settled, settling, validation, final):
                c.update(settled=state, settling=first, validation=check, final=end)
                reasons = []
                if first['fault'] or first['passive_recovery'] or not first['finite']:
                    reasons.append('settling_fault_or_passive_recovery_or_nonfinite')
                if not .5 <= state['upright_cos'] <= .7:
                    reasons.append('settled_cosine_outside_0.5_to_0.7')
                for flag in ('history_and_zero_velocity_ok', 'within_limits', 'finite'):
                    if not check[flag]:
                        reasons.append(flag)
                if check['fault'] or check['passive_recovery']:
                    reasons.append('validation_fault_or_passive_recovery')
                if check['max_planar_excursion_m'] > 1e-4:
                    reasons.append('planar_drift')
                if check['shaping_credit'] > 1e-4:
                    reasons.append('passive_shaping_credit')
                if check['total_return'] > 0:
                    reasons.append('positive_passive_return')
                c['rejection_reasons'] = reasons
                c['valid'] = not reasons
                report['valid_counts'][c['side']] += int(c['valid'])
        report['bank_valid'] = all(report['valid_counts'].values())
        report['decision'] = ('Both sides pass; conditional training gate met.' if report['bank_valid']
                              else 'STOP: at least one side has no valid bank. Do not relax gates or train.')
        save()
        if report['bank_valid']:
            bank = {'provenance': report['provenance'], 'contract_fingerprint': env.contract.fingerprint,
                    'poses': [{'id': c['id'], 'side': c['side'], **c['settled']}
                              for c in candidates if c['valid']]}
            (args.output / 'bank.json').write_text(json.dumps(bank, indent=2, allow_nan=False))
        print(json.dumps({'valid_counts': report['valid_counts'], 'decision': report['decision']}), flush=True)
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
