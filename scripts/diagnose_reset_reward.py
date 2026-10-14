#!/usr/bin/env python3
"""Measure reset settling and policy reward without changing training behavior."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error('--output must be new')
app = AppLauncher(args).app

try:
    import torch
    from le_snail.bundle import load_bundle
    from le_snail.sim.env import SnailEnv, make_config

    contract, manifest = load_bundle(args.bundle)
    args.output.mkdir(parents=True)
    cfg = make_config('assets/so101/snail.urdf', args.bundle / 'policy.json',
                      args.bundle / 'sim.json', num_envs=1, stage='crawl',
                      evaluation=True, seed=7, device=args.device or 'cuda:0')
    env = SnailEnv(cfg)
    policy = torch.jit.load(str(args.bundle / 'policy.pt'), map_location=env.device).eval()
    zero = torch.zeros((1, 6), device=env.device)
    report = {'stage': 'crawl', 'seed': 7, 'policy_sha256': manifest['policy_sha256'],
              'conditions': {}, 'training_behavior_changed': False}

    def sample(step, start, reward):
        pos, gravity, _ = env.last_transition
        return {'time_s': step * env.step_dt,
                'forward_m': float(pos[0, 0] - start[0, 0]),
                'lateral_m': float(pos[0, 1] - start[0, 1]),
                'height_m': float(pos[0, 2]), 'upright_cos': float(-gravity[0, 2]),
                'reward': float(reward[0])}

    # Nominal mechanics isolate reward effects from robustness randomization.
    with torch.no_grad():
        for side in ('upright', 'left', 'right'):
            env.cfg.initial_side = side
            for mode in ('zero_raw', 'zero_settled', 'zero_restored', 'policy_raw', 'policy_settled'):
                obs, _ = env.reset(seed=7)
                if mode.endswith(('settled', 'restored')):
                    for _ in range(40):
                        obs, _, term, trunc, _ = env.step(zero)
                        if bool(term.any() or trunc.any()):
                            raise RuntimeError('Fault during diagnostic settling')
                if mode == 'zero_restored':
                    # Candidate reset: measured simulated resting pose and joint
                    # angles, zero velocities. This changes only this diagnostic.
                    root = env.robot.data.root_link_pose_w.clone()
                    joints = env.robot.data.joint_pos.clone()
                    env.reset(seed=7)
                    env.robot.write_root_pose_to_sim(root)
                    env.robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
                    env.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
                    env.previous_q.copy_(joints[:, env.joint_ids])
                    env.filtered_velocity.zero_()
                    env.observation_step = -1
                    obs = env._get_observations()
                start = env.robot.data.root_pos_w.clone()
                env.start_pos.copy_(start)
                env.episode_length_buf.zero_()
                env.fallen_time.zero_()
                records = []
                for step in range(1, 101):
                    action = policy(obs['policy']) if mode.startswith('policy') else zero
                    obs, reward, term, trunc, _ = env.step(action)
                    row = sample(step, start, reward)
                    row.update(terminated=bool(term[0]), truncated=bool(trunc[0]))
                    records.append(row)
                    if row['terminated'] or row['truncated']:
                        break
                key = f'{side}-{mode}'
                (args.output / f'{key}.json').write_text(json.dumps(records, indent=2) + '\n')
                result = {'steps': len(records), 'reward_sum': sum(x['reward'] for x in records),
                          'first_2s_reward': sum(x['reward'] for x in records if x['time_s'] <= 2),
                          'first_2s_forward_m': next((x['forward_m'] for x in records if x['time_s'] == 2), None),
                          'final': records[-1]}
                report['conditions'][key] = result
                print(key, json.dumps(result), flush=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Reset/reward diagnosis complete: {args.output}', flush=True)
except BaseException:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    if 'env' in globals():
        env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
