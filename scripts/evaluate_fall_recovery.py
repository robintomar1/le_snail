#!/usr/bin/env python3
"""End-to-end test: crawl, get knocked over mid-crawl, recover, continue (#14). No training.

For each trial (seed = --seed + index, sides alternate left/right): reset to the settled upright start, hold
zero action for 2 s (as evaluate.py), reset the bundle's policy if it has reset(), run it, and at --push-s
push the upper arm sideways: a horizontal world-frame force of --push-n newtons, perpendicular to the base
heading, for --push-duration-s, towards the requested side (env 'left' = roll +90 deg, top of the base towards
the robot's right). An instantaneous roll velocity kick (8 rad/s) did not topple the wide base on dev seeds. Success (same rule as a recovery trial): the base falls (cos < 0.5) within 2 s of
the kick, then reaches a stable upright (cos >= 0.8 for 0.5 s), then gains +5 cm of forward x within 30 s
of the kick. Physics: robust stage (as evaluate.py) unless --physics-override. Per-step traces (with the
controller status when available) and optional videos. Writes only under --output.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle', type=Path, required=True)
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--physics-override', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--trials', type=int, default=10)
p.add_argument('--push-s', type=float, default=20.)
p.add_argument('--push-n', type=float, default=10.)
p.add_argument('--push-duration-s', type=float, default=.3)
p.add_argument('--push-body', default='upper_arm_link')
p.add_argument('--window-s', type=float, default=30.)
p.add_argument('--video', action='store_true')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new')
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app
env = None
writer = None
try:
    import numpy as np
    import torch
    from le_snail.bundle import file_hash, load_bundle
    from le_snail.sim.env import SnailEnv, make_config

    contract, manifest = load_bundle(args.bundle)
    args.output.mkdir(parents=True)
    device = args.device or 'cuda:0'
    physics_path = args.physics_override or args.bundle / 'sim.json'
    cfg = make_config(args.asset, args.bundle / 'policy.json', physics_path, num_envs=1, stage='robust', device=device,
                      evaluation=True, seed=args.seed)
    cfg.viewer.resolution = (960, 720)
    env = SnailEnv(cfg, render_mode='rgb_array' if args.video else None)
    policy = torch.jit.load(str(args.bundle / 'policy.pt'), map_location=device).eval()
    dt = contract.dt
    ids, _ = env.robot.find_bodies(args.push_body)
    push_body = torch.tensor(ids, device=device, dtype=torch.int32)
    results = []
    for index in range(args.trials):
        side = ('left', 'right')[index % 2]
        seed = args.seed + index
        env.cfg.initial_side = 'upright'
        obs, _ = env.reset(seed=seed)
        for _ in range(round(2 / dt)):
            obs, *_ = env.step(torch.zeros((1, 6), device=device))
        start = env.robot.data.root_pos_w[0].clone()
        env.start_pos[0] = start
        env.episode_length_buf[:] = 0
        env.fallen_time.zero_()
        if hasattr(policy, 'reset'):
            policy.reset()
        if args.video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(args.output / f'{index:02d}-{side}.mp4'), fps=round(contract.policy_hz))
        rows, kicked, fell_t, upright_t, x_at_upright, success_t, streak, ended = [], None, None, None, None, None, 0, None
        pushing = True
        env.robot.permanent_wrench_composer.reset()
        total = round((args.push_s + args.window_s) / dt)
        with torch.no_grad():
            for step in range(1, total + 1):
                t = step * dt
                if kicked is None and t >= args.push_s:
                    yaw = SnailEnv.yaw(env.robot.data.root_quat_w)[0].item()
                    # Env 'left' (roll +90 deg) puts the top of the base towards the robot's right: push that way.
                    sign = -1. if side == 'left' else 1.
                    force = torch.zeros((1, 1, 3), device=device)
                    force[0, 0, :2] = sign * args.push_n * torch.tensor([-math.sin(yaw), math.cos(yaw)], device=device)
                    env.robot.permanent_wrench_composer.set_forces_and_torques(forces=force, body_ids=push_body, is_global=True)
                    kicked = t
                if kicked is not None and t >= kicked + args.push_duration_s and pushing:
                    env.robot.permanent_wrench_composer.reset()
                    pushing = False
                if args.video:
                    center = env.robot.data.body_link_pos_w[0].mean(dim=0).cpu().numpy()
                    center[2] = max(.08, center[2])
                    env.sim.set_camera_view(eye=center + np.array([.5, .5, .35]), target=center)
                actions = policy(obs['policy'])
                status = policy.status()[0].tolist() if hasattr(policy, 'status') else None
                obs, _, term, trunc, _ = env.step(actions)
                pos, gravity, _ = env.last_transition
                up = float(-gravity[0, 2])
                x = float(pos[0, 0] - start[0])
                gy = float(gravity[0, 1])
                rows.append({'t': t, 'x': x, 'y': float(pos[0, 1] - start[1]), 'upright': up, 'gravity_y': gy, 'status': status})
                if kicked is not None:
                    if fell_t is None and up < .5 and t - kicked <= 2.:
                        fell_t = t
                    if fell_t is not None and upright_t is None:
                        streak = streak + 1 if up >= .8 else 0
                        if streak * dt >= .5 - 1e-9:
                            upright_t, x_at_upright = t, x
                    if upright_t is not None and success_t is None and x - x_at_upright >= .05:
                        success_t = t
                if writer is not None:
                    frame = env.terminal_frame if bool(term[0] or trunc[0]) else env.render()
                    if frame is not None:
                        from PIL import Image, ImageDraw
                        picture = Image.fromarray(np.asarray(frame, dtype=np.uint8))
                        draw = ImageDraw.Draw(picture)
                        draw.rectangle((0, 0, 960, 30), fill='black')
                        mode = '' if status is None else ('gait', 'settle', 'getup', 'confirm', 'abort', 'init')[int(status[0])]
                        draw.text((8, 8), f'{side} kick  t={t:.2f}s  x={x:+.3f}m  upright={up:+.2f}  mode={mode}', fill='white')
                        writer.append_data(np.asarray(picture))
                if bool(term[0] or trunc[0]):
                    ended = 'terminated' if bool(term[0]) else 'truncated'
                    break
                if success_t is not None:
                    break
        if writer is not None:
            writer.close()
            writer = None
        with (args.output / f'{index:02d}-{side}.jsonl').open('w') as trace:
            for row in rows:
                trace.write(json.dumps(row) + '\n')
        fell_side = None
        if fell_t is not None:
            gy_at_fall = [r['gravity_y'] for r in rows if r['t'] >= fell_t][:round(1. / dt)]
            fell_side = 'left' if np.mean(gy_at_fall) < 0 else 'right'
        result = {'index': index, 'seed': seed, 'requested_side': side, 'kick_s': kicked, 'fell_s': fell_t, 'fell_side': fell_side,
                  'stable_upright_s': upright_t, 'plus_5cm_s': success_t, 'ended': ended,
                  'success': success_t is not None and success_t - kicked <= args.window_s,
                  'x_at_kick': next((r['x'] for r in rows if kicked is not None and r['t'] >= kicked), None)}
        results.append(result)
        print(json.dumps(result), flush=True)
    by_side = {s: {'trials': sum(r['requested_side'] == s for r in results), 'fell': sum(r['requested_side'] == s and r['fell_s'] is not None for r in results),
                   'success': sum(r['requested_side'] == s and r['success'] for r in results)} for s in ('left', 'right')}
    report = {'schema_version': 1, 'bundle_manifest': manifest, 'physics': str(physics_path), 'physics_sha256': file_hash(physics_path),
              'stage': 'robust', 'seed': args.seed, 'push_s': args.push_s, 'push_n': args.push_n, 'push_duration_s': args.push_duration_s, 'push_body': args.push_body, 'window_s': args.window_s,
              'script_sha256': file_hash(__file__), 'environment_sha256': file_hash('src/le_snail/sim/env.py'),
              'by_side': by_side, 'results': results}
    (args.output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(by_side), flush=True)
except BaseException:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    if writer is not None:
        writer.close()
    if env is not None:
        env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
