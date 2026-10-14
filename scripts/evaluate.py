#!/usr/bin/env python3
"""Evaluate five crawl, five left recovery and five right recovery trials."""
import argparse
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', type=Path, required=True)
parser.add_argument('--asset', default='assets/so101/snail.urdf')
parser.add_argument('--physics-override', type=Path, help='Explicit diagnostic domain-shift evaluation; preserves original bundle and cannot qualify a milestone')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--video', action='store_true')
parser.add_argument('--camera', choices=('follow', 'wide'), default='follow', help='Close base-following view or original fixed wide view')
parser.add_argument('--probe', action='store_true', help='One diagnostic trial per start; cannot qualify a milestone')
parser.add_argument('--probe-seconds', type=float, default=5., help='Diagnostic duration, bounded by each original trial deadline (default: 5)')
parser.add_argument('--nominal-probe', action='store_true', help='Use nominal training physics for diagnostic comparison; requires --probe')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error('--output must be new')
if not 0 < args.probe_seconds <= 120:
    parser.error('--probe-seconds must be in (0, 120]')
if args.nominal_probe and not args.probe:
    parser.error('--nominal-probe requires --probe and cannot run qualification')
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

from dataclasses import asdict
import json
import math
import numpy as np
import torch
from le_snail.bundle import file_hash, load_bundle
from le_snail.evaluation import Trial, summarize
from le_snail.sim.env import SnailEnv, make_config

try:
    contract, manifest = load_bundle(args.bundle)
    asset_validation = json.loads(Path(args.asset).with_suffix('.manifest.json').read_text())
    for key in ('source_sha256', 'mesh_sha256'):
        if asset_validation[key] != manifest['asset_validation'][key]:
            raise ValueError('Evaluation asset differs from the training asset')
    args.output.mkdir(parents=True)
    device = args.device or 'cuda:0'
    physics_stage = 'crawl' if args.nominal_probe else 'robust'
    physics_path = args.physics_override or args.bundle / 'sim.json'
    # Capture provenance before stepping; collaborators may edit shared files
    # while a long rendered evaluation is using its already-imported code.
    runtime_hashes = {name: file_hash(name) for name in (__file__, 'src/le_snail/sim/env.py', 'src/le_snail/physics.py', 'src/le_snail/speed.py', 'src/le_snail/action_stats.py')}
    physics_override_configuration = json.loads(args.physics_override.read_text()) if args.physics_override else None
    runtime_physics_hash = file_hash(physics_path)
    cfg = make_config(args.asset, args.bundle / 'policy.json', physics_path, num_envs=1, stage=physics_stage, device=device, evaluation=True, seed=args.seed)
    cfg.viewer.resolution = (960, 720)
    env = SnailEnv(cfg, render_mode='rgb_array' if args.video else None)
    policy = torch.jit.load(str(args.bundle / 'policy.pt'), map_location=device).eval()
    trials = []

    def base_yaw(quaternion):
        w, x, y, z = (float(v) for v in quaternion)
        return math.atan2(2*(w*z + x*y), 1-2*(y*y + z*z))

    for kind in ('crawl', 'left', 'right'):
        env.cfg.initial_side = 'upright' if kind == 'crawl' else kind
        for index in range(1 if args.probe else 5):
            seed = args.seed + len(trials)
            obs, _ = env.reset(seed=seed)
            # Let gravity settle the start pose; these seconds are not scored as locomotion.
            settling_fault = False
            for _ in range(round(2 / contract.dt)):
                obs, _, term, trunc, _ = env.step(torch.zeros((1, 6), device=device))
                if bool(term[0] or trunc[0]):
                    settling_fault = True
                    break
            start = env.robot.data.root_pos_w[0].clone()
            env.start_pos[0] = start
            env.episode_length_buf[:] = 0
            # Settling is excluded from both the trial clock and the maximum
            # fallen duration; otherwise side trials can stop at ~28 seconds.
            env.fallen_time.zero_()
            previous_yaw = base_yaw(env.robot.data.root_quat_w[0])
            yaw_from_start = 0.
            trial = Trial(kind=kind)
            trace_path = args.output / f'{kind}-{index}.jsonl'
            writer = None
            if args.video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(args.output / f'{kind}-{index}.mp4'), fps=round(contract.policy_hz))
            # Simulator caches survive this scope and are mutated by the next
            # reset; inference tensors would reject that later in-place update.
            with trace_path.open('w') as trace, torch.no_grad():
                upright = float(-env.robot.data.projected_gravity_b[0, 2])
                trial.update(0., 0., 0., upright, fault=settling_fault)
                duration = min(args.probe_seconds, trial.timeout_s) if args.probe else trial.timeout_s
                for step in range(1, round(duration / contract.dt) + 1):
                    if trial.finished:
                        break
                    if args.video and args.camera == 'follow':
                        # Center the whole arm, not just its base: the gripper
                        # extends far enough to leave a base-centered close shot.
                        center = env.robot.data.body_link_pos_w[0].mean(dim=0).cpu().numpy()
                        center[2] = max(.08, center[2])
                        env.sim.set_camera_view(eye=center + np.array([.5, .5, .35]), target=center)
                    actions = policy(obs['policy'])
                    obs, _, term, trunc, _ = env.step(actions)
                    pos, gravity, contact = env.last_transition
                    displacement = (pos[0] - start).cpu().numpy()
                    upright = float(-gravity[0, 2])
                    yaw = base_yaw(env.last_transition_quat[0])
                    yaw_from_start += (yaw - previous_yaw + math.pi) % (2*math.pi) - math.pi
                    previous_yaw = yaw
                    elapsed = step * contract.dt
                    trial.update(elapsed, float(displacement[0]), float(displacement[1]), upright, fault=bool(term[0]))
                    trace.write(json.dumps({'time_s': elapsed, 'forward_m': float(displacement[0]), 'lateral_m': float(displacement[1]), 'upright_cos': upright,
                                            'base_yaw_rad': yaw, 'yaw_from_start_rad': yaw_from_start,
                                            'terminated': bool(term[0]), 'truncated': bool(trunc[0])}) + '\n')
                    if writer is not None:
                        frame = env.terminal_frame if bool(term[0] or trunc[0]) else env.render()
                        if frame is not None:
                            # A moving camera hides displacement; retain scored
                            # world-frame progress visibly alongside the close view.
                            from PIL import Image, ImageDraw
                            picture = Image.fromarray(np.asarray(frame, dtype=np.uint8))
                            draw = ImageDraw.Draw(picture)
                            draw.rectangle((0, 0, 800, 30), fill='black')
                            draw.text((8, 8), f'{kind}  t={elapsed:.2f}s  forward={displacement[0]:+.3f}m  lateral={displacement[1]:+.3f}m  upright={upright:+.2f}  yaw={math.degrees(yaw_from_start):+.1f}deg', fill='white')
                            writer.append_data(np.asarray(picture))
                    if bool(trunc[0]) and not trial.finished:
                        trial.finished, trial.reason = True, 'simulation_timeout'
                if not trial.finished:
                    trial.finished, trial.reason = True, 'probe_limit' if args.probe else 'timeout'
            if writer is not None:
                writer.close()
                writer = None
            trials.append(trial)
            print(kind, index, trial.reason, flush=True)
    report = summarize(trials)
    report['speed_diagnostics'] = env.speed_diagnostics()
    report['action_diagnostics'] = env.action_diagnostics()
    report['runtime_source_hashes'] = runtime_hashes
    report.update({'seed': args.seed, 'camera': args.camera, 'probe_only': args.probe,
                   'physics_stage': physics_stage,
                   'probe_seconds': args.probe_seconds if args.probe else None,
                   'bundle_manifest': manifest, 'trials': [asdict(t) for t in trials], 'physics_validated': manifest['physics_validated']})
    report['physics_override'] = ({'path': str(args.physics_override), 'sha256': runtime_physics_hash,
                                   'original_bundle_physics_sha256': file_hash(args.bundle / 'sim.json'),
                                   'configuration': physics_override_configuration,
                                   'diagnostic_only': True,
                                   'evaluation_script_sha256': runtime_hashes[__file__],
                                   'environment_sha256': runtime_hashes['src/le_snail/sim/env.py'],
                                   'physics_validation_sha256': runtime_hashes['src/le_snail/physics.py'],
                                   'speed_diagnostics_sha256': runtime_hashes['src/le_snail/speed.py']} if args.physics_override else None)
    if args.physics_override:
        report['physics_validated'] = False
    report['milestone_candidate'] = not args.physics_override and not args.probe and report['passed'] and manifest['physics_validated'] and manifest['stage'] == 'robust'
    report['milestone_passed'] = False  # Video/contact review must also confirm an acceptable gait.
    report['visual_gait_review_required'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
except BaseException:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    # Release render products even when a trial fails before normal completion.
    if 'writer' in globals() and writer is not None:
        writer.close()
    if 'env' in globals():
        env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
