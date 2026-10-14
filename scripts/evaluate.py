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
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--video', action='store_true')
parser.add_argument('--probe', action='store_true', help='One five-second trial per start for pipeline diagnostics; cannot qualify a milestone')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error('--output must be new')
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

from dataclasses import asdict
import json
import numpy as np
import torch
from le_snail.bundle import load_bundle
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
    cfg = make_config(args.asset, args.bundle / 'policy.json', args.bundle / 'sim.json', num_envs=1, stage='robust', device=device, evaluation=True, seed=args.seed)
    env = SnailEnv(cfg, render_mode='rgb_array' if args.video else None)
    policy = torch.jit.load(str(args.bundle / 'policy.pt'), map_location=device).eval()
    trials = []
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
                duration = min(5., trial.timeout_s) if args.probe else trial.timeout_s
                for step in range(1, round(duration / contract.dt) + 1):
                    if trial.finished:
                        break
                    actions = policy(obs['policy'])
                    obs, _, term, trunc, _ = env.step(actions)
                    pos, gravity, contact = env.last_transition
                    displacement = (pos[0] - start).cpu().numpy()
                    upright = float(-gravity[0, 2])
                    elapsed = step * contract.dt
                    trial.update(elapsed, float(displacement[0]), float(displacement[1]), upright, fault=bool(term[0]))
                    trace.write(json.dumps({'time_s': elapsed, 'forward_m': float(displacement[0]), 'lateral_m': float(displacement[1]), 'upright_cos': upright, 'terminated': bool(term[0]), 'truncated': bool(trunc[0])}) + '\n')
                    if writer is not None:
                        frame = env.terminal_frame if bool(term[0] or trunc[0]) else env.render()
                        if frame is not None:
                            writer.append_data(np.asarray(frame, dtype=np.uint8))
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
    report.update({'seed': args.seed, 'probe_only': args.probe, 'bundle_manifest': manifest, 'trials': [asdict(t) for t in trials], 'physics_validated': manifest['physics_validated']})
    report['milestone_candidate'] = not args.probe and report['passed'] and manifest['physics_validated'] and manifest['stage'] == 'robust'
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
