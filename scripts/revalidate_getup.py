#!/usr/bin/env python3
"""Revalidate that the scripted side get-ups work under the full simulated physics; zero training (#6).

Runs the confirmed keyframe plans (two variants per side) and the 64 +-0.1 rad perturbations through
the normal policy action path (clamp, joint limits, slew limit, delay queue, PD servos with effort and
velocity limits, gravity, ground contact, tether drag), from the settled side starts used in training:
  nominal  - the training physics (stage recover: no randomization)
  robust   - stage robust randomization (friction, mass, gains, effort/velocity limits, delay, drag)
  rated    - nominal, but every servo capped at the catalogue rated torque 0.98 N m
Each trial: 2 s settle with zero action, then the plan, then home held until --seconds. Success uses the
evaluation rule (cos >= 0.8 for 0.5 s); also records whether it stays upright to the end, peak torque
use (Isaac Lab's PD estimate, not measured solver torque), saturation, joint-limit margins, peak joint
speed and the lowest body height. Optional videos of env 0 for the confirmed batches. Writes only under
--output.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--contract', default='configs/policy-50hz-001.json')
p.add_argument('--physics', default='configs/experiments/two-portion-002.json')
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--confirmation', default='runs/feasibility-side-recovery-20261014T111801Z/results.json')
p.add_argument('--seconds', type=float, default=10.)
p.add_argument('--seed', type=int, default=4343)
p.add_argument('--video', action='store_true')
p.add_argument('--max-process-seconds', type=float, default=2400.)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new; evidence is never overwritten')
if args.video:
    args.enable_cameras = True
started = time.monotonic()
args.output.mkdir(parents=True)
app = AppLauncher(args).app
env = None

try:
    import numpy as np
    import torch
    from le_snail.bundle import file_hash
    from le_snail.evaluation import Trial
    from le_snail.sim.env import SnailEnv, make_config

    N = 64
    cfg = make_config(args.asset, args.contract, args.physics, num_envs=N, stage='recover',
                      evaluation=True, initial_side='left', device=args.device or 'cuda:0', seed=args.seed)
    if args.video:
        cfg.viewer.resolution = (960, 720)
    env = SnailEnv(cfg, render_mode='rgb_array' if args.video else None)
    device, dt = env.device, env.step_dt
    source = json.loads(Path(args.confirmation).read_text())
    perturb = np.random.default_rng(4242).uniform(-.1, .1, (64, 3, 6))   # the same perturbations as the earlier study
    ar = torch.arange(N, device=device)
    servo = env.robot.actuators['servos']
    report = {'schema_version': 1, 'training_performed': False, 'num_envs': N, 'policy_dt': dt, 'seconds': args.seconds,
              'settle_seconds': 2.0, 'seed': args.seed, 'physics_dt': env.physics['physics_dt'],
              'gravity_enabled': True, 'torque_note': 'applied_torque is the Isaac Lab implicit-actuator PD estimate clipped to the effort limit, not the PhysX solver torque',
              'provenance': {str(f): file_hash(f) for f in (args.asset, args.contract, args.physics, args.confirmation, __file__, 'src/le_snail/sim/env.py')},
              'batches': []}

    def confirmed_variants(side):
        return [{'left': (9, 34), 'right': (48, 62)}[side][i % 2] for i in range(N)]

    def run(side, condition, kind, variants):
        env.cfg.initial_side = side
        env.cfg.stage = 'robust' if condition == 'robust' else 'recover'
        env.reset(seed=args.seed)
        if condition == 'rated':
            rated = torch.full((N, 6), .980665, device=device)
            env.robot.write_joint_effort_limit_to_sim(rated, joint_ids=env.joint_ids)
            servo.effort_limit[:] = rated
        zero = torch.zeros((N, 6), device=device)
        for _ in range(round(2 / dt)):
            env.step(zero)
        start = env.robot.data.root_pos_w.clone()
        base = np.asarray(source['keys'][side])
        keys = env.tensor(np.array([np.vstack((base + perturb[v], env.contract.home_rad)) for v in variants]))
        key_i = torch.zeros(N, dtype=torch.long, device=device)
        held = torch.zeros_like(key_i)
        trials = [Trial(kind=side) for _ in range(N)]
        live = torch.ones(N, dtype=torch.bool, device=device)
        limit = env.robot.data.joint_effort_limits[:, env.joint_ids].clone()
        lower, upper = env.robot.data.joint_pos_limits[:, env.joint_ids, 0], env.robot.data.joint_pos_limits[:, env.joint_ids, 1]
        peak_ratio = torch.zeros(N, device=device)
        demand_ratio = torch.zeros(N, device=device)
        saturated_steps = torch.zeros(N, device=device)
        joint_margin = torch.full((N,), float('inf'), device=device)
        speed_ratio = torch.zeros(N, device=device)
        min_body_z = torch.full((N,), float('inf'), device=device)
        upright_trace = []
        writer = None
        if args.video and kind == 'confirmed':
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(args.output / f'{side}-{condition}-env0.mp4'), fps=round(1 / dt))
        for step in range(1, round(args.seconds / dt) + 1):
            assert time.monotonic() - started < args.max_process_seconds, 'Process budget exhausted'
            goal = keys[ar, key_i.clamp(max=3)]
            actions = ((goal - env.home) / env.scale).clamp(-1, 1)
            if writer is not None:
                center = (env.robot.data.body_link_pos_w[0].mean(dim=0)).cpu().numpy()
                center[2] = max(.08, center[2])
                env.sim.set_camera_view(eye=center + np.array([.5, .5, .35]), target=center)
            _, _, terminated, truncated, _ = env.step(actions)
            clipped = torch.clamp(env.home + actions * env.scale, env.lower, env.upper)
            reached = (env.target - clipped).abs().amax(dim=1) < 1e-3
            running = key_i <= 3
            held = torch.where(reached & running, held + 1, 0)
            advance = running & (held >= torch.where(key_i == 3, round(1. / dt), round(.1 / dt)))
            key_i += advance.long()
            held = torch.where(advance, 0, held)
            data = env.robot.data
            applied = data.applied_torque[:, env.joint_ids].abs()
            computed = data.computed_torque[:, env.joint_ids].abs()
            q, qd = data.joint_pos[:, env.joint_ids], data.joint_vel[:, env.joint_ids]
            peak_ratio = torch.maximum(peak_ratio, (applied / limit).amax(dim=1))
            demand_ratio = torch.maximum(demand_ratio, (computed / limit).amax(dim=1))
            saturated_steps += (computed > limit * 1.0001).any(dim=1).float()
            joint_margin = torch.minimum(joint_margin, torch.minimum(q - lower, upper - q).amin(dim=1))
            speed_ratio = torch.maximum(speed_ratio, (qd.abs() / data.joint_vel_limits[:, env.joint_ids]).amax(dim=1))
            min_body_z = torch.minimum(min_body_z, (data.body_link_pos_w[:, :, 2] - env.scene.env_origins[:, None, 2]).amin(dim=1))
            pos, gravity, _ = env.last_transition
            up = -gravity[:, 2]
            ended = terminated | truncated
            disp, upc, endc, livec = (pos - start).cpu().numpy(), up.cpu().numpy(), ended.cpu().numpy(), live.cpu().numpy()
            for i, trial in enumerate(trials):
                if livec[i]:
                    trial.update(step * dt, float(disp[i, 0]), float(disp[i, 1]), float(upc[i]), fault=bool(endc[i]))
            live &= ~ended
            upright_trace.append(upc)
            if writer is not None:
                frame = env.render()
                if frame is not None:
                    from PIL import Image, ImageDraw
                    picture = Image.fromarray(np.asarray(frame, dtype=np.uint8))
                    draw = ImageDraw.Draw(picture)
                    draw.rectangle((0, 0, 960, 30), fill='black')
                    draw.text((8, 8), f'{side} {condition} variant {variants[0]}  t={step * dt:.2f}s  upright={upc[0]:+.2f}  key={int(key_i[0])}', fill='white')
                    writer.append_data(np.asarray(picture))
        if writer is not None:
            writer.close()
        upright_trace = np.asarray(upright_trace)
        stats = {name: value.cpu().numpy() for name, value in (('peak_torque_ratio', peak_ratio), ('peak_demand_ratio', demand_ratio),
                 ('saturated_steps', saturated_steps), ('joint_limit_margin_rad', joint_margin), ('peak_speed_ratio', speed_ratio),
                 ('min_body_z_m', min_body_z))}
        labels = []
        for i, trial in enumerate(trials):
            success_t = trial.recovery_time_s
            after = upright_trace[round(success_t / dt):, i] if success_t is not None else None
            labels.append({'env': i, 'variant': int(variants[i]), 'stable_upright_time_s': success_t, 'reason': trial.reason,
                           'stays_upright_after': None if after is None else bool((after >= .5).all()),
                           'final_upright': float(upright_trace[-1, i]), 'max_upright': float(upright_trace[:, i].max()),
                           **{name: float(values[i]) for name, values in stats.items()}})
        ok = [l for l in labels if l['stable_upright_time_s'] is not None]
        summary = {'side': side, 'condition': condition, 'kind': kind, 'trials': N,
                   'stable_upright': len(ok), 'stays_upright_after': sum(bool(l['stays_upright_after']) for l in ok),
                   'faults': sum(l['reason'] == 'invalid_state_or_fault' for l in labels),
                   'time_s_range': [min(l['stable_upright_time_s'] for l in ok), max(l['stable_upright_time_s'] for l in ok)] if ok else None,
                   'successes': {name: {'max': max(l[name] for l in ok), 'min': min(l[name] for l in ok)} for name in stats} if ok else None,
                   'per_variant': {str(v): f"{sum(1 for l in ok if l['variant'] == v)}/{sum(1 for l in labels if l['variant'] == v)}" for v in sorted(set(variants))} if kind == 'confirmed' else None,
                   'labels': labels}
        report['batches'].append(summary)
        (args.output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
        print(json.dumps({k: summary[k] for k in ('side', 'condition', 'kind', 'stable_upright', 'stays_upright_after', 'faults', 'time_s_range', 'per_variant')}
                         | {'success_extremes': summary['successes'], 'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)

    with torch.no_grad():
        for condition in ('nominal', 'robust', 'rated'):
            for side in ('left', 'right'):
                run(side, condition, 'confirmed', confirmed_variants(side))
                run(side, condition, 'perturbed', list(range(N)))
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print('Get-up revalidation complete', flush=True)
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
