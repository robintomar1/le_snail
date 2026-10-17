#!/usr/bin/env python3
"""Clone hybrid-001's crawl gait into an ordinary 24-input policy by DAgger (#15).

Teacher: PhaseGait (src/le_snail/gait_teacher.py), the hybrid gait locked to the applied target, so it can label
any state. Student: obs normalizer + MLP 24-128-128-6 (ELU), the RSL-RL actor's shape, exported as a normal bundle.
Rollouts run in sim from upright starts (robust physics unless --stage crawl). Round 0 is driven by the teacher;
in later rounds each episode is driven by the teacher with probability beta, otherwise by the current student.
Every visited step with upright cosine >= 0.5 is labelled with the teacher's action and kept (aggregated over
rounds); the student is refit on everything after each round. Writes only under --output.
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
p.add_argument('--teacher-config', type=Path, default=Path('configs/controllers/hybrid-001.json'))
p.add_argument('--contract', type=Path, default=Path('configs/policy-50hz-001.json'))
p.add_argument('--physics', type=Path, default=Path('configs/experiments/physics-v2-50hz-001.json'))
p.add_argument('--reference-bundle', type=Path, default=Path('runs/two-portion-50hz-001-100/exported'))
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--stage', choices=('robust', 'crawl'), default='robust')
p.add_argument('--num-envs', type=int, default=1024)
p.add_argument('--steps', type=int, default=750, help='policy steps per round (50 Hz)')
p.add_argument('--betas', type=float, nargs='+', default=[1., .5, .25, 0., 0., 0.])
p.add_argument('--updates', type=int, default=4000, help='gradient steps per refit')
p.add_argument('--batch', type=int, default=8192)
p.add_argument('--seed', type=int, default=7)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new')
args.output.mkdir(parents=True)
app = AppLauncher(args).app
env = None


def main():
    global env
    import torch
    from le_snail.bundle import file_hash, load_bundle
    from le_snail.contract import PolicyContract
    from le_snail.gait_teacher import load_phase_gait
    from le_snail.sim.env import SnailEnv, make_config

    class Student(torch.nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
            self.actor = torch.nn.Sequential(torch.nn.Linear(24, 128), torch.nn.ELU(), torch.nn.Linear(128, 128), torch.nn.ELU(),
                                             torch.nn.Linear(128, 6))

        def forward(self, obs):
            return self.actor((obs - self.mean) / self.std)

    torch.manual_seed(args.seed)
    device = args.device or 'cuda:0'
    contract = PolicyContract.load(args.contract)
    cfg = make_config(args.asset, args.contract, args.physics, num_envs=args.num_envs, stage=args.stage, device=device,
                      evaluation=True, initial_side='upright', seed=args.seed)
    env = SnailEnv(cfg)
    teacher = load_phase_gait(args.teacher_config, args.contract).to(device)
    n = args.num_envs
    observations, labels, report = [], [], {'args': {k: str(v) for k, v in vars(args).items()}, 'rounds': []}
    student = None
    obs, _ = env.reset(seed=args.seed)
    teacher.reset()
    for r, beta in enumerate(args.betas):
        started = time.monotonic()
        driver_teacher = torch.rand(n, device=device) < beta if student is not None else torch.ones(n, dtype=torch.bool, device=device)
        x_prev = env.robot.data.root_pos_w[:, 0].clone()
        forward = {'teacher': 0., 'student': 0.}
        env_steps = {'teacher': 0, 'student': 0}
        terminations = {'teacher': 0, 'student': 0}
        steps_student_upright = 0
        kept = 0
        for _ in range(args.steps):
            o = obs['policy']
            with torch.no_grad():
                label = teacher(o)
                act = label if student is None else torch.where(driver_teacher[:, None], label, student(o))
            upright = -o[:, 14] >= .5
            observations.append(o[upright].clone())
            labels.append(label[upright].clone())
            kept += int(upright.sum())
            steps_student_upright += int((~driver_teacher & upright).sum())
            obs, _, term, trunc, _ = env.step(act)
            dx = env.last_transition[0][:, 0] - x_prev   # before any auto-reset
            for name, rows in (('teacher', driver_teacher), ('student', ~driver_teacher)):
                forward[name] += float(dx[rows].sum())
                env_steps[name] += int(rows.sum())
                terminations[name] += int((term & rows).sum())
            ended = term | trunc
            if bool(ended.any()):
                ids = ended.nonzero().flatten()
                teacher.reset_envs(ids)
                if student is not None:
                    driver_teacher[ids] = torch.rand(len(ids), device=device) < beta
            x_prev = env.robot.data.root_pos_w[:, 0].clone()
        X, Y = torch.cat(observations), torch.cat(labels)
        if student is None:
            student = Student(X.mean(0), X.std(0).clamp_min(1e-3)).to(device)
            optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        # The normalizer (buffers) is fixed from round 0's data, so every refit sees the same input scaling.
        for g in optimizer.param_groups:
            g['lr'] = 1e-3
        losses = []
        for u in range(args.updates):
            if u == args.updates // 2:
                for g in optimizer.param_groups:
                    g['lr'] = 3e-4
            index = torch.randint(len(X), (args.batch,), device=device)
            loss = torch.nn.functional.mse_loss(student(X[index]), Y[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if u % 500 == 0 or u == args.updates - 1:
                losses.append(float(loss))
        per = lambda name: {'env_seconds': env_steps[name] * contract.dt,
                            'forward_cm_s': 100 * forward[name] / max(1e-9, env_steps[name] * contract.dt),
                            'terminations_per_env_minute': 60 * terminations[name] / max(1e-9, env_steps[name] * contract.dt)}
        stats = {'round': r, 'beta': beta, 'samples_kept': kept, 'dataset': len(X), 'teacher_driven': per('teacher'),
                 'student_driven': per('student'),
                 'student_driven_upright_fraction': steps_student_upright / max(1, env_steps['student']),
                 'fit_loss': losses, 'seconds': round(time.monotonic() - started, 1)}
        report['rounds'].append(stats)
        print(json.dumps(stats), flush=True)
        (args.output / 'dagger.json').write_text(json.dumps(report, indent=1) + '\n')

    # Export as a bundle with the same layout as a trained policy.
    bundle = args.output / 'exported'
    bundle.mkdir()
    deploy = Student(student.mean.cpu(), student.std.cpu())
    deploy.load_state_dict({k: v.cpu() for k, v in student.state_dict().items()})
    scripted = torch.jit.script(deploy.eval())
    scripted.save(str(bundle / 'policy.pt'))
    torch.save({'student_state_dict': deploy.state_dict()}, args.output / 'student.pt')
    contract.save(bundle / 'policy.json')
    (bundle / 'sim.json').write_bytes(args.physics.read_bytes())
    held_out = X[torch.randint(len(X), (4096,), device=device)].cpu()
    with torch.no_grad():
        reloaded = torch.jit.load(str(bundle / 'policy.pt'), map_location='cpu')(held_out)
        expected = deploy(held_out)
    torch.testing.assert_close(reloaded, expected, rtol=1e-5, atol=1e-5)
    _, reference = load_bundle(args.reference_bundle)
    manifest = dict(reference)
    manifest.update({'policy_sha256': file_hash(bundle / 'policy.pt'), 'physics_sha256': file_hash(bundle / 'sim.json'),
                     'stage': 'cloned',
                     'controller': {'type': 'dagger_clone', 'teacher': 'src/le_snail/gait_teacher.py PhaseGait',
                                    'teacher_config': str(args.teacher_config), 'teacher_config_sha256': file_hash(args.teacher_config),
                                    'teacher_module_sha256': file_hash('src/le_snail/gait_teacher.py'),
                                    'script_sha256': file_hash(__file__), 'stateful': False, 'training_performed': True},
                     'physics_source': str(args.physics), 'asset_validation_from': str(args.reference_bundle)})
    manifest['reference_training_source_sha256'] = manifest.pop('source_sha256', None)
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    load_bundle(bundle)
    print(f'Exported cloned gait bundle: {bundle}', flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(1)
finally:
    if env is not None:
        env.close()
    sys.stdout.flush()
    app.close()
