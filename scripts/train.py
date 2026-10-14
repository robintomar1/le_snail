#!/usr/bin/env python3
"""Run one curriculum stage; resume explicitly for later stages."""
import argparse
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', default='assets/so101/snail.urdf')
parser.add_argument('--contract', default='configs/policy.json')
parser.add_argument('--physics', default='configs/sim.json')
parser.add_argument('--stage', choices=['crawl', 'recover', 'robust'], default='crawl')
parser.add_argument('--num-envs', type=int, default=256)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resume', type=Path)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.iterations < 1 or args.num_envs < 1:
    parser.error('Iterations and environment count must be positive')
if args.output.exists():
    parser.error('--output must be a new directory to preserve earlier experiments')
app = AppLauncher(args).app

import json
import shutil
import tarfile
import importlib.metadata
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit
from le_snail.bundle import file_hash
from le_snail.contract import PolicyContract
from le_snail.sim.env import SnailEnv, make_config

try:
    device = args.device or 'cuda:0'
    cfg = make_config(args.asset, args.contract, args.physics, num_envs=args.num_envs, stage=args.stage, device=device, seed=args.seed)
    contract = PolicyContract.load(args.contract)
    if args.resume:
        previous = PolicyContract.load(args.resume.parent / 'policy.json')
        if previous.fingerprint != contract.fingerprint:
            raise ValueError('Resume contract differs from checkpoint training contract')
    run = args.output.resolve()
    run.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    # This workspace may not be a Git checkout. Keep the exact source used by
    # the experiment without copying local credentials or hardware calibration.
    with tarfile.open(run / 'source.tar.gz', 'w:gz') as archive:
        for folder in ('src', 'scripts', 'docker'):
            for path in sorted((project / folder).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.add(path, arcname=str(path.relative_to(project)))
        archive.add(project / 'pyproject.toml', arcname='pyproject.toml')
    contract.save(run / 'policy.json')
    shutil.copy2(args.physics, run / 'sim.json')
    asset_manifest = Path(args.asset).with_suffix('.manifest.json')
    shutil.copy2(asset_manifest, run / 'asset.manifest.json')
    source = Path(args.asset).parent / 'source.json'
    shutil.copy2(source, run / 'asset.source.json')
    config = {
        'seed': args.seed, 'device': device, 'num_steps_per_env': 64,
        'save_interval': 100, 'logger': 'tensorboard', 'obs_groups': {'policy': ['policy'], 'critic': ['policy']},
        'policy': {'class_name': 'ActorCritic', 'init_noise_std': 0.5,
                   'actor_obs_normalization': True, 'critic_obs_normalization': True,
                   'actor_hidden_dims': [128, 128], 'critic_hidden_dims': [128, 128], 'activation': 'elu'},
        'algorithm': {'class_name': 'PPO', 'value_loss_coef': 1., 'use_clipped_value_loss': True,
                      'clip_param': 0.2, 'entropy_coef': 0.01, 'num_learning_epochs': 5,
                      'num_mini_batches': 4, 'learning_rate': 3e-4, 'schedule': 'adaptive',
                      'gamma': 0.995, 'lam': 0.95, 'desired_kl': 0.01, 'max_grad_norm': 1.},
    }
    (run / 'run.json').write_text(json.dumps({'stage': args.stage, 'seed': args.seed, 'iterations': args.iterations, 'num_envs': args.num_envs, 'runner': config,
        'source_sha256': file_hash(run / 'source.tar.gz'),
        'resume_checkpoint': str(args.resume.resolve()) if args.resume else None,
        'resume_checkpoint_sha256': file_hash(args.resume) if args.resume else None}, indent=2) + '\n')
    env = RslRlVecEnvWrapper(SnailEnv(cfg))
    runner = OnPolicyRunner(env, config, log_dir=str(run), device=device)
    if args.resume:
        runner.load(str(args.resume))
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    runner.save(str(run / 'model_final.pt'))
    bundle = run / 'exported'
    network = runner.alg.policy
    export_policy_as_jit(network, normalizer=network.actor_obs_normalizer, path=str(bundle), filename='policy.pt')
    contract.save(bundle / 'policy.json')
    shutil.copy2(run / 'sim.json', bundle / 'sim.json')
    physics = json.loads((run / 'sim.json').read_text())
    manifest = {
        'schema_version': 1, 'contract_fingerprint': contract.fingerprint,
        'policy_sha256': file_hash(bundle / 'policy.pt'), 'physics_sha256': file_hash(bundle / 'sim.json'),
        'observation_size': 24, 'action_size': 6, 'stage': args.stage,
        'isaac_lab': '2.3.2', 'isaac_sim': '5.1.0', 'rsl_rl': '3.0.1',
        'asset_source': json.loads((run / 'asset.source.json').read_text()),
        'asset_validation': json.loads((run / 'asset.manifest.json').read_text()),
        'source_sha256': file_hash(run / 'source.tar.gz'),
        'installed_versions': {name: importlib.metadata.version(name) for name in ('torch', 'rsl-rl-lib', 'numpy')},
        'physics_validated': bool(contract.parameters_measured and physics['parameters_measured'] and physics['contact_geometry_validated'] and physics['protected_points_validated'] and physics['protected_points']),
    }
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Compare the reloaded CPU deployment artifact with the training inference
    # path on real rollout observations, including its learned normalizer.
    from le_snail.bundle import load_bundle
    load_bundle(bundle)
    inference = runner.get_inference_policy()
    exported = torch.jit.load(str(bundle / 'policy.pt'), map_location='cpu').eval()
    observations = env.get_observations()
    samples, expected, actual = [], [], []
    with torch.inference_mode():
        for _ in range(8):
            action = inference(observations)
            reloaded = exported(observations['policy'].cpu())
            torch.testing.assert_close(reloaded, action.cpu(), rtol=1e-5, atol=1e-5)
            samples.append(observations['policy'].cpu().clone())
            expected.append(action.cpu().clone())
            actual.append(reloaded.clone())
            observations, _, _, _ = env.step(action)
    samples, expected, actual = map(torch.cat, (samples, expected, actual))
    torch.save({'observations': samples, 'training_actions': expected, 'exported_actions': actual}, run / 'export-parity.pt')
    (run / 'export-parity.json').write_text(json.dumps({
        'passed': True, 'samples': len(samples), 'rtol': 1e-5, 'atol': 1e-5,
        'max_action_error': float((expected - actual).abs().max()),
        'tolerance_basis': 'float32 CPU versus CUDA inference; absolute tolerance 0.001% of unit action scale',
    }, indent=2) + '\n')
    env.close()
    print(f'Training and export completed: {run}', flush=True)
except BaseException:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
