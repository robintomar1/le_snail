#!/usr/bin/env python3
"""Export the hybrid scripted controller (#14) as a policy bundle; no simulation.

The bundle has the same layout as a trained policy (policy.pt TorchScript, policy.json contract, sim.json
physics snapshot, manifest.json), so scripts/evaluate.py and the deploy runtime load it unchanged. The
controller is stateful: callers reset() it at the start of every episode (evaluate.py does).
With --phase-gait, exports only the crawl gait, locked to the applied target instead of a clock
(src/le_snail/gait_teacher.py; the teacher for cloning the gait into a policy, #15). It has no fall handling.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

from le_snail.bundle import file_hash, load_bundle
from le_snail.contract import PolicyContract
from le_snail.gait_teacher import load_phase_gait
from le_snail.hybrid import load_controller

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, default=Path('configs/controllers/hybrid-001.json'))
p.add_argument('--physics', type=Path, default=Path('configs/experiments/physics-v2-50hz-001.json'))
p.add_argument('--reference-bundle', type=Path, default=Path('runs/two-portion-50hz-001-100/exported'),
               help='trained bundle on the same contract and asset; its asset validation is copied')
p.add_argument('--phase-gait', action='store_true', help='export the phase-locked crawl gait only')
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new')
config = json.loads(args.config.read_text())
contract_path = Path(config['contract'])
if file_hash(contract_path) != config['contract_sha256']:
    raise SystemExit('Contract changed since the controller config was written')
contract = PolicyContract.load(contract_path)
_, reference = load_bundle(args.reference_bundle)
if reference['contract_fingerprint'] != contract.fingerprint:
    raise SystemExit('Reference bundle uses a different contract')

load = load_phase_gait if args.phase_gait else load_controller
controller = torch.jit.script(load(args.config, contract_path).eval())
args.output.mkdir(parents=True)
controller.save(str(args.output / 'policy.pt'))
shutil.copyfile(contract_path, args.output / 'policy.json')
shutil.copyfile(args.physics, args.output / 'sim.json')

# Self-check: the saved module loads, resets and returns finite, bounded actions.
loaded = torch.jit.load(str(args.output / 'policy.pt'))
loaded.reset()
obs = torch.zeros((2, 24))
obs[:, 14] = -1.          # upright gravity
obs[1, 13], obs[1, 14] = -1., 0.   # lying on the left
for _ in range(10):
    action = loaded(obs)
assert action.shape == (2, 6) and torch.isfinite(action).all() and action.abs().max() <= 1.

module = 'src/le_snail/gait_teacher.py' if args.phase_gait else 'src/le_snail/hybrid.py'
manifest = dict(reference)
manifest.update({
    'policy_sha256': file_hash(args.output / 'policy.pt'),
    'physics_sha256': file_hash(args.output / 'sim.json'),
    'stage': 'scripted',
    'controller': {'type': 'phase_gait' if args.phase_gait else 'hybrid_scripted', 'config': str(args.config), 'config_sha256': file_hash(args.config),
                   'module': module, 'module_sha256': file_hash(module),
                   'stateful': True, 'reset_required_each_episode': True, 'training_performed': False},
    'physics_source': str(args.physics),
    'asset_validation_from': str(args.reference_bundle),
})
# The reference's training source archive hash does not describe this controller.
manifest['reference_training_source_sha256'] = manifest.pop('source_sha256', None)
(args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
load_bundle(args.output)
print(f'Exported hybrid controller bundle: {args.output}')
