#!/usr/bin/env python3
"""Write the handover-curriculum training config from the passed Phase 0 sweep (#6); no simulation."""
import argparse
import json
from pathlib import Path

import numpy as np
from le_snail.bundle import file_hash
from le_snail.physics import load_physics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--sweep', type=Path, default=Path('runs/slew-handover-sweep-001/results.json'))
p.add_argument('--base', type=Path, default=Path('configs/experiments/physics-v2-002.json'))
p.add_argument('--confirmation', type=Path, default=Path('runs/feasibility-side-recovery-20261014T111801Z/results.json'))
p.add_argument('--fraction', type=float, default=.75)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new')
sweep = json.loads(args.sweep.read_text())
if not sweep.get('complete') or not sweep['phase0_physics_grid']['passes']:
    raise SystemExit('Phase 0 physics grid did not pass; no handover config is written')
grid = sweep['summary']['1']
source = json.loads(args.confirmation.read_text())
perturb = np.random.default_rng(sweep['seed']).uniform(-.1, .1, (64, 3, 6))
scripts = {}
for side, variants in (('left', (9, 34)), ('right', (48, 62))):
    scripts[side] = []
    for variant in variants:
        points = sorted(g['prefix_steps'] for g in grid[side]['grid'].values()
                        if g['variant'] == variant and g['eligible'] and g['prefix_steps'] > 0)
        if not points:
            raise SystemExit(f'No eligible handover point for {side} p{variant}')
        keys = (np.asarray(source['keys'][side]) + perturb[variant]).tolist()
        scripts[side].append({'id': f'p{variant}', 'keyframes_rad': keys, 'prefix_steps': points})
config = json.loads(args.base.read_text())
config['handover'] = {
    'fraction_of_side_starts': args.fraction,
    'scripts': scripts,
    'provenance': {
        'decision': 'LAPTOP 2026-10-15T02:36:00Z [DECISION robin] C: handover curriculum, Phase 1 on the 1.0 contract',
        'base_config': str(args.base), 'base_config_sha256': file_hash(args.base),
        'sweep_report': str(args.sweep), 'sweep_report_sha256': file_hash(args.sweep),
        'confirmation': str(args.confirmation), 'confirmation_sha256': file_hash(args.confirmation),
        'keyframes': f"confirmation keys + seed-{sweep['seed']} +-0.1 rad perturbation of each confirmed variant",
        'eligibility': sweep['passive_gates'], 'passive_window_s': sweep['passive_window_s'],
        'prefix_steps': 'Eligible grid points with k>0 at slew 1.0; k counts teacher policy steps from the reset.',
    },
}
args.output.write_text(json.dumps(config, indent=1) + '\n')
load_physics(args.output)
print(json.dumps({side: {v['id']: v['prefix_steps'] for v in variants} for side, variants in scripts.items()}))
