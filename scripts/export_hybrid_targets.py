#!/usr/bin/env python3
"""Joint-target sequences of the hybrid controller (#14) for replay on the suspended real arm (#12); no simulation.

Runs the exported controller with the environment's action path reproduced exactly: target = home + clip(a) * scale,
clamped to the joint limits, then slewed at the contract's max target speed (no action delay). The base attitude is
scripted, not simulated: 'crawl' is upright throughout; 'left'/'right' lie on that side (settled-start gravity) until
the get-up plan reaches its final home key, then upright. Gyro is zero, so SETTLE takes its 1 s minimum. Writes per
episode t_s, the commanded action, the post-slew target in URDF radians and the controller mode, as .npz and .csv.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from le_snail.bundle import load_bundle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle', type=Path, default=Path('runs/hybrid-001-frozen/exported'))
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds', type=float, default=30.)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new')
contract, manifest = load_bundle(args.bundle)
policy = torch.jit.load(str(args.bundle / 'policy.pt'))
home, scale = torch.tensor(contract.home_rad), torch.tensor(contract.action_scale_rad)
lower, upper = torch.tensor(contract.lower_rad), torch.tensor(contract.upper_rad)
slew = torch.tensor(contract.max_target_speed_rad_s) * contract.dt
gravity = {'upright': [0., 0., -1.], 'left': [0.0395, -0.9941, 0.1007], 'right': [0.0123, 0.9940, 0.1083]}   # settled starts
modes = ('gait', 'settle', 'getup', 'confirm', 'abort', 'init')
args.output.mkdir(parents=True)
summary = {'bundle': str(args.bundle), 'policy_sha256': manifest['policy_sha256'], 'contract': contract.fingerprint,
           'policy_hz': contract.policy_hz, 'joint_names': list(contract.joint_names), 'episodes': {}}
for episode in ('crawl', 'left', 'right'):
    policy.reset()
    target = home.clone()
    attitude = 'upright' if episode == 'crawl' else episode
    rows = []
    for step in range(round(args.seconds / contract.dt)):
        obs = torch.zeros((1, 24))
        obs[0, 12:15] = torch.tensor(gravity[attitude])
        obs[0, 18:24] = (target - home) / scale
        action = policy(obs)[0]
        status = policy.status()[0]
        desired = torch.clamp(home + torch.clamp(action, -1., 1.) * scale, lower, upper)
        target = target + torch.clamp(desired - target, -slew, slew)
        rows.append((step * contract.dt, *action.tolist(), *target.tolist(), int(status[0])))
        # The get-up is done once the confirm step starts: from then on the base is upright.
        if attitude != 'upright' and int(status[0]) == 3:
            attitude = 'upright'
    data = np.array(rows, dtype=np.float64)
    np.savez_compressed(args.output / f'{episode}.npz', t_s=data[:, 0], action=data[:, 1:7], target_rad=data[:, 7:13], mode=data[:, 13].astype(int))
    with (args.output / f'{episode}.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t_s'] + [f'action_{j}' for j in contract.joint_names] + [f'target_rad_{j}' for j in contract.joint_names] + ['mode'])
        for row in rows:
            writer.writerow([f'{row[0]:.3f}'] + [f'{v:.6f}' for v in row[1:13]] + [modes[row[13]]])
    mode_changes = [(round(rows[i][0], 2), modes[rows[i][13]]) for i in range(len(rows)) if i == 0 or rows[i][13] != rows[i - 1][13]]
    summary['episodes'][episode] = {'steps': len(rows), 'mode_changes': mode_changes,
                                    'target_range_rad': [data[:, 7:13].min(0).round(3).tolist(), data[:, 7:13].max(0).round(3).tolist()]}
(args.output / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps(summary['episodes'], indent=1))
