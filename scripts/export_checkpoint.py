#!/usr/bin/env python3
"""Export an exact saved feedforward checkpoint for review without more training.

Run in the training container. Requires a completed run's final bundle and real
export-parity observations so intermediate snapshots can use the same contract.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import torch
from rsl_rl.modules import ActorCritic
from le_snail.bundle import file_hash, load_bundle


class DeploymentActor(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.actor = copy.deepcopy(network.actor).cpu()
        self.normalizer = copy.deepcopy(network.actor_obs_normalizer).cpu()

    def forward(self, observations):
        return self.actor(self.normalizer(observations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be new')
    run = args.checkpoint.resolve().parent
    contract, manifest = load_bundle(run / 'exported')
    if file_hash(run / 'source.tar.gz') != manifest['source_sha256']:
        raise ValueError('Training source archive checksum mismatch')
    config = json.loads((run / 'run.json').read_text())['runner']
    policy_config = dict(config['policy'])
    if policy_config.pop('class_name') != 'ActorCritic':
        raise ValueError('Only the agreed feedforward ActorCritic is supported')
    observations = torch.load(run / 'export-parity.pt', map_location='cpu', weights_only=True)['observations']
    if observations.ndim != 2 or observations.shape[1] != 24 or not torch.isfinite(observations).all():
        raise ValueError('Expected finite recorded 24-element observations')
    network = ActorCritic({'policy': observations[:1]}, config['obs_groups'], num_actions=6, **policy_config)
    # These are this workspace's own RSL training artifacts, including optimizer metadata.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    args.output.mkdir(parents=True)
    exported = torch.jit.script(DeploymentActor(network).eval())
    exported.save(str(args.output / 'policy.pt'))
    reloaded = torch.jit.load(str(args.output / 'policy.pt'), map_location='cpu').eval()
    with torch.no_grad():
        expected = network.act_inference({'policy': observations})
        actual = reloaded(observations)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for name in ('policy.json', 'sim.json'):
        shutil.copy2(run / 'exported' / name, args.output / name)
    manifest.update(policy_sha256=file_hash(args.output / 'policy.pt'),
                    checkpoint_sha256=file_hash(args.checkpoint), checkpoint_name=args.checkpoint.name,
                    checkpoint_iteration=checkpoint['iter'])
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    load_bundle(args.output)
    parity = {'passed': True, 'samples': len(observations),
              'max_action_error': float((actual-expected).abs().max()), 'rtol': 1e-5, 'atol': 1e-5,
              'observations_sha256': file_hash(run / 'export-parity.pt'),
              'checkpoint_sha256': manifest['checkpoint_sha256']}
    (args.output / 'checkpoint-parity.json').write_text(json.dumps(parity, indent=2) + '\n')
    print(json.dumps(parity, indent=2), flush=True)


if __name__ == '__main__':
    main()
