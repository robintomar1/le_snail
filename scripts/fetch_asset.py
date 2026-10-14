#!/usr/bin/env python3
"""Fetch once, record the immutable revision, and prepare the SO-101 URDF."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from le_snail.assets import prepare_urdf
from le_snail.contract import PolicyContract

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--revision', help='Commit/tag to fetch; main is resolved once if omitted')
parser.add_argument('--source', type=Path, help='Existing SO-ARM100 checkout, for offline preparation')
parser.add_argument('--output', type=Path, default=Path('assets/so101'))
parser.add_argument('--contract', default='configs/policy.json')
args = parser.parse_args()
contract = PolicyContract.load(args.contract)
output = args.output.resolve()
lock = output / 'source.json'
if lock.exists():
    record = json.loads(lock.read_text())
    if args.revision and args.revision != record['commit']:
        parser.error('Existing asset revision differs. Use a new --output directory.')
else:
    if output.exists() and any(output.iterdir()):
        parser.error('Output is nonempty without a source manifest; choose a fresh directory.')
    with tempfile.TemporaryDirectory(prefix='le-snail-asset-') as temp:
        if args.source:
            checkout = args.source.resolve()
        else:
            checkout = Path(temp) / 'upstream'
            subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', 'https://github.com/TheRobotStudio/SO-ARM100.git', str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), 'checkout', args.revision or 'main'], check=True)
        revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--', 'Simulation/SO101'], text=True)
        if dirty:
            parser.error('Source SO101 assets have uncommitted changes; use a clean revision.')
        shutil.copytree(checkout / 'Simulation/SO101', output / 'upstream')
        for license_file in checkout.glob('LICENSE*'):
            if license_file.is_file():
                shutil.copy2(license_file, output / license_file.name)
        record = {'repository': 'https://github.com/TheRobotStudio/SO-ARM100', 'commit': revision}
        lock.write_text(json.dumps(record, indent=2) + '\n')
manifest = prepare_urdf(output / 'upstream/so101_new_calib.urdf', output / 'snail.urdf', contract)
print(json.dumps({'source': record, 'asset': str(output / 'snail.urdf'), 'validation': manifest}, indent=2))
