"""Portable policy bundle integrity and interface checks."""
import hashlib
import json
from pathlib import Path

from .contract import PolicyContract


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    contract = PolicyContract.load(directory / 'policy.json')
    if manifest['contract_fingerprint'] != contract.fingerprint:
        raise ValueError('Policy contract does not match the exported model')
    if file_hash(directory / 'policy.pt') != manifest['policy_sha256']:
        raise ValueError('Policy file checksum mismatch')
    if file_hash(directory / 'sim.json') != manifest['physics_sha256']:
        raise ValueError('Physics snapshot checksum mismatch')
    if manifest['observation_size'] != 24 or manifest['action_size'] != 6:
        raise ValueError('Unsupported policy dimensions')
    return contract, manifest
