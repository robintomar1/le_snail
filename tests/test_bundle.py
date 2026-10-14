import json
from pathlib import Path
import pytest
from le_snail.bundle import file_hash, load_bundle
from le_snail.contract import PolicyContract


def bundle(path):
    contract = PolicyContract.load(Path(__file__).resolve().parents[1]/'configs/policy.json')
    contract.save(path/'policy.json')
    (path/'policy.pt').write_bytes(b'test policy artifact')
    (path/'sim.json').write_text('{}')
    (path/'manifest.json').write_text(json.dumps({'contract_fingerprint': contract.fingerprint, 'policy_sha256': file_hash(path/'policy.pt'), 'physics_sha256': file_hash(path/'sim.json'), 'observation_size': 24, 'action_size': 6}))


def test_changed_weights_are_rejected_before_loading(tmp_path):
    bundle(tmp_path)
    load_bundle(tmp_path)
    (tmp_path/'policy.pt').write_bytes(b'different weights')
    with pytest.raises(ValueError, match='checksum'):
        load_bundle(tmp_path)


def test_changed_joint_mapping_contract_is_rejected(tmp_path):
    bundle(tmp_path)
    values = json.loads((tmp_path/'policy.json').read_text())
    values['home_rad'][0] = 0.1
    (tmp_path/'policy.json').write_text(json.dumps(values))
    with pytest.raises(ValueError, match='contract'):
        load_bundle(tmp_path)
