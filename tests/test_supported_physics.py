import copy
import json
from pathlib import Path

import pytest
from le_snail.physics import load_physics


@pytest.mark.parametrize('change', ['missing_side', 'nonunit_quaternion', 'nonfinite_joint',
                                  'missing_identity', 'reversed_gate', 'out_of_range_gate'])
def test_invalid_experimental_reset_or_gate_is_rejected(tmp_path, change):
    source = Path(__file__).resolve().parents[1] / 'configs/experiments/crawl-supported-001.json'
    config = copy.deepcopy(json.loads(source.read_text()))
    if change == 'missing_side':
        del config['settled_starts']['left']
    elif change == 'nonunit_quaternion':
        config['settled_starts']['upright']['root_pose'][3:] = [0., 0., 0., 0.]
    elif change == 'nonfinite_joint':
        config['settled_starts']['right']['joint_rad'][0] = float('nan')
    elif change == 'missing_identity':
        del config['settled_start_asset_sha256']
    elif change == 'reversed_gate':
        config['positive_progress_upright_range'] = [.8, .5]
    else:
        config['positive_progress_upright_range'] = [.5, 1.1]
    path = tmp_path / 'physics.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_physics(path)
