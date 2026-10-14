import copy
import json
from pathlib import Path

import pytest
from le_snail.physics import load_physics


@pytest.mark.parametrize('change', ['missing_side', 'nonunit_quaternion', 'nonfinite_joint',
                                  'missing_identity', 'reversed_gate', 'out_of_range_gate',
                                  'negative_yaw_cost', 'nonfinite_yaw_cost',
                                  'negative_recovery_cap', 'nonfinite_recovery_cap', 'excessive_recovery_cap',
                                  'recovery_without_settled_starts', 'recovery_from_upright_side'])
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
    elif change == 'out_of_range_gate':
        config['positive_progress_upright_range'] = [.5, 1.1]
    elif change == 'negative_yaw_cost':
        config['yaw_cost_weight'] = -1.
    elif change == 'nonfinite_yaw_cost':
        config['yaw_cost_weight'] = float('nan')
    elif change == 'negative_recovery_cap':
        config['side_upright_progress_reward_cap'] = -1.
    elif change == 'nonfinite_recovery_cap':
        config['side_upright_progress_reward_cap'] = float('nan')
    elif change == 'excessive_recovery_cap':
        config['side_upright_progress_reward_cap'] = 1.01
    elif change == 'recovery_without_settled_starts':
        config['side_upright_progress_reward_cap'] = 1.
        del config['settled_starts']
    elif change == 'recovery_from_upright_side':
        config['side_upright_progress_reward_cap'] = 1.
        config['settled_starts']['left']['root_pose'][3:] = [1., 0., 0., 0.]
    path = tmp_path / 'physics.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_physics(path)


@pytest.mark.parametrize('change', ['negative_payload', 'nonfinite_center', 'bad_box', 'missing_measured', 'bad_effort', 'bad_velocity'])
def test_invalid_payload_or_limit_interval(tmp_path, change):
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/experiments/physics-v2-bootstrap-001.json').read_text())
    if change == 'negative_payload':
        config['base_payload']['mass_kg'] = -1.
    elif change == 'nonfinite_center':
        config['base_payload']['center_m'][0] = float('nan')
    elif change == 'bad_box':
        config['base_payload']['box_size_m'][0] = 0.
    elif change == 'missing_measured':
        del config['base_payload']['measured']
    elif change == 'bad_effort':
        config['effort_limit_range_nm'] = [1.6, 2.9]
    else:
        config['velocity_limit_range_rad_s'] = [5., 3.]
    path = tmp_path / 'physics.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_physics(path)


def test_payload_combination_preserves_origin_inertia_and_symmetry():
    import numpy as np
    from le_snail.physics import combine_box_payload
    mass, com, inertia = combine_box_payload(2., [0., 0., 0.], np.diag([1., 2., 3.]),
                                           {'mass_kg': 1., 'center_m': [3., 0., 0.], 'box_size_m': [2., 2., 2.]})
    assert mass == 3.
    np.testing.assert_allclose(com, [1., 0., 0.])
    np.testing.assert_allclose(inertia, np.diag([1.+2/3, 8.+2/3, 9.+2/3]))
    np.testing.assert_allclose(inertia + mass*(np.dot(com, com)*np.eye(3)-np.outer(com, com)),
                               np.diag([1.+2/3, 11.+2/3, 12.+2/3]))
