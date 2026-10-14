import json
from pathlib import Path

import numpy as np
import pytest

from le_snail.contract import PolicyContract, VelocityEstimator, imu_base_vectors

CONFIG = Path(__file__).resolve().parents[1] / 'configs/policy.json'


@pytest.fixture
def contract():
    return PolicyContract.load(CONFIG)


def test_observation_at_home_has_gravity_and_no_motion(contract):
    obs = contract.observation(contract.home_rad, [0]*6, [0,0,-1], [0,0,0], contract.home_rad)
    assert obs.shape == (24,)
    np.testing.assert_array_equal(obs, [0]*14 + [-1] + [0]*9)


def test_position_commands_cannot_jump_even_for_extreme_policy_output(contract):
    target = contract.position_target([100, -100, 100, -100, 100, -100], contract.home_rad)
    np.testing.assert_allclose(target, [0.05, -0.65, 1.15, -0.55, 0.05, 0.30])
    for _ in range(500):
        target = contract.position_target([100]*6, target)
    assert np.all(target <= contract.upper_rad)
    assert target[2] == pytest.approx(1.60)
    assert target[5] == pytest.approx(0.95)


@pytest.mark.parametrize('action', [[float('nan')]*6, [float('inf')]*6, [0]*5])
def test_invalid_policy_output_rejected(contract, action):
    with pytest.raises(ValueError):
        contract.position_target(action, contract.home_rad)


def test_missed_deadline_does_not_allow_larger_motor_jump(contract):
    with pytest.raises(ValueError, match='interval'):
        contract.position_target([1]*6, contract.home_rad, dt=0.2)


def test_initial_target_outside_limits_rejected(contract):
    with pytest.raises(ValueError, match='outside'):
        contract.position_target([0]*6, [99]*6)


def test_interface_fingerprint_changes_with_joint_units(tmp_path, contract):
    values = json.loads(CONFIG.read_text())
    values['action_scale_rad'][0] = 0.5
    changed = PolicyContract(**values)
    assert changed.fingerprint != contract.fingerprint
    changed.save(tmp_path/'contract.json')
    assert PolicyContract.load(tmp_path/'contract.json').fingerprint == changed.fingerprint


def test_joint_order_cannot_silently_change():
    values = json.loads(CONFIG.read_text())
    values['joint_names'].reverse()
    with pytest.raises(ValueError, match='ordering'):
        PolicyContract(**values)


def test_imu_gravity_for_a_sideways_base():
    c = 2**-0.5
    gravity, gyro = imu_base_vectors([c,c,0,0], [1,2,3], np.eye(3))
    np.testing.assert_allclose(gravity, [0,-1,0], atol=1e-12)
    np.testing.assert_array_equal(gyro, [1,2,3])


def test_sensor_mount_rotation_is_applied_to_both_vectors():
    mount = [[0,-1,0],[1,0,0],[0,0,1]]
    gravity, gyro = imu_base_vectors([1,0,0,0], [1,0,0], mount)
    np.testing.assert_allclose(gravity, [0,0,-1])
    np.testing.assert_allclose(gyro, [0,1,0])


def test_reflection_is_not_a_valid_sensor_mount():
    with pytest.raises(ValueError, match='rotation'):
        imu_base_vectors([1,0,0,0], [0,0,0], np.diag([1,1,-1]))


def test_velocity_uses_timestamps_and_filter():
    estimator = VelocityEstimator(0.25)
    np.testing.assert_array_equal(estimator.update([0]*6, 5.0), [0]*6)
    np.testing.assert_allclose(estimator.update([0.1]*6, 5.1), [0.25]*6)
    np.testing.assert_allclose(estimator.update([0.1]*6, 5.2), [0.1875]*6)
    with pytest.raises(ValueError, match='increase'):
        estimator.update([0.1]*6, 5.2)
