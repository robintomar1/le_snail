import json
import numpy as np
import pytest
from le_snail.contract import JOINT_NAMES
from le_snail.hardware import ImuDecoder, JointMapping


def test_gripper_percent_mapping_is_distinct_from_body_degrees():
    mapping = JointMapping({'radians_per_native_unit': [np.pi/180]*5 + [.01], 'offset_rad': [0]*5 + [.2], 'native_lower': [-180]*5 + [0], 'native_upper': [180]*5+[100]})
    native = {f'{name}.pos': 90. for name in JOINT_NAMES}
    q = mapping.to_radians(native)
    np.testing.assert_allclose(q, [np.pi/2]*5+[1.1])
    assert mapping.to_native(q) == pytest.approx(native)
    with pytest.raises(ValueError, match='outside'):
        mapping.to_native([0]*5+[2.])


def test_unfilled_mapping_cannot_enable_motion():
    with pytest.raises(ValueError):
        JointMapping({'radians_per_native_unit': [None]*6, 'offset_rad': [0]*6, 'native_lower': [-1]*6, 'native_upper': [1]*6})


def packet(seq, timestamp):
    return json.dumps({'seq': seq, 'sample_us': timestamp, 'q_wxyz': [1,0,0,0], 'gyro_rad_s': [0,0,0]})


def test_imu_queued_packets_retain_old_sample_time():
    decoder = ImuDecoder(np.eye(3))
    decoder.decode(packet(1, 1_000_000), 10.)
    delayed = decoder.decode(packet(2, 1_010_000), 11.)
    assert delayed['sample_at'] == pytest.approx(10.01)
    assert delayed['received_at'] - delayed['sample_at'] == pytest.approx(.99)


def test_imu_replay_and_reboot_are_rejected():
    decoder = ImuDecoder(np.eye(3))
    decoder.decode(packet(100, 1_000_000), 10.)
    with pytest.raises(ValueError, match='replay'):
        decoder.decode(packet(0, 1_010_000), 10.01)


def test_nonfinite_feedback_timeout_cannot_disable_freshness_check():
    from pathlib import Path
    from le_snail.contract import PolicyContract
    from le_snail.hardware import validate_hardware_settings
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/hardware.example.json').read_text())
    contract = PolicyContract.load(root/'configs/policy.json')
    config.update(calibration_verified=True, telemetry_register_units_verified=True,
                  contract_fingerprint=contract.fingerprint, max_feedback_age_s=float('nan'))
    with pytest.raises(ValueError, match='max_feedback_age_s'):
        validate_hardware_settings(config, contract)


def test_string_false_does_not_count_as_verified_hardware():
    from pathlib import Path
    from le_snail.contract import PolicyContract
    from le_snail.hardware import validate_hardware_settings
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/hardware.example.json').read_text())
    config['calibration_verified'] = 'false'
    with pytest.raises(ValueError, match='verified'):
        validate_hardware_settings(config, PolicyContract.load(root/'configs/policy.json'))
