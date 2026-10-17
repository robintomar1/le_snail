import json
import struct
import time

import numpy as np
import pytest
from le_snail.contract import JOINT_NAMES
from le_snail.hardware import ImuDecoder, JointMapping, XbusReader, parse_mtdata2, xbus_frame


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


def mtdata2(counter, ticks, q_wxyz=(1., 0., 0., 0.), gyro=(0., 0., 0.), quaternion_id=0x2010, extra=b''):
    items = [(0x1020, struct.pack('>H', counter)), (0x1060, struct.pack('>I', ticks)),
             (0x8020, struct.pack('>3f', *gyro))]
    if quaternion_id is not None:
        items.append((quaternion_id, struct.pack('>4f', *q_wxyz)))
    return b''.join(struct.pack('>HB', i, len(d)) + d for i, d in items) + extra


def test_xbus_frames_match_published_command_bytes():
    # GoToConfig / GoToMeasurement as written in Xsens' low-level protocol documentation.
    assert xbus_frame(0x30) == bytes.fromhex('FAFF3000D1')
    assert xbus_frame(0x10) == bytes.fromhex('FAFF1000F1')
    assert XbusReader().feed(bytes.fromhex('FAFF3000D1FAFF1000F1')) == [(0x30, b''), (0x10, b'')]


def test_xbus_reader_resynchronizes_across_split_and_corrupt_input():
    good = xbus_frame(0x36, mtdata2(1, 10))
    corrupt = bytearray(good)
    corrupt[-2] ^= 0x01
    stream = b'\x00\x13garbage' + bytes(corrupt) + good
    reader = XbusReader()
    messages = []
    for i in range(0, len(stream), 5):
        messages += reader.feed(stream[i:i+5])
    assert messages == [(0x36, mtdata2(1, 10))]


def test_xbus_extended_length_round_trip():
    payload = bytes(range(256)) * 2
    frame = xbus_frame(0x36, payload)
    assert frame[3] == 0xFF and frame[4:6] == struct.pack('>H', 512)
    assert XbusReader().feed(frame) == [(0x36, payload)]


def test_persistent_xbus_corruption_fails_instead_of_resyncing_forever():
    reader = XbusReader(max_consecutive_errors=3)
    with pytest.raises(ValueError, match='Persistent'):
        reader.feed(b'\xfa\xff\x36\x00\x00' * 10)


def test_identity_orientation_gives_downward_base_gravity_and_10khz_time():
    decoder = ImuDecoder(np.eye(3))
    first = decoder.decode(mtdata2(7, 10_000, gyro=(.1, -.2, .3)), 5.)
    np.testing.assert_allclose(first['gravity'], [0, 0, -1], atol=1e-7)
    np.testing.assert_allclose(first['gyro'], [.1, -.2, .3], atol=1e-7)
    second = decoder.decode(mtdata2(8, 10_100), 6.)
    assert second['sample_at'] - first['sample_at'] == pytest.approx(.01)
    assert second['received_at'] - second['sample_at'] == pytest.approx(.99)


def test_sensor_rolled_onto_its_side_reports_sideways_gravity():
    s = np.sqrt(.5)
    # +90 degrees about sensor x: sensor y axis points up, so gravity appears along -y.
    sample = ImuDecoder(np.eye(3)).decode(mtdata2(1, 1, q_wxyz=(s, s, 0., 0.)), 0.)
    np.testing.assert_allclose(sample['gravity'], [0, -1, 0], atol=1e-6)


def test_packet_counter_wrap_keeps_sequence_monotonic():
    decoder = ImuDecoder(np.eye(3))
    a = decoder.decode(mtdata2(65535, 100), 1.)
    b = decoder.decode(mtdata2(0, 200), 1.01)
    assert b['seq'] == a['seq'] + 1


def test_imu_replay_and_reboot_are_rejected():
    decoder = ImuDecoder(np.eye(3))
    decoder.decode(mtdata2(100, 1_000_000), 10.)
    with pytest.raises(ValueError, match='replay'):
        decoder.decode(mtdata2(101, 10), 10.01)
    decoder = ImuDecoder(np.eye(3))
    decoder.decode(mtdata2(100, 1_000), 10.)
    with pytest.raises(ValueError, match='replay'):
        decoder.decode(mtdata2(100, 1_100), 10.01)


@pytest.mark.parametrize('quaternion_id, message', [
    (None, 'missing'), (0x2014, 'ENU'), (0x2011, 'float32')])
def test_unhandled_orientation_output_is_rejected(quaternion_id, message):
    # Raw acceleration is present but must never be used as a silent gravity fallback.
    payload = mtdata2(1, 1, quaternion_id=quaternion_id, extra=struct.pack('>HB3f', 0x4020, 12, 0, 0, 9.81))
    with pytest.raises(ValueError, match=message):
        ImuDecoder(np.eye(3)).decode(payload, 0.)


def test_truncated_mtdata2_is_rejected():
    with pytest.raises(ValueError, match='Corrupt'):
        parse_mtdata2(mtdata2(1, 1)[:-1])


def test_missing_imu_baudrate_cannot_enable_motion():
    from pathlib import Path
    from le_snail.contract import PolicyContract
    from le_snail.hardware import validate_hardware_settings
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'configs/hardware.example.json').read_text())
    contract = PolicyContract.load(root/'configs/policy.json')
    config.update(calibration_verified=True, telemetry_register_units_verified=True, contract_fingerprint=contract.fingerprint)
    with pytest.raises(ValueError, match='imu_baudrate'):
        validate_hardware_settings(config, contract)


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


def test_startup_overflow_reports_and_incomplete_packets_are_tolerated_only_during_warmup():
    from le_snail.hardware import XsensStream
    stream = XsensStream(np.eye(3), warmup_s=0.5)
    assert stream.handle(0x42, bytes((0x29,)), 10.0) is None
    assert stream.handle(0x36, mtdata2(1, 100, quaternion_id=None), 10.1) is None
    assert stream.handle(0x36, mtdata2(2, 200), 10.2)['seq'] == 2
    assert (stream.overflow_reports, stream.skipped_warmup_packets) == (1, 1)
    with pytest.raises(ValueError, match='missing'):
        stream.handle(0x36, mtdata2(3, 300, quaternion_id=None), 10.6)


def test_other_xsens_errors_always_stop_the_stream():
    from le_snail.hardware import XsensStream
    with pytest.raises(RuntimeError, match='error code 28'):
        XsensStream(np.eye(3)).handle(0x42, bytes((0x28,)), 0.0)


def test_mtdata2_fields_extracts_every_logged_quantity_and_marks_missing_ones_nan():
    from le_snail.hardware import mtdata2_fields, parse_mtdata2
    acc = struct.pack('>HB', 0x4020, 12) + struct.pack('>3f', 0.1, 0.2, 9.8)
    fields = mtdata2_fields(parse_mtdata2(mtdata2(7, 1234, q_wxyz=(0.5, 0.5, 0.5, 0.5), gyro=(1., 2., 3.), extra=acc)))
    assert fields[:2] == [7, 1234]
    np.testing.assert_allclose(fields[2:], [0.5, 0.5, 0.5, 0.5, 1, 2, 3, 0.1, 0.2, 9.8], rtol=1e-6)
    missing = mtdata2_fields(parse_mtdata2(mtdata2(8, 1300, quaternion_id=None)))
    assert missing[:2] == [8, 1300] and np.all(np.isnan(missing[2:6])) and np.all(np.isnan(missing[9:]))


class FakeBno:
    """Adafruit BNO08X stand-in with the report stamping Bno085Imu installs: each processing call delivers a
    fresh pair of reports (identical values, as when the robot sits still) unless frozen."""
    def __init__(self):
        self.frozen, self.enabled, self.report_times, self._readings = False, [], {}, {}

    def enable_feature(self, feature, report_interval=50000):
        self.enabled.append((feature, report_interval))

    def _process_available_packets(self):
        if self.frozen:
            return
        s, c = np.sin(np.radians(15)), np.cos(np.radians(15))
        now = time.monotonic()
        self._readings = {0x08: (s, 0.0, 0.0, c), 0x02: (0.0, 0.0, 0.1)}
        self.report_times = {0x08: now, 0x02: now}


def test_bno085_gives_base_frame_gravity_and_goes_stale_when_reports_stop():
    from le_snail.hardware import Bno085Imu
    fake = FakeBno()
    imu = Bno085Imu(3, np.eye(3), device_factory=lambda bus: fake, poll_s=0.002)
    try:
        assert len(fake.enabled) == 2 and all(interval == 20000 for _, interval in fake.enabled)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                sample = imu.read(0.05)
                break
            except RuntimeError:
                time.sleep(0.01)
        # Sensor tilted +30 deg about x: gravity (world -z) seen in the sensor frame.
        assert np.allclose(sample['gravity'], [0.0, -np.sin(np.radians(30)), -np.cos(np.radians(30))], atol=1e-6)
        assert sample['gyro'][2] == pytest.approx(0.1)
        time.sleep(0.1)
        assert imu.read(0.05)['seq'] > sample['seq']      # identical values still count as fresh reports
        fake.frozen = True
        time.sleep(0.1)
        with pytest.raises(RuntimeError, match='Stale'):
            imu.read(0.05)
    finally:
        imu.close()


def test_out_of_order_samples_at_startup_restart_the_decoder_but_not_later():
    from le_snail.hardware import XsensStream
    stream = XsensStream(np.eye(3), warmup_s=0.5)
    assert stream.handle(0x36, mtdata2(10, 5000), 0.00)['seq'] == 10
    assert stream.handle(0x36, mtdata2(3, 1000), 0.10) is None        # stale backlog packet at port open
    assert stream.handle(0x36, mtdata2(4, 1100), 0.11) is not None     # decoder restarted from the new packet
    assert stream.handle(0x36, mtdata2(5, 1200), 0.12)['seq'] == 5
    with pytest.raises(ValueError, match='out-of-order'):
        stream.handle(0x36, mtdata2(2, 900), 0.90)                    # after the warm-up it is fatal
