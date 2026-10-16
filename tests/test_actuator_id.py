import numpy as np
import pytest
from le_snail.actuator_id import (
    EnvelopeMonitor, SafetyLimits, abort_reason, chirp, closest_recent_goal, plan_motion, decode_state, hold_then, joint_window, ramp, rate_summary, steps,
)
from le_snail.servo_bus import STATE_BLOCK


def block(position=2048, velocity=0, load=0, voltage=121, temperature=30, status=0, moving=0, current=0):
    data = bytearray(STATE_BLOCK[1])
    data[0:2] = position.to_bytes(2, 'little')
    data[2:4] = (abs(velocity) | (0x8000 if velocity < 0 else 0)).to_bytes(2, 'little')
    data[4:6] = (abs(load) | (0x400 if load < 0 else 0)).to_bytes(2, 'little')
    data[6], data[7], data[9], data[10] = voltage, temperature, status, moving
    data[13:15] = current.to_bytes(2, 'little')
    return bytes(data)


def test_state_block_decodes_present_registers():
    state = decode_state(block(3000, -12, -300, 120, 41, 0x20, 1, 77))
    assert state == {'position': 3000, 'velocity': -12, 'load': -300, 'voltage_v': 12.0, 'temperature_c': 41,
                     'status': 0x20, 'moving': 1, 'current': 77}


LIMITS = SafetyLimits()


def test_nominal_state_does_not_abort():
    assert abort_reason({1: decode_state(block()), 2: decode_state(block())}, LIMITS) is None


@pytest.mark.parametrize('field, value, reason', [
    ('temperature', 56, 'temperature'), ('voltage', 99, 'voltage'), ('voltage', 127, 'voltage'),
    ('load', -950, 'load'), ('current', 900, 'current'), ('status', 0x01, 'status'),
])
def test_each_envelope_violation_aborts_and_names_the_servo(field, value, reason):
    states = {1: decode_state(block()), 4: decode_state(block(**{field: value}))}
    message = abort_reason(states, LIMITS)
    assert reason in message and 'id 4' in message


def test_tracking_error_aborts_only_against_a_commanded_goal():
    states = {1: decode_state(block(position=2048 + 300))}
    assert abort_reason(states, LIMITS) is None
    assert 'tracking' in abort_reason(states, LIMITS, goals={1: 2048})


def test_too_many_consecutive_lost_cycles_abort():
    assert abort_reason({}, LIMITS, consecutive_lost=LIMITS.max_consecutive_lost) is None
    assert 'lost' in abort_reason({}, LIMITS, consecutive_lost=LIMITS.max_consecutive_lost + 1)


def test_rate_summary_reports_cycle_rate_transaction_time_and_losses():
    sent = np.arange(5) * 4_000_000          # 4 ms period
    received = sent + 1_500_000              # 1.5 ms transaction
    summary = rate_summary(sent, received, lost=np.array([0, 1, 0, 0, 2]))
    assert summary['cycles'] == 5
    assert summary['rate_hz'] == pytest.approx(250)
    assert summary['transaction_ms']['p50'] == pytest.approx(1.5)
    assert summary['period_ms']['max'] == pytest.approx(4)
    assert summary['lost_replies'] == 3 and summary['cycles_with_loss'] == 2


def test_joint_window_shrinks_servo_limits_by_the_margin_and_rejects_inverted_ones():
    lower, upper = joint_window({'Min_Position_Limit': 806, 'Max_Position_Limit': 3520}, margin_ticks=100)
    assert (lower, upper) == (906, 3420)
    with pytest.raises(ValueError):
        joint_window({'Min_Position_Limit': 1000, 'Max_Position_Limit': 1100}, margin_ticks=100)


def test_ramp_interpolates_from_start_to_goal_without_overshoot():
    path = ramp(np.array([1000, 3000]), np.array([2000, 3000]), duration_s=1.0, rate_hz=10)
    assert path.shape == (10, 2)
    assert path[0, 0] > 1000 and path[-1, 0] == 2000
    assert np.all(np.diff(path[:, 0]) > 0) and np.all(path[:, 1] == 3000)


def test_steps_move_one_joint_and_return_to_center_between_steps():
    center = np.array([2000, 2000, 2000])
    path = steps(center, joint=1, amplitudes=[50, -50], hold_s=0.5, rate_hz=10)
    assert path.shape == (20, 3)
    assert set(path[:, 1]) == {2050, 2000, 1950}
    assert np.all(path[:, [0, 2]] == 2000)
    assert np.all(path[5:10, 1] == 2000) and np.all(path[15:20, 1] == 2000)


def test_chirp_is_bounded_by_its_amplitude_and_starts_at_center():
    path = chirp(np.array([2000, 2000]), joint=0, amplitude=40, f0_hz=0.2, f1_hz=3.0, duration_s=10, rate_hz=50)
    assert path.shape == (500, 2)
    assert path[0, 0] == 2000
    assert np.abs(path[:, 0] - 2000).max() <= 40 and np.all(path[:, 1] == 2000)


def test_hold_then_prepends_a_hold_at_the_first_goal():
    path = hold_then(np.array([[1, 2], [3, 4]]), hold_s=0.2, rate_hz=10)
    assert path.tolist() == [[1, 2], [1, 2], [1, 2], [3, 4]]


def test_tracking_reference_is_the_closest_goal_within_the_settle_window():
    recent = np.array([[1000, 2000], [1300, 2000], [1350, 2010]])
    assert closest_recent_goal(recent, np.array([1010, 2004])).tolist() == [1000, 2000]
    assert closest_recent_goal(recent, np.array([1400, 2020])).tolist() == [1350, 2010]


def test_plan_ramps_from_the_present_pose_into_the_window_then_plays_offsets_about_the_start_pose():
    present, start = np.array([3006, 2000]), np.array([2500, 2000])
    offsets = np.array([[0, 0], [50, 0], [-50, 0]])
    goals = plan_motion(present, start, offsets, lower=[900, 1000], upper=[2910, 3000], ramp_s=1.0, rate_hz=10)
    assert goals.shape == (13, 2)
    assert goals[9].tolist() == [2500, 2000] and goals[0, 0] < 3006
    assert goals[10:].tolist() == [[2500, 2000], [2550, 2000], [2450, 2000]]


def test_plan_rejects_a_start_pose_or_offsets_outside_the_window_and_a_nonzero_first_offset():
    window = dict(lower=[900], upper=[2910], ramp_s=1.0, rate_hz=10)
    with pytest.raises(ValueError, match='window'):
        plan_motion(np.array([2000]), np.array([2900]), np.array([[0], [20]]), **window)
    with pytest.raises(ValueError, match='zero'):
        plan_motion(np.array([2000]), np.array([2000]), np.array([[5]]), **window)


def test_monitor_ignores_single_sample_temperature_and_voltage_spikes():
    monitor = EnvelopeMonitor(LIMITS)
    for k in range(3 * LIMITS.slow_signal_window):
        spike = k % LIMITS.slow_signal_window == 7
        states = {2: decode_state(block(temperature=58 if spike else 33, voltage=99 if spike else 121))}
        assert monitor.check(states) is None


def test_monitor_aborts_on_persistent_overheat_within_one_window():
    monitor = EnvelopeMonitor(LIMITS)
    reasons = [monitor.check({2: decode_state(block(temperature=60))}) for _ in range(LIMITS.slow_signal_window)]
    assert reasons[:-1] == [None] * (LIMITS.slow_signal_window - 1)
    assert 'temperature' in reasons[-1] and 'id 2' in reasons[-1]


def test_monitor_keeps_fast_checks_instant():
    monitor = EnvelopeMonitor(LIMITS)
    assert 'load' in monitor.check({1: decode_state(block(load=950))})
    assert 'status' in EnvelopeMonitor(LIMITS).check({1: decode_state(block(status=0x20))})
