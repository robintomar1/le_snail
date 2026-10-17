"""Stage 1 actuator identification (#12): state decoding, the run safety envelope, bus-rate statistics and
bounded excitation motions, all in raw servo units (encoder ticks, 4096 per turn).

Units of Present_Load (0.1% PWM duty) and Present_Current (~6.5 mA) are the reseller's and unverified;
the load/current thresholds below are provisional and must be reviewed before the first powered run.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .servo_bus import STATE_BLOCK, decode_sign_magnitude

TICKS_PER_RAD = 4096 / (2 * np.pi)


def decode_state(params):
    """Raw Present_* block (addresses 56-70) -> decoded values in register units."""
    if len(params) != STATE_BLOCK[1]:
        raise ValueError('State block has the wrong length')
    word = lambda offset: int.from_bytes(params[offset:offset + 2], 'little')
    return {
        'position': decode_sign_magnitude(word(0), 15),
        'velocity': decode_sign_magnitude(word(2), 15),
        'load': decode_sign_magnitude(word(4), 10),
        'voltage_v': params[6] / 10,
        'temperature_c': params[7],
        'status': params[9],
        'moving': params[10],
        'current': word(13),
    }


@dataclass(frozen=True)
class SafetyLimits:
    max_temperature_c: float = 55.0        # hardware config gate; servo protection trips at 70 C
    voltage_range_v: tuple = (10.0, 12.6)
    max_abs_load: int = 900                # 0.1 % PWM duty (provisional)
    max_current: int = 500                 # ~6.5 mA units, ~3.3 A (provisional)
    max_tracking_error_ticks: int = round(0.35 * TICKS_PER_RAD)  # hardware config max_tracking_error_rad
    max_consecutive_lost: int = 5
    max_start_drift_ticks: int = round(0.15 * TICKS_PER_RAD)  # hardware config start_tolerance_rad
    slow_signal_window: int = 25           # samples per servo (~50 ms) for the temperature/voltage median
    load_persist_s: float = 0.3            # |load| must stay above max_abs_load this long (hard moves saturate briefly)


def abort_reason(states, limits, goals=None, consecutive_lost=0):
    """First envelope violation as a message, or None. `states`: {id: decode_state(...)}; `goals`: {id: ticks}."""
    if consecutive_lost > limits.max_consecutive_lost:
        return f'{consecutive_lost} consecutive cycles lost a reply'
    lo, hi = limits.voltage_range_v
    for servo_id, s in states.items():
        if s['temperature_c'] > limits.max_temperature_c:
            return f'id {servo_id}: temperature {s["temperature_c"]} C above {limits.max_temperature_c}'
        if not lo <= s['voltage_v'] <= hi:
            return f'id {servo_id}: voltage {s["voltage_v"]} V outside {lo}-{hi}'
        if abs(s['load']) > limits.max_abs_load:
            return f'id {servo_id}: load {s["load"]} above {limits.max_abs_load}'
        if s['current'] > limits.max_current:
            return f'id {servo_id}: current {s["current"]} above {limits.max_current}'
        if s['status']:
            return f'id {servo_id}: status error bits 0x{s["status"]:02x}'
        if goals and servo_id in goals and abs(s['position'] - goals[servo_id]) > limits.max_tracking_error_ticks:
            return f'id {servo_id}: tracking error {s["position"] - goals[servo_id]} ticks'
    return None


class EnvelopeMonitor:
    """abort_reason over a stream of cycles, with temperature and voltage judged on a rolling median.

    STS3215 reads show single-sample spikes (temperature up to +25 C, voltage down ~1.4 V) in ~0.5 % of
    samples per servo even with torque off (#12 hold run of 2026-10-16 aborted on one 58 C sample).
    Real heating and supply sag are far slower than the window, so they still trip it within ~50 ms;
    Load aborts only once |load| has stayed above the limit for load_persist_s: a fast 30 deg shoulder_lift
    step against gravity briefly reaches ~91 % duty, while a stalled or blocked joint stays there. Current,
    status, tracking and lost feedback stay per-sample.
    """
    def __init__(self, limits):
        self.limits, self.history, self.overload_since = limits, {}, {}

    def check(self, states, goals=None, consecutive_lost=0, now_s=0.0):
        judged = {}
        for servo_id, s in states.items():
            if abs(s['load']) > self.limits.max_abs_load:
                since = self.overload_since.setdefault(servo_id, now_s)
                if now_s - since < self.limits.load_persist_s:
                    s = dict(s, load=0)  # not sustained yet
            else:
                self.overload_since.pop(servo_id, None)
            window = self.limits.slow_signal_window
            temperatures, voltages = self.history.setdefault(servo_id, (deque(maxlen=window), deque(maxlen=window)))
            temperatures.append(s['temperature_c'])
            voltages.append(s['voltage_v'])
            if len(temperatures) < window:  # no slow-signal verdict until the window fills
                judged[servo_id] = dict(s, temperature_c=0, voltage_v=self.limits.voltage_range_v[0])
            else:  # odd window: the middle element is the median
                judged[servo_id] = dict(s, temperature_c=sorted(temperatures)[window // 2],
                                        voltage_v=sorted(voltages)[window // 2])
        return abort_reason(judged, self.limits, goals, consecutive_lost)


def _stats(values_ms):
    return {'p50': float(np.percentile(values_ms, 50)), 'p99': float(np.percentile(values_ms, 99)),
            'max': float(np.max(values_ms))}


def rate_summary(sent_ns, received_ns, lost):
    sent_ns, received_ns, lost = map(np.asarray, (sent_ns, received_ns, lost))
    summary = {'cycles': int(len(sent_ns)), 'transaction_ms': _stats((received_ns - sent_ns) / 1e6),
               'lost_replies': int(lost.sum()), 'cycles_with_loss': int(np.count_nonzero(lost))}
    if len(sent_ns) > 1:
        summary['rate_hz'] = float((len(sent_ns) - 1) / ((sent_ns[-1] - sent_ns[0]) / 1e9))
        summary['period_ms'] = _stats(np.diff(sent_ns) / 1e6)
    return summary


def joint_window(registers, margin_ticks):
    """Software goal window: the servo's calibrated EEPROM position limits shrunk by a margin."""
    lower = registers['Min_Position_Limit'] + margin_ticks
    upper = registers['Max_Position_Limit'] - margin_ticks
    if lower >= upper:
        raise ValueError('Position limits leave no window after the margin')
    return lower, upper


def urdf_to_ticks(targets_rad, signs, offsets):
    """URDF joint angles (samples, joints) to goal ticks: ticks = offset + sign * rad * TICKS_PER_RAD."""
    return np.rint(np.asarray(offsets) + np.asarray(signs) * np.asarray(targets_rad) * TICKS_PER_RAD).astype(int)


# Motions are integer goal-tick arrays of shape (samples, joints) at a fixed command rate.

def ramp(start, goal, duration_s, rate_hz):
    n = max(1, round(duration_s * rate_hz))
    fraction = np.arange(1, n + 1)[:, None] / n
    return np.rint(start + (goal - start) * fraction).astype(int)


def steps(center, joint, amplitudes, hold_s, rate_hz):
    n = round(hold_s * rate_hz)
    rows = []
    for amplitude in amplitudes:
        target = np.array(center, dtype=int)
        target[joint] += amplitude
        rows += [target] * n + [np.array(center, dtype=int)] * n
    return np.array(rows)


def chirp(center, joint, amplitude, f0_hz, f1_hz, duration_s, rate_hz):
    t = np.arange(round(duration_s * rate_hz)) / rate_hz
    phase = 2 * np.pi * (f0_hz * t + (f1_hz - f0_hz) * t ** 2 / (2 * duration_s))
    path = np.tile(np.array(center, dtype=int), (len(t), 1))
    path[:, joint] += np.rint(amplitude * np.sin(phase)).astype(int)
    return path


def hold_then(path, hold_s, rate_hz):
    return np.concatenate([np.repeat(path[:1], round(hold_s * rate_hz), axis=0), path])


def closest_recent_goal(recent_goals, positions):
    """Per joint, the goal from the recent window nearest the measured position.

    Tracking error is judged against this so a large step is not an error until it has had the settle
    window to arrive, while a servo that stalls far from every recent goal still trips the envelope.
    """
    recent_goals = np.asarray(recent_goals)
    nearest = np.argmin(np.abs(recent_goals - np.asarray(positions)), axis=0)
    return recent_goals[nearest, np.arange(recent_goals.shape[1])]


def plan_motion(present, start, offsets, lower, upper, ramp_s, rate_hz):
    """Absolute goals: a ramp from the present pose to `start`, then `start + offsets`.

    The present pose may lie outside the window (a limp arm often rests against a limit); the start pose
    and every offset goal must lie inside it, and the offsets must begin at zero so nothing jumps.
    """
    offsets = np.asarray(offsets, dtype=int)
    if np.any(offsets[0]):
        raise ValueError('The motion must begin at zero offset')
    motion = np.asarray(start, dtype=int) + offsets
    if np.any(motion < np.asarray(lower)) or np.any(motion > np.asarray(upper)):
        outside = np.flatnonzero(np.any((motion < np.asarray(lower)) | (motion > np.asarray(upper)), axis=0))
        raise ValueError(f'Goals leave the window on joints {outside.tolist()}')
    return np.concatenate([ramp(np.asarray(present), np.asarray(start), ramp_s, rate_hz), motion])


def step_metrics(t, pos, t_cmd, goal, t_end, settle_s=0.2, move_ticks=2):
    """Response of one joint to a goal step at t_cmd (seconds, ticks); fractions are of the distance to travel.

    Dead time is to the first move of `move_ticks` from the pre-step median; rise and overshoot are relative
    to goal - start, so a servo that stops short of the goal (static friction, no integral term) shows a
    final error rather than a failed rise.
    """
    t, pos = np.asarray(t, dtype=float), np.asarray(pos, dtype=float)
    start = float(np.median(pos[(t >= t_cmd - 0.1) & (t < t_cmd)]))
    window = (t >= t_cmd) & (t < t_end)
    tt, x = t[window] - t_cmd, (pos[window] - start) / (goal - start)

    def first(mask):
        hits = np.flatnonzero(mask)
        return float(tt[hits[0]]) if len(hits) else float('nan')
    final = float(np.median(pos[window][tt >= tt[-1] - settle_s]))
    return {'start_ticks': start, 'step_ticks': goal - start,
            'dead_time_s': first(np.abs(pos[window] - start) >= move_ticks),
            'rise_10_90_s': first(x >= 0.9) - first(x >= 0.1), 'time_to_90_s': first(x >= 0.9),
            'overshoot': max(0.0, float(x.max()) - 1), 'final_error_ticks': final - goal}


def frequency_response(t_in, u, t_out, y, bands, dt=0.005):
    """Band-averaged transfer u -> y from one excitation record (e.g. a chirp), on a uniform resampling.

    Each band's estimate is sum(Y conj(U)) / sum(|U|^2) over its FFT bins; delay is -phase / (2 pi f_mid),
    so it is meaningful only while the phase lag stays under half a turn.
    """
    t = np.arange(max(t_in[0], t_out[0]), min(t_in[-1], t_out[-1]), dt)
    u = np.interp(t, t_in, u) - np.mean(u)
    y = np.interp(t, t_out, y) - np.mean(y)
    U, Y, f = np.fft.rfft(u), np.fft.rfft(y), np.fft.rfftfreq(len(t), dt)
    result = []
    for lo, hi in bands:
        b = (f >= lo) & (f < hi)
        h = np.sum(Y[b] * np.conj(U[b])) / np.sum(np.abs(U[b]) ** 2)
        mid = (lo + hi) / 2
        result.append({'f_hz': mid, 'gain': float(np.abs(h)), 'phase_deg': float(np.degrees(np.angle(h))),
                       'delay_s': float(-np.angle(h) / (2 * np.pi * mid))})
    return result
