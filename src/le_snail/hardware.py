"""Calibrated unit mapping and timestamped IMU transport, independent of Isaac."""
from __future__ import annotations

import json
import math
import threading
import time

import numpy as np

from .contract import JOINT_NAMES, finite_vector, imu_base_vectors


def validate_hardware_settings(config, contract):
    if config['calibration_verified'] is not True or config['telemetry_register_units_verified'] is not True:
        raise ValueError('Hardware calibration and telemetry units must be verified')
    if config['contract_fingerprint'] != contract.fingerprint:
        raise ValueError('Hardware configuration and policy contract differ')
    for key in ('max_feedback_age_s', 'max_tracking_error_rad', 'start_tolerance_rad', 'max_temperature_c', 'voltage_v_per_register_unit'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    lo, hi = finite_vector(config['voltage_range_v'], 2, 'voltage range')
    if not 0 < lo < hi:
        raise ValueError('Voltage limits must be positive and ordered')
    imu_base_vectors([1, 0, 0, 0], [0, 0, 0], config['imu_rotation_base_from_sensor'])
    mapping = JointMapping(config['mapping'])
    mapping.to_native(contract.lower_rad)
    mapping.to_native(contract.upper_rad)
    return mapping


class JointMapping:
    """URDF radians = scale * LeRobot native value + offset.

    Body native values are degrees (use_degrees=True); gripper native values are 0..100.
    Coefficients must be measured; gripper endpoints cannot be inferred from the URDF.
    """
    def __init__(self, config):
        self.scale = finite_vector(config['radians_per_native_unit'], 6, 'joint scale')
        self.offset = finite_vector(config['offset_rad'], 6, 'joint offset')
        self.lower = finite_vector(config['native_lower'], 6, 'native lower limits')
        self.upper = finite_vector(config['native_upper'], 6, 'native upper limits')
        if np.any(self.scale == 0) or np.any(self.lower >= self.upper):
            raise ValueError('Invalid joint mapping or native limits')

    def to_radians(self, observation):
        native = finite_vector([observation[f'{name}.pos'] for name in JOINT_NAMES], 6, 'native feedback')
        if np.any(native < self.lower) or np.any(native > self.upper):
            raise ValueError('Feedback outside validated native limits')
        return native * self.scale + self.offset

    def to_native(self, radians):
        q = finite_vector(radians, 6, 'joint target')
        native = (q - self.offset) / self.scale
        if np.any(native < self.lower) or np.any(native > self.upper):
            raise ValueError('Command outside validated native limits')
        return {f'{name}.pos': float(value) for name, value in zip(JOINT_NAMES, native)}


class ImuDecoder:
    def __init__(self, mounting):
        self.mounting = mounting
        self.last_seq = -1
        self.last_sample_us = -1
        self.clock_offset = None

    def decode(self, line, received_at):
        packet = json.loads(line)
        seq, timestamp = packet['seq'], packet['sample_us']
        if type(seq) is not int or type(timestamp) is not int or timestamp < 0:
            raise ValueError('Invalid IMU sequence/timestamp')
        if seq <= self.last_seq or timestamp <= self.last_sample_us:
            raise ValueError('IMU replay, clock reset, or out-of-order sample')
        sensor_s = timestamp / 1e6
        # Minimum observed transport offset approximates clock alignment without accepting queue buildup.
        offset = received_at - sensor_s
        self.clock_offset = offset if self.clock_offset is None else min(self.clock_offset, offset)
        sample_at = sensor_s + self.clock_offset
        gravity, gyro = imu_base_vectors(packet['q_wxyz'], packet['gyro_rad_s'], self.mounting)
        self.last_seq, self.last_sample_us = seq, timestamp
        return {'gravity': gravity, 'gyro': gyro, 'sample_at': sample_at, 'received_at': received_at, 'seq': seq}


class ImuSerial:
    def __init__(self, port, mounting, baudrate=460800):
        import serial
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=0.05)
        self.serial.reset_input_buffer()
        self.decoder = ImuDecoder(mounting)
        self.latest = None
        self.error = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        # The first line can be a partial packet after opening/flushing the port.
        synchronized = False
        pending = bytearray()
        while not self.stop.is_set():
            try:
                data = self.serial.read_until(b'\n', size=2048)
                if not data:
                    continue
                pending.extend(data)
                if len(pending) > 4096:
                    raise ValueError('IMU packet exceeds size limit')
                if not pending.endswith(b'\n'):
                    continue
                line, pending = bytes(pending), bytearray()
                if not synchronized:
                    synchronized = True
                    continue
                decoded = self.decoder.decode(line, time.monotonic())
                with self.lock:
                    self.latest = decoded
            except Exception as exc:
                with self.lock:
                    self.error = exc
                return

    def read(self, max_age_s):
        with self.lock:
            if self.error is not None:
                raise RuntimeError('IMU stream failed') from self.error
            result = self.latest
        if result is None:
            raise RuntimeError('No IMU sample received')
        now = time.monotonic()
        if now - result['sample_at'] > max_age_s or now - result['received_at'] > max_age_s:
            raise RuntimeError('Stale IMU sample')
        return result

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.2)
        self.serial.close()
