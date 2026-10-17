"""Calibrated unit mapping and timestamped IMU transport, independent of Isaac."""
from __future__ import annotations

import math
import struct
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
    if type(config['imu_baudrate']) is not int or config['imu_baudrate'] <= 0:
        raise ValueError('imu_baudrate must be a positive integer')
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


# Xsens Xbus constants, checked against Xsens' public driver sources (xspublic, in
# github.com/xsenssupport/Xsens_MTi_ROS_Driver_and_Ntrip_Client @ 71c487c):
#   framing/checksum: xscommon/xbus.{h,c}; message ids: xstypes/xsxbusmessageid.h;
#   data identifiers: xstypes/xsdataidentifier.h; MTData2 item layout: xstypes/xsdatapacket.cpp.
# All multi-byte fields are big-endian (XsMessage_getEndianCorrectData swaps them).
XBUS_PREAMBLE = 0xFA
XBUS_MASTER_DEVICE = 0xFF
XBUS_EXTENDED_LENGTH = 0xFF
XMID_REQ_DID = 0x00
XMID_GOTO_MEASUREMENT = 0x10
XMID_REQ_FIRMWARE_REVISION = 0x12
XMID_REQ_PRODUCT_CODE = 0x1C
XMID_GOTO_CONFIG = 0x30
XMID_MTDATA2 = 0x36
XMID_ERROR = 0x42
XMID_REQ_AVAILABLE_FILTER_PROFILES = 0x62
XMID_REQ_FILTER_PROFILE = 0x64
XMID_REQ_OUTPUT_CONFIGURATION = 0xC0
XMID_REQ_ALIGNMENT_ROTATION = 0xEC
XDI_FULL_TYPE_MASK = 0xFFF0
XDI_SUBFORMAT_MASK = 0x0003
XDI_SUBFORMAT_FLOAT = 0x0000
XDI_COORDSYS_MASK = 0x000C
XDI_COORDSYS_ENU = 0x0000
XDI_PACKET_COUNTER = 0x1020
XDI_SAMPLE_TIME_FINE = 0x1060
XDI_QUATERNION = 0x2010
XDI_ACCELERATION = 0x4020
XDI_RATE_OF_TURN = 0x8020
SAMPLE_TIME_FINE_HZ = 10_000.0


def xbus_frame(mid, payload=b'', bid=XBUS_MASTER_DEVICE):
    """Encode one Xbus message: preamble, bus id, message id, length, payload, checksum."""
    payload = bytes(payload)
    if len(payload) < XBUS_EXTENDED_LENGTH:
        header = bytes((bid, mid, len(payload)))
    elif len(payload) <= 0xFFFF:
        header = bytes((bid, mid, XBUS_EXTENDED_LENGTH)) + struct.pack('>H', len(payload))
    else:
        raise ValueError('Xbus payload too long')
    body = header + payload
    # The checksum makes the sum of every byte after the preamble zero modulo 256.
    return bytes((XBUS_PREAMBLE,)) + body + bytes(((-sum(body)) & 0xFF,))


class XbusReader:
    """Incremental Xbus framer: resynchronizes on corrupt data but fails after persistent corruption."""
    def __init__(self, max_buffer=8192, max_consecutive_errors=16):
        self.buffer = bytearray()
        self.max_buffer = max_buffer
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_errors = 0

    def _reject(self):
        del self.buffer[0]
        self.consecutive_errors += 1
        if self.consecutive_errors > self.max_consecutive_errors:
            raise ValueError('Persistent Xbus framing/checksum errors')

    def feed(self, data):
        self.buffer.extend(data)
        messages = []
        while True:
            start = self.buffer.find(XBUS_PREAMBLE)
            if start < 0:
                self.buffer.clear()
                break
            del self.buffer[:start]
            if len(self.buffer) < 4:
                break
            if self.buffer[3] == XBUS_EXTENDED_LENGTH:
                if len(self.buffer) < 6:
                    break
                length, offset = struct.unpack_from('>H', self.buffer, 4)[0], 6
            else:
                length, offset = self.buffer[3], 4
            total = offset + length + 1
            if total > self.max_buffer:
                self._reject()
                continue
            if len(self.buffer) < total:
                break
            if sum(self.buffer[1:total]) & 0xFF:
                self._reject()
                continue
            messages.append((self.buffer[2], bytes(self.buffer[offset:total - 1])))
            del self.buffer[:total]
            self.consecutive_errors = 0
        if len(self.buffer) > self.max_buffer:
            raise ValueError('Xbus buffer exceeds size limit')
        return messages


def parse_mtdata2(payload):
    """Split an MTData2 payload into {data identifier: bytes}; corrupt payloads raise."""
    items, offset = {}, 0
    while offset + 3 <= len(payload):
        identifier, size = struct.unpack_from('>HB', payload, offset)
        if offset + 3 + size > len(payload):
            break
        items[identifier] = payload[offset + 3:offset + 3 + size]
        offset += 3 + size
    if offset != len(payload):
        raise ValueError('Corrupt MTData2 payload')
    return items


def _xsens_item(items, data_type, count, fmt, *, enu=False):
    """Return one item only in a format this decoder handles; never reinterpret other formats/frames."""
    matches = [i for i in items if i & XDI_FULL_TYPE_MASK == data_type]
    if not matches:
        raise ValueError(f'Xsens stream is missing data identifier 0x{data_type:04X}; check the output configuration')
    identifier = matches[0]
    if fmt == 'f':
        if identifier & XDI_SUBFORMAT_MASK != XDI_SUBFORMAT_FLOAT:
            raise ValueError(f'Xsens 0x{identifier:04X} is not float32 output')
        if enu and identifier & XDI_COORDSYS_MASK != XDI_COORDSYS_ENU:
            raise ValueError(f'Xsens 0x{identifier:04X} is not ENU; the gravity convention requires z up')
    elif identifier != data_type:
        raise ValueError(f'Unexpected Xsens identifier 0x{identifier:04X}')
    data = items[identifier]
    if len(data) != struct.calcsize(f'>{count}{fmt}'):
        raise ValueError(f'Unexpected size for Xsens 0x{identifier:04X}')
    return struct.unpack(f'>{count}{fmt}', data)


def mtdata2_fields(items):
    """[counter, sample_time_fine, qw, qx, qy, qz, gx, gy, gz, ax, ay, az] for logging; NaN where absent.

    Unlike ImuDecoder this never raises on a missing item, so start-up packets are kept in the log.
    """
    def item(data_type, count, fmt):
        try:
            return list(_xsens_item(items, data_type, count, fmt, enu=data_type == XDI_QUATERNION))
        except ValueError:
            return [math.nan] * count
    return (item(XDI_PACKET_COUNTER, 1, 'H') + item(XDI_SAMPLE_TIME_FINE, 1, 'I') + item(XDI_QUATERNION, 4, 'f')
            + item(XDI_RATE_OF_TURN, 3, 'f') + item(XDI_ACCELERATION, 3, 'f'))


class ImuDecoder:
    """Decode Xsens MTData2 orientation/gyro packets into base-frame gravity and angular rate."""
    def __init__(self, mounting):
        self.mounting = mounting
        self.seq = None
        self.last_counter = None
        self.last_sample_ticks = None
        self.clock_offset = None

    def decode(self, payload, received_at):
        items = parse_mtdata2(payload)
        counter, = _xsens_item(items, XDI_PACKET_COUNTER, 1, 'H')
        ticks, = _xsens_item(items, XDI_SAMPLE_TIME_FINE, 1, 'I')
        # The orientation filter output is required; raw acceleration is never substituted for gravity.
        q_wxyz = _xsens_item(items, XDI_QUATERNION, 4, 'f', enu=True)
        gyro = _xsens_item(items, XDI_RATE_OF_TURN, 3, 'f')
        # SampleTimeFine is 32-bit (wraps after ~119 h), so a decrease indicates a restart or replay.
        if self.last_sample_ticks is not None and ticks <= self.last_sample_ticks:
            raise ValueError('IMU replay, clock reset, or out-of-order sample')
        # PacketCounter wraps at 65536; unwrap it for a monotonic sequence and reject backwards steps.
        if self.last_counter is None:
            seq = counter
        else:
            step = (counter - self.last_counter) % 65536
            if step == 0 or step >= 32768:
                raise ValueError('IMU replay, clock reset, or out-of-order sample')
            seq = self.seq + step
        sensor_s = ticks / SAMPLE_TIME_FINE_HZ
        # Minimum observed transport offset approximates clock alignment without accepting queue buildup.
        offset = received_at - sensor_s
        self.clock_offset = offset if self.clock_offset is None else min(self.clock_offset, offset)
        sample_at = sensor_s + self.clock_offset
        gravity, gyro = imu_base_vectors(q_wxyz, gyro, self.mounting)
        self.seq, self.last_counter, self.last_sample_ticks = seq, counter, ticks
        return {'gravity': gravity, 'gyro': gyro, 'sample_at': sample_at, 'received_at': received_at, 'seq': seq}


# XRV_DATAOVERFLOW (xspublic xsresultvalue.h): the device produced more data than the bus carried,
# observed when the port has not been read for a while. Asynchronous, not a reply to a command.
XRV_DATAOVERFLOW = 0x29


class XsensStream:
    """Turn Xbus messages into IMU samples, tolerating only the known start-up artefacts.

    When the port opens, the MTi reports any backlog overflow and may send a few packets
    without orientation. Those are counted and skipped during a short warm-up window; after
    it, any error message or incomplete packet stops the stream.
    """
    def __init__(self, mounting, warmup_s=0.5):
        self.decoder = ImuDecoder(mounting)
        self.warmup_s = warmup_s
        self.started_at = None
        self.overflow_reports = 0
        self.skipped_warmup_packets = 0

    def handle(self, mid, payload, received_at):
        """Return a decoded sample, or None for a message that carries no sample."""
        if self.started_at is None:
            self.started_at = received_at
        if mid == XMID_ERROR:
            if payload[:1] == bytes((XRV_DATAOVERFLOW,)):
                self.overflow_reports += 1
                return None
            raise RuntimeError(f'Xsens reported error code {payload.hex() or "(none)"}')
        if mid != XMID_MTDATA2:
            return None
        try:
            return self.decoder.decode(payload, received_at)
        except ValueError as exc:
            if received_at - self.started_at <= self.warmup_s and 'missing data identifier' in str(exc):
                self.skipped_warmup_packets += 1
                return None
            raise


def enable_low_latency(port):
    """Ask the USB-serial driver to deliver bytes immediately.

    The dev board's FTDI chip otherwise holds a partly filled USB buffer for its 16 ms latency
    timer, and every 73-byte MTData2 frame overflows one 62-byte FTDI packet, so samples arrive
    ~40 ms late in bursts. ftdi_sio maps ASYNC_LOW_LATENCY (user-settable, no root) to 1 ms.
    Returns False when the platform or driver does not support it.
    """
    try:
        port.set_low_latency_mode(True)
        return True
    except (AttributeError, NotImplementedError, OSError, ValueError):
        return False


class ImuSerial:
    """Read an Xsens MTi already in measurement mode. Never writes to the device."""
    def __init__(self, port, mounting, baudrate):
        import serial
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=0.05)
        self.low_latency = enable_low_latency(self.serial)
        self.serial.reset_input_buffer()
        self.reader = XbusReader()
        self.stream = XsensStream(mounting)
        self.latest = None
        self.error = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while not self.stop.is_set():
            try:
                data = self.serial.read(self.serial.in_waiting or 1)
                if not data:
                    continue
                received_at = time.monotonic()
                for mid, payload in self.reader.feed(data):
                    decoded = self.stream.handle(mid, payload, received_at)
                    if decoded is not None:
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


# SH-2 report ids (same values as adafruit_bno08x.BNO_REPORT_*), so tests need no Adafruit install.
BNO_REPORT_GYROSCOPE = 0x02
BNO_REPORT_GAME_ROTATION_VECTOR = 0x08


def _adafruit_bno085(bus):
    """Adafruit BNO08X_I2C that stamps each decoded report (adafruit-circuitpython-bno08x 1.3.1 internals)."""
    from adafruit_bno08x.i2c import BNO08X_I2C
    from adafruit_extended_bus import ExtendedI2C

    class StampedBno08x(BNO08X_I2C):
        report_times = None

        def _process_report(self, report_id, report_bytes):
            super()._process_report(report_id, report_bytes)
            if self.report_times is None:
                self.report_times = {}
            self.report_times[report_id] = time.monotonic()

    device = StampedBno08x(ExtendedI2C(bus))
    device.report_times = {}
    return device


class Bno085Imu:
    """BNO085 on I2C (Adafruit driver): game rotation vector + gyro, drained on a thread.

    Same read() contract as ImuSerial (base-frame gravity unit vector and angular rate, sample freshness).
    The game rotation vector is gravity-referenced with free yaw, which is all the policy uses. Freshness
    comes from report arrival (the device stamps every decoded report), not from value changes: a robot at
    rest repeats identical gyro and quaternion values. sample_at is the older of the two latest reports.
    Reports are requested at 50 Hz (the Adafruit default is 20 Hz); on the Pi's bit-banged i2c-gpio bus
    (needed because the BNO085's clock stretching upsets the Pi 5 hardware I2C) that keeps the bus ~50 %
    busy, measured 2026-10-17; 100 Hz works but loads it ~90 %.
    """
    REPORT_INTERVAL_US = 20000

    def __init__(self, bus, mounting, device_factory=_adafruit_bno085, poll_s=0.002, attempts=4):
        self.mounting = mounting
        for attempt in range(attempts):
            try:
                self.device = device_factory(bus)
                self.device.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR, self.REPORT_INTERVAL_US)
                time.sleep(0.2)
                self.device.enable_feature(BNO_REPORT_GYROSCOPE, self.REPORT_INTERVAL_US)
                time.sleep(0.2)
                break
            except Exception as exc:
                if attempt == attempts - 1:
                    raise RuntimeError(f'BNO085 init failed on I2C bus {bus} after {attempts} attempts: {exc!r} '
                                       '(power-cycle the sensor)') from exc
                time.sleep(1.5)
        self.poll_s, self.latest, self.error, self.seq = poll_s, None, None, 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        last = None
        failures = 0
        while not self.stop.is_set():
            try:
                self.device._process_available_packets()
                failures = 0
            except Exception as exc:  # occasional I2C hiccups; a persistent failure ends the stream
                failures += 1
                if failures > 50:
                    with self.lock:
                        self.error = exc
                    return
                time.sleep(self.poll_s)
                continue
            times = self.device.report_times
            stamps = (times.get(BNO_REPORT_GAME_ROTATION_VECTOR), times.get(BNO_REPORT_GYROSCOPE))
            if None not in stamps and stamps != last:
                last = stamps
                i, j, k, real = self.device._readings[BNO_REPORT_GAME_ROTATION_VECTOR]
                gyro = self.device._readings[BNO_REPORT_GYROSCOPE]
                gravity, rate = imu_base_vectors((real, i, j, k), gyro, self.mounting)
                self.seq += 1
                with self.lock:
                    self.latest = {'gravity': gravity, 'gyro': rate, 'sample_at': min(stamps), 'received_at': max(stamps),
                                   'seq': self.seq, 'quaternion_wxyz': (real, i, j, k)}
            time.sleep(self.poll_s)

    def read(self, max_age_s):
        with self.lock:
            if self.error is not None:
                raise RuntimeError('IMU stream failed') from self.error
            result = self.latest
        if result is None:
            raise RuntimeError('No IMU sample received')
        if time.monotonic() - result['sample_at'] > max_age_s:
            raise RuntimeError('Stale IMU sample')
        return result

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.2)
