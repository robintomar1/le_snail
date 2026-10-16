"""Feetech STS3215 bus access for actuator identification, independent of LeRobot and Isaac.

Framing, the register table and the sign-magnitude bits follow LeRobot at the commit the deploy path
pins (huggingface/lerobot @ ae2b57f: src/lerobot/motors/feetech/{tables,feetech}.py). Multi-byte
registers are little-endian. A ServoBus refuses WRITE / SYNC_WRITE unless it was built with
allow_writes=True, so read-only tools cannot move or reconfigure a servo by mistake.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

PING, READ, WRITE, SYNC_READ, SYNC_WRITE = 0x01, 0x02, 0x03, 0x82, 0x83
BROADCAST_ID = 0xFE
BAUDRATE = 1_000_000
LEROBOT_COMMIT = 'ae2b57f12223c8c5184fcd814eb63cd9ce7de203'

# name: (address, size in bytes); STS_SMS_SERIES_CONTROL_TABLE at the pinned commit.
STS3215_REGISTERS = {
    # EEPROM
    'Firmware_Major_Version': (0, 1), 'Firmware_Minor_Version': (1, 1), 'Model_Number': (3, 2),
    'ID': (5, 1), 'Baud_Rate': (6, 1), 'Return_Delay_Time': (7, 1), 'Response_Status_Level': (8, 1),
    'Min_Position_Limit': (9, 2), 'Max_Position_Limit': (11, 2), 'Max_Temperature_Limit': (13, 1),
    'Max_Voltage_Limit': (14, 1), 'Min_Voltage_Limit': (15, 1), 'Max_Torque_Limit': (16, 2), 'Phase': (18, 1),
    'Unloading_Condition': (19, 1), 'LED_Alarm_Condition': (20, 1), 'P_Coefficient': (21, 1),
    'D_Coefficient': (22, 1), 'I_Coefficient': (23, 1), 'Minimum_Startup_Force': (24, 2), 'CW_Dead_Zone': (26, 1),
    'CCW_Dead_Zone': (27, 1), 'Protection_Current': (28, 2), 'Angular_Resolution': (30, 1),
    'Homing_Offset': (31, 2), 'Operating_Mode': (33, 1), 'Protective_Torque': (34, 1), 'Protection_Time': (35, 1),
    'Overload_Torque': (36, 1), 'Velocity_closed_loop_P_proportional_coefficient': (37, 1),
    'Over_Current_Protection_Time': (38, 1), 'Velocity_closed_loop_I_integral_coefficient': (39, 1),
    # SRAM
    'Torque_Enable': (40, 1), 'Acceleration': (41, 1), 'Goal_Position': (42, 2), 'Goal_Time': (44, 2),
    'Goal_Velocity': (46, 2), 'Torque_Limit': (48, 2), 'Lock': (55, 1), 'Present_Position': (56, 2),
    'Present_Velocity': (58, 2), 'Present_Load': (60, 2), 'Present_Voltage': (62, 1),
    'Present_Temperature': (63, 1), 'Status': (65, 1), 'Moving': (66, 1), 'Present_Current': (69, 2),
    'Goal_Position_2': (71, 2),
    # Factory
    'Moving_Velocity_Threshold': (80, 1), 'DTs': (81, 1), 'Velocity_Unit_factor': (82, 1), 'Hts': (83, 1),
    'Maximum_Velocity_Limit': (84, 1), 'Maximum_Acceleration': (85, 1), 'Acceleration_Multiplier': (86, 1),
}
# STS_SMS_SERIES_ENCODINGS_TABLE: sign bit of sign-magnitude registers.
STS3215_SIGN_BITS = {'Present_Load': 10, 'Homing_Offset': 11, 'Goal_Position': 15, 'Goal_Velocity': 15,
                     'Present_Position': 15, 'Present_Velocity': 15}
# READ chunks for a full dump: EEPROM 0-39, SRAM 40-70, and the factory area 80-86, which holds the
# Maximum_Acceleration register that LeRobot configure() writes.
DUMP_RANGES = ((0, 40), (40, 31), (80, 7))
# Contiguous Present_* block a Stage 1 cycle reads: position (56) .. current (69-70).
STATE_BLOCK = (56, 15)

# What SOFollower.connect() -> configure() writes at the pinned commit (so_follower.py:184-196,
# feetech.py:209-225, config_so_follower.py:45-47). Phase is checked separately: only bit 4 is cleared.
LEROBOT_CONFIGURE = {'Return_Delay_Time': 0, 'Maximum_Acceleration': 254, 'Acceleration': 254,
                     'Operating_Mode': 0, 'P_Coefficient': 16, 'I_Coefficient': 0, 'D_Coefficient': 32}
GRIPPER_ID = 6
LEROBOT_CONFIGURE_GRIPPER = {'Max_Torque_Limit': 500, 'Protection_Current': 250, 'Overload_Torque': 25}


def _checksum(body):
    return ~sum(body) & 0xFF


def sts_packet(servo_id, instruction, params=b''):
    body = bytes((servo_id, len(params) + 2, instruction)) + bytes(params)
    return b'\xff\xff' + body + bytes((_checksum(body),))


def status_frame(servo_id, error, params=b''):
    """A servo's reply: same framing as a request, with the error byte in the instruction slot."""
    return sts_packet(servo_id, error, params)


def sync_read_packet(ids, address, length):
    return sts_packet(BROADCAST_ID, SYNC_READ, bytes((address, length, *ids)))


def sync_write_packet(values, address, length):
    params = bytes((address, length))
    for servo_id, value in values.items():
        params += bytes((servo_id,)) + int(value).to_bytes(length, 'little')
    return sts_packet(BROADCAST_ID, SYNC_WRITE, params)


def decode_sign_magnitude(value, sign_bit):
    magnitude = value & ((1 << sign_bit) - 1)
    return -magnitude if value >> sign_bit & 1 else magnitude


class StatusReader:
    """Incremental status-frame parser that resynchronizes on garbage and bad checksums."""
    def __init__(self):
        self.buffer = b''

    def feed(self, data):
        self.buffer += data
        frames = []
        while True:
            start = self.buffer.find(b'\xff\xff')
            if start < 0:
                self.buffer = self.buffer[-1:] if self.buffer.endswith(b'\xff') else b''
                return frames
            self.buffer = self.buffer[start:]
            if len(self.buffer) >= 3 and self.buffer[2] == 0xFF:  # header run longer than two bytes
                self.buffer = self.buffer[1:]
                continue
            if len(self.buffer) < 4:
                return frames
            length = self.buffer[3]
            if length < 2:
                self.buffer = self.buffer[2:]
                continue
            if len(self.buffer) < 4 + length:
                return frames
            frame, rest = self.buffer[:4 + length], self.buffer[4 + length:]
            if _checksum(frame[2:-1]) != frame[-1]:
                self.buffer = self.buffer[2:]
                continue
            frames.append((frame[2], frame[4], frame[5:-1]))
            self.buffer = rest


@dataclass
class SyncReadResult:
    replies: dict = field(default_factory=dict)  # id -> (error byte, parameter bytes)
    lost: list = field(default_factory=list)
    sent_ns: int = 0      # monotonic_ns immediately before the request is written
    received_ns: int = 0  # monotonic_ns once every reply arrived or the timeout expired


class ServoBus:
    """Request/reply transactions on an open half-duplex port (pyserial-like object)."""
    def __init__(self, port, timeout_s=0.01, allow_writes=False, echo=False):
        # echo: the adapter hears its own request before the reply. Off by default because a PING reply
        # with error byte 0x01 (voltage error) is byte-identical to the request and would be discarded.
        self.port, self.timeout_s, self.allow_writes, self.echo = port, timeout_s, allow_writes, echo

    def _transact(self, request, expected_ids, length=None):
        self.port.reset_input_buffer()
        reader, replies, head, echo_checked = StatusReader(), {}, b'', not self.echo
        sent = time.monotonic_ns()
        self.port.write(request)
        deadline = sent + int(self.timeout_s * 1e9)
        while len(replies) < len(expected_ids) and time.monotonic_ns() < deadline:
            data = self.port.read(self.port.in_waiting or 1)
            if not echo_checked:
                head += data
                if head.startswith(request):
                    head = head[len(request):]
                elif request.startswith(head):
                    continue
                data, head, echo_checked = head, b'', True
            for servo_id, error, params in reader.feed(data):
                if servo_id in expected_ids and (length is None or len(params) == length):
                    replies[servo_id] = (error, params)
        lost = [servo_id for servo_id in expected_ids if servo_id not in replies]
        return SyncReadResult(replies, lost, sent, time.monotonic_ns())

    def ping(self, servo_id):
        return self._transact(sts_packet(servo_id, PING), [servo_id]).replies.get(servo_id)

    def read(self, servo_id, address, length):
        request = sts_packet(servo_id, READ, bytes((address, length)))
        return self._transact(request, [servo_id], length).replies.get(servo_id)

    def sync_read(self, ids, address, length):
        return self._transact(sync_read_packet(ids, address, length), list(ids), length)

    def sync_write(self, values, address, length):
        """Broadcast write; servos do not reply. Returns monotonic_ns before the write and after the flush."""
        if not self.allow_writes:
            raise PermissionError('This bus was opened read-only')
        request = sync_write_packet(values, address, length)
        sent = time.monotonic_ns()
        self.port.write(request)
        self.port.flush()
        return sent, time.monotonic_ns()


def open_bus_port(path, baudrate=BAUDRATE, serial_factory=None):
    """Open the servo port through the ESP32-S3 CH343 pass-through.

    DTR released + RTS asserted holds the ESP32 in reset (EN low) so its firmware leaves the shared UART
    lines alone; the lines are set before open() because pyserial's default open asserts both, which
    releases reset and loses the servo replies (#12 handoff, #8 probe of 2026-10-16).
    """
    if serial_factory is None:
        import serial
        serial_factory = serial.Serial
    port = serial_factory()
    port.port, port.baudrate, port.timeout = path, baudrate, 0.001
    port.dtr, port.rts = False, True
    port.open()
    return port


def dump_registers(bus, servo_id, ranges=DUMP_RANGES):
    """READ every range; a range without a valid reply is listed in 'failed' instead of aborting the dump."""
    raw, failed, errors = {}, [], {}
    for address, length in ranges:
        reply = bus.read(servo_id, address, length)
        if reply is None:
            failed.append([address, length])
            continue
        error, params = reply
        if error:
            errors[str(address)] = error
        raw.update({address + i: byte for i, byte in enumerate(params)})
    return {'raw': raw, 'failed': failed, 'errors': errors}


def dump_servos(bus, ids, gripper_id=GRIPPER_ID):
    """{str(id): dump record} for the arm: raw bytes, decoded registers, failed ranges and configure() changes."""
    servos = {}
    for servo_id in ids:
        ping = bus.ping(servo_id)
        if ping is None:
            servos[str(servo_id)] = {'answered': False}
            continue
        dump = dump_registers(bus, servo_id)
        values = decode_registers(dump['raw'])
        servos[str(servo_id)] = {
            'answered': True,
            'ping_error': ping[0],
            'raw': {str(address): byte for address, byte in sorted(dump['raw'].items())},
            'failed_ranges': dump['failed'],
            'error_bytes': dump['errors'],
            'registers': values,
            'lerobot_configure_would_change': {
                name: {'present': present, 'after_configure': target}
                for name, (present, target) in configure_differences(values, servo_id == gripper_id).items()},
        }
    return servos


def decode_registers(raw):
    """{address: byte} -> {register name: value} for every register fully present in `raw`."""
    values = {}
    for name, (address, size) in STS3215_REGISTERS.items():
        if all(address + i in raw for i in range(size)):
            value = int.from_bytes(bytes(raw[address + i] for i in range(size)), 'little')
            values[name] = decode_sign_magnitude(value, STS3215_SIGN_BITS[name]) if name in STS3215_SIGN_BITS else value
    return values


def configure_differences(values, gripper):
    """{register: (present, after LeRobot configure())} for registers configure() would change."""
    targets = dict(LEROBOT_CONFIGURE, **(LEROBOT_CONFIGURE_GRIPPER if gripper else {}))
    if 'Phase' in values:
        targets['Phase'] = values['Phase'] & ~0x10
    return {name: (values[name], target) for name, target in targets.items()
            if name in values and values[name] != target}
