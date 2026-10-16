import pytest
from le_snail.servo_bus import (
    DUMP_RANGES, PING, READ, STS3215_REGISTERS, ServoBus, StatusReader, configure_differences, decode_registers,
    decode_sign_magnitude, dump_registers, open_bus_port, status_frame, sts_packet, sync_read_packet,
)


class FakePort:
    """Half-duplex bus stand-in: each written request is answered from `replies(request)`."""
    def __init__(self, replies, echo=False):
        self.replies, self.echo = replies, echo
        self.written, self.pending = [], b''

    @property
    def in_waiting(self):
        return len(self.pending)

    def reset_input_buffer(self):
        self.pending = b''

    def write(self, data):
        self.written.append(bytes(data))
        self.pending += (bytes(data) if self.echo else b'') + self.replies(bytes(data))

    def read(self, n=1):
        data, self.pending = self.pending[:n], self.pending[n:]
        return data


def test_ping_packet_matches_feetech_checksum():
    assert sts_packet(1, PING) == bytes.fromhex('FFFF010201FB')
    assert sts_packet(1, READ, bytes((56, 2))) == bytes.fromhex('FFFF0104023802BE')


def test_sync_read_packet_is_broadcast_with_address_length_and_ids():
    assert sync_read_packet([1, 2, 3], 56, 15) == sts_packet(0xFE, 0x82, bytes((56, 15, 1, 2, 3)))


def test_status_reader_resynchronizes_across_split_corrupt_and_garbage_input():
    good = status_frame(2, 0x00, bytes((0x10, 0x08)))
    corrupt = bytearray(status_frame(1, 0x00, b'\x01'))
    corrupt[-1] ^= 0xFF
    stream = b'\x00\xff\x13' + bytes(corrupt) + b'\xff' + good
    reader, frames = StatusReader(), []
    for i in range(0, len(stream), 3):
        frames += reader.feed(stream[i:i + 3])
    assert frames == [(2, 0x00, bytes((0x10, 0x08)))]


def test_read_returns_error_byte_and_parameters_and_strips_an_echo():
    port = FakePort(lambda request: status_frame(3, 0x01, bytes((0x34, 0x12))), echo=True)
    bus = ServoBus(port, echo=True)
    assert bus.read(3, 56, 2) == (0x01, bytes((0x34, 0x12)))
    assert port.written == [sts_packet(3, READ, bytes((56, 2)))]


def test_ping_reply_with_voltage_error_is_kept_although_it_equals_the_request():
    assert status_frame(1, 0x01) == sts_packet(1, PING)
    assert ServoBus(FakePort(lambda request: status_frame(1, 0x01))).ping(1) == (0x01, b'')


def test_missing_reply_is_reported_as_none_not_a_hang():
    bus = ServoBus(FakePort(lambda request: b''), timeout_s=0.002)
    assert bus.read(1, 56, 2) is None
    assert bus.ping(1) is None


def test_sync_read_collects_each_reply_and_marks_absent_ids_lost():
    def replies(request):
        return status_frame(1, 0, b'\x0a\x0b') + status_frame(3, 0x20, b'\x0c\x0d')
    bus = ServoBus(FakePort(replies), timeout_s=0.002)
    result = bus.sync_read([1, 2, 3], 56, 2)
    assert result.replies == {1: (0, b'\x0a\x0b'), 3: (0x20, b'\x0c\x0d')}
    assert result.lost == [2]
    assert result.sent_ns <= result.received_ns


def test_reply_of_the_wrong_length_counts_as_lost():
    bus = ServoBus(FakePort(lambda request: status_frame(1, 0, b'\x0a')), timeout_s=0.002)
    assert bus.sync_read([1], 56, 2).lost == [1]


def test_writes_are_refused_unless_the_bus_was_opened_for_writing():
    port = FakePort(lambda request: status_frame(1, 0))
    with pytest.raises(PermissionError):
        ServoBus(port).sync_write({1: 2048}, 42, 2)
    assert port.written == []
    ServoBus(port, allow_writes=True).sync_write({1: 2048, 2: 1}, 42, 2)
    assert port.written == [sts_packet(0xFE, 0x83, bytes((42, 2, 1, 0x00, 0x08, 2, 0x01, 0x00)))]


def test_port_lines_hold_the_esp32_in_reset_before_the_port_opens():
    events = []

    class FakeSerial:
        def __setattr__(self, name, value):
            events.append((name, value))
            object.__setattr__(self, name, value)

        def open(self):
            events.append(('open', None))

    port = open_bus_port('/dev/fake', serial_factory=FakeSerial)
    assert port.dtr is False and port.rts is True and port.baudrate == 1_000_000
    opened = events.index(('open', None))
    assert ('dtr', False) in events[:opened] and ('rts', True) in events[:opened]


def test_sign_magnitude_matches_lerobot_bits():
    assert decode_sign_magnitude(0x8000 | 100, 15) == -100
    assert decode_sign_magnitude(100, 15) == 100
    assert decode_sign_magnitude(0x400 | 7, 10) == -7


def test_register_dump_decodes_little_endian_and_signed_fields():
    raw = {a: 0 for a in range(87)}
    raw.update({3: 0x09, 4: 0x03, 21: 16, 22: 32, 41: 254, 56: 0x00, 57: 0x08, 58: 0x0A, 59: 0x80,
                60: 0x05, 61: 0x04, 62: 121, 63: 29, 85: 254})
    values = decode_registers(raw)
    assert values['Model_Number'] == 777
    assert values['Present_Position'] == 2048
    assert values['Present_Velocity'] == -10
    assert values['Present_Load'] == -5
    assert values['Present_Voltage'] == 121 and values['Present_Temperature'] == 29


def test_registers_missing_from_the_dump_are_left_out():
    values = decode_registers({a: 0 for a in range(71)})
    assert 'Present_Current' in values and 'Maximum_Acceleration' not in values


def test_configure_differences_lists_what_lerobot_connect_would_change():
    body = {'Return_Delay_Time': 0, 'Maximum_Acceleration': 254, 'Acceleration': 254, 'Phase': 0x10,
            'Operating_Mode': 0, 'P_Coefficient': 32, 'I_Coefficient': 0, 'D_Coefficient': 32}
    assert configure_differences(body, gripper=False) == {'Phase': (0x10, 0x00), 'P_Coefficient': (32, 16)}
    gripper = dict(body, Phase=0, P_Coefficient=16, Max_Torque_Limit=1000, Protection_Current=250, Overload_Torque=25)
    assert configure_differences(gripper, gripper=True) == {'Max_Torque_Limit': (1000, 500)}


def test_register_table_is_the_pinned_lerobot_sts_table():
    assert STS3215_REGISTERS['Present_Current'] == (69, 2)
    assert STS3215_REGISTERS['Maximum_Acceleration'] == (85, 1)
    assert STS3215_REGISTERS['Goal_Position'] == (42, 2)


class FakeServos:
    """Answers READ from per-servo register memories; addresses outside a memory get no reply."""
    def __init__(self, memories):
        self.memories = memories

    def __call__(self, request):
        servo_id, instruction = request[2], request[4]
        if instruction != READ or servo_id not in self.memories:
            return b''
        address, length = request[5], request[6]
        memory = self.memories[servo_id]
        if address + length > len(memory):
            return b''
        return status_frame(servo_id, 0, bytes(memory[address:address + length]))


def test_register_dump_reads_chunks_and_records_ranges_that_fail():
    memory = list(range(80))  # no factory area on this servo
    bus = ServoBus(FakePort(FakeServos({1: memory})), timeout_s=0.002)
    dump = dump_registers(bus, 1)
    assert dump['raw'] == {a: a for a in range(71)}
    assert dump['failed'] == [[80, 7]]
    assert dump['errors'] == {}
    assert dump_registers(bus, 2)['failed'] == [list(r) for r in DUMP_RANGES]
