"""scripts/actuator_log.py motion path against simulated STS servos (no hardware)."""
import importlib.util
import io
import json
from pathlib import Path
import struct
import time

import numpy as np
import pytest
from le_snail.hardware import xbus_frame
from le_snail.servo_bus import PING, READ, SYNC_READ, SYNC_WRITE, status_frame

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'actuator_log.py'
IDS = [1, 2, 3]


def load_script():
    spec = importlib.util.spec_from_file_location('actuator_log', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimServos:
    """Register memories that answer PING/READ/SYNC_READ and apply SYNC_WRITE.

    With torque on, a servo reaches its goal instantly. `after_read` runs after each request so a test
    can change the world mid-run; `keep_goal=False` mimics firmware that ignores goals while torque is off.
    """
    def __init__(self, keep_goal=True):
        self.memory = {}
        for servo_id in IDS:
            m = [0] * 87
            m[3:5] = (0x09, 0x03)
            m[9:13] = (0x00, 0x04, 0x00, 0x0C)        # limits 1024..3072
            m[18], m[21], m[22], m[33], m[85] = 0x0C, 16, 32, 0, 254
            m[56:58] = (0x00, 0x08)                    # 2048
            m[62], m[63] = 121, 30
            self.memory[servo_id] = m
        self.keep_goal, self.after_read, self.writes, self.pending = keep_goal, None, [], b''

    in_waiting = property(lambda self: len(self.pending))

    def reset_input_buffer(self):
        self.pending = b''

    def flush(self):
        pass

    def close(self):
        pass

    def read(self, n=1):
        data, self.pending = self.pending[:n], self.pending[n:]
        return data

    def write(self, request):
        servo_id, instruction, params = request[2], request[4], request[5:-1]
        for m in self.memory.values():
            if m[40]:
                m[56:58] = m[42:44]
        if instruction == PING and servo_id in self.memory:
            self.pending += status_frame(servo_id, 0)
        elif instruction == READ and servo_id in self.memory:
            address, length = params
            self.pending += status_frame(servo_id, 0, bytes(self.memory[servo_id][address:address + length]))
        elif instruction == SYNC_READ:
            address, length = params[:2]
            for i in params[2:]:
                self.pending += status_frame(i, 0, bytes(self.memory[i][address:address + length]))
            if self.after_read:
                self.after_read(self)
        elif instruction == SYNC_WRITE:
            address, length, rest = params[0], params[1], params[2:]
            for k in range(0, len(rest), length + 1):
                m, data = self.memory[rest[k]], list(rest[k + 1:k + 1 + length])
                self.writes.append((address, rest[k], data))
                if address == 42 and not (self.keep_goal or m[40]):
                    continue
                m[address:address + length] = data


@pytest.fixture
def run(tmp_path, monkeypatch):
    motion = tmp_path / 'steps.npz'
    offsets = np.zeros((40, len(IDS)), dtype=int)
    offsets[10:, 1] = 100
    np.savez(motion, offsets_ticks=offsets, rate_hz=100.0, ids=np.array(IDS), name='steps')

    def go(sim, confirm=None, imu=None):
        module = load_script()
        monkeypatch.setattr(module, 'open_bus_port', lambda path: sim)
        monkeypatch.setattr(module, 'open_imu_port', lambda path, baudrate: imu)
        monkeypatch.setattr(module, 'stdin_has_line', lambda: True)
        monkeypatch.setattr('sys.stdin', io.StringIO('\n'))
        monkeypatch.setattr('builtins.input', lambda prompt='': (confirm and confirm(sim), 'MOVE')[1])
        monkeypatch.setattr('sys.argv', ['actuator_log.py', 'SIM', '--ids', *map(str, IDS), '--motion', str(motion),
                                         '--allow-motion', '--ramp-s', '0.1', '--out', str(tmp_path / 'out')]
                            + (['--imu', 'FAKE'] if imu else []))
        module.main()
        meta_path = next((tmp_path / 'out').glob('*.json'))
        return dict(json.loads(meta_path.read_text()), log=np.load(meta_path.with_suffix('.npz')))
    return go


def torque_writes(sim):
    return [data[0] for address, _, data in sim.writes if address == 40]


def test_normal_run_ends_with_torque_released(run):
    sim = SimServos()
    meta = run(sim)
    assert meta['finished'] and meta['abort'] is None
    assert [w['register'] for w in meta['writes']] == ['Goal_Position', 'Acceleration', 'Torque_Enable', 'Torque_Enable']
    assert all(m[40] == 0 and m[41] == 254 for m in sim.memory.values())
    assert sim.memory[2][56:58] == [0x64, 0x08]  # 2048 + 100


def test_envelope_violation_releases_torque_at_once(run):
    sim = SimServos()
    sim.after_read = lambda s: s.memory[3].__setitem__(63, 60 if s.memory[3][40] else 30)
    meta = run(sim)
    assert 'temperature' in meta['abort'] and not meta['finished']
    assert torque_writes(sim)[-1] == 0 and all(m[40] == 0 for m in sim.memory.values())


def test_arm_moved_between_plan_and_confirmation_refuses_without_writing(run):
    sim = SimServos()
    with pytest.raises(SystemExit, match='moved'):
        run(sim, confirm=lambda s: s.memory[2].__setitem__(57, 0x09))  # 2048 -> 2304
    assert sim.writes == []


def test_goal_that_does_not_read_back_refuses_before_torque_enable(run):
    sim = SimServos(keep_goal=False)
    with pytest.raises(SystemExit, match='Goal_Position read back'):
        run(sim)
    assert torque_writes(sim) == []


class FakeImu:
    """Streams one MTData2 frame (counter, sample time, quaternion) per read, like the Xsens at ~1 kHz."""
    def __init__(self):
        self.counter = 0

    in_waiting = 0

    def reset_input_buffer(self):
        pass

    def close(self):
        pass

    def read(self, n=1):
        time.sleep(0.001)
        self.counter += 1
        items = [(0x1020, struct.pack('>H', self.counter)), (0x1060, struct.pack('>I', 10 * self.counter)),
                 (0x2010, struct.pack('>4f', 1., 0., 0., 0.))]
        return xbus_frame(0x36, b''.join(struct.pack('>HB', i, len(d)) + d for i, d in items))


def test_imu_is_logged_on_the_servo_clock_alongside_the_run(run):
    meta = run(SimServos(), imu=FakeImu())
    log = meta['log']
    assert meta['imu']['samples'] > 10 and meta['imu']['errors'] == []
    assert log['imu_fields'].shape == (meta['imu']['samples'], 12)
    assert np.all(np.diff(log['imu_fields'][:, 0]) == 1) and log['imu_fields'][0, 2] == 1.0
    assert np.isnan(log['imu_fields'][0, 6])  # no gyro in this stream
    assert log['sent_ns'][0] <= log['imu_received_ns'][-1] and log['imu_received_ns'][0] <= log['sent_ns'][-1]
