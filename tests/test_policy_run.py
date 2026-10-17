"""scripts/policy_run.py against simulated servos, a fake IMU and a scripted stand-in policy (no hardware)."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pytest
import torch

from le_snail.actuator_id import TICKS_PER_RAD
from le_snail.contract import PolicyContract

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'policy_run.py'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_actuator_log import SimServos  # noqa: E402

# Ids 1-3 drive pan, lift and elbow; wrist_flex, wrist_roll and gripper stay virtual.
JOINTS = {'shoulder_pan': {'id': 1, 'sign': 1, 'offset_ticks': 2500.0},
          'shoulder_lift': {'id': 2, 'sign': 1, 'offset_ticks': 2400.0},
          'elbow_flex': {'id': 3, 'sign': 1, 'offset_ticks': 1600.0}}


class ConstantPolicy(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.register_buffer('action', torch.tensor([action], dtype=torch.float32))

    def forward(self, obs):
        return self.action.expand(obs.shape[0], 6)


def make_bundle(path, action, physics_validated=False):
    path.mkdir()
    shutil.copy(ROOT / 'configs' / 'policy-50hz-001.json', path / 'policy.json')
    torch.jit.save(torch.jit.script(ConstantPolicy(action)), str(path / 'policy.pt'))
    (path / 'sim.json').write_text('{}')
    digest = lambda name: hashlib.sha256((path / name).read_bytes()).hexdigest()
    (path / 'manifest.json').write_text(json.dumps({
        'contract_fingerprint': PolicyContract.load(path / 'policy.json').fingerprint,
        'policy_sha256': digest('policy.pt'), 'physics_sha256': digest('sim.json'),
        'observation_size': 24, 'action_size': 6, 'stage': 'robust', 'physics_validated': physics_validated}))
    return path


class FakeImu:
    """ImuSerial stand-in: upright gravity, small yaw rate; stale_after_s makes later samples old."""
    def __init__(self, stale_after_s=None):
        self.start, self.stale_after_s, self.seq, self.closed = time.monotonic(), stale_after_s, 0, False

    def read(self, max_age_s):
        now = time.monotonic()
        self.seq += 1
        at = now if self.stale_after_s is None or now - self.start < self.stale_after_s else now - 1.0
        if now - at > max_age_s:
            raise RuntimeError('Stale IMU sample')
        return {'gravity': np.array([0., 0., -1.]), 'gyro': np.array([0., 0., 0.1]), 'sample_at': at,
                'received_at': at, 'seq': self.seq}

    def close(self):
        self.closed = True


@pytest.fixture
def run(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('policy_run', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    joint_map = tmp_path / 'map.json'
    joint_map.write_text(json.dumps({'joints': JOINTS}))
    hardware = tmp_path / 'hw.json'
    hardware.write_text(json.dumps({'servo_port': 'SIM', 'imu_port': 'FAKE', 'imu_baudrate': 115200,
                                    'joint_map': str(joint_map), 'imu_rotation_base_from_sensor': np.eye(3).tolist()}))

    def go(sim, bundle, mode='live', extra=(), imu=None):
        imu = imu or FakeImu()
        monkeypatch.setattr(module, 'open_bus_port', lambda path: sim)
        monkeypatch.setattr(module, 'open_imu', lambda hardware, rotation: imu)
        monkeypatch.setattr(module, 'stdin_has_line', lambda: True)
        monkeypatch.setattr('sys.stdin', io.StringIO('\n'))
        monkeypatch.setattr('builtins.input', lambda prompt='': 'RUN POLICY')
        monkeypatch.setattr('sys.argv', ['policy_run.py', '--hardware', str(hardware), '--bundle', str(bundle),
                                         '--mode', mode, '--seconds', '0.3', '--ramp-s', '0.1',
                                         '--out', str(tmp_path / 'out'), *extra])
        module.main()
        meta_path = next((tmp_path / 'out').glob('*.json'))
        return dict(json.loads(meta_path.read_text()), log=dict(np.load(meta_path.with_suffix('.npz'))), imu=imu)
    return go


def home_ticks():
    home = PolicyContract.load(ROOT / 'configs' / 'policy-50hz-001.json').home_rad
    return [round(JOINTS[n]['offset_ticks'] + home[i] * TICKS_PER_RAD) for i, n in enumerate(JOINTS)]


def test_shadow_mode_never_writes_and_logs_one_step_per_period(run, tmp_path):
    sim = SimServos()
    meta = run(sim, make_bundle(tmp_path / 'b', [0.0] * 6), mode='shadow')
    assert sim.writes == [] and meta['abort'] is None
    log = meta['log']
    assert 12 <= len(log['step_time_s']) <= 17                      # 0.3 s at 50 Hz
    assert np.allclose(log['observation'][:, 12:15], [0., 0., -1.])
    # Previous target = measured pose each step, so the applied-action input is (q - home) / scale.
    contract = PolicyContract.load(ROOT / 'configs' / 'policy-50hz-001.json')
    assert np.allclose(log['observation'][:, 18:24], log['observation'][:, 0:6] / contract.action_scale_rad, atol=1e-5)
    assert (tmp_path / 'out').glob('*.jsonl')


def test_live_run_ramps_to_home_then_follows_the_policy_and_releases_torque(run, tmp_path, monkeypatch):
    from le_snail import contract as contract_module
    updates = []
    original = contract_module.VelocityEstimator.update
    monkeypatch.setattr(contract_module.VelocityEstimator, 'update', lambda self, q, t: updates.append(t) or original(self, q, t))
    sim = SimServos()
    meta = run(sim, make_bundle(tmp_path / 'b', [0.0] * 6), extra=['--allow-unvalidated-physics'])
    assert meta['abort'] is None and meta['finished']
    assert [w['register'] for w in meta['writes']] == ['Goal_Position', 'Acceleration', 'Torque_Enable', 'Torque_Enable']
    assert all(m[40] == 0 for m in sim.memory.values())
    log = meta['log']
    assert log['command_goal_ticks'][-1].tolist() == home_ticks()   # zero action keeps home
    assert len(updates) == len(log['step_time_s'])                  # velocity filter fed once per policy step
    assert np.allclose(log['observation'][0, 6:12], 0.0)            # sim reset: filtered velocity 0
    assert np.allclose(log['observation'][0, 18:24], 0.0)           # sim reset: previous target = home
    assert meta['virtual_joints'] == ['wrist_flex', 'wrist_roll', 'gripper']


def test_clipped_goal_is_what_feeds_back_as_the_previous_target(run, tmp_path):
    sim = SimServos()   # limits 1024..3072, margin 100 -> pan window 1124..2972
    meta = run(sim, make_bundle(tmp_path / 'b', [1.0] + [0.0] * 5), extra=['--allow-unvalidated-physics', '--seconds', '3'])
    log = meta['log']
    assert log['command_goal_ticks'][:, 0].max() == 2972 and log['clipped'][:, 0].any()
    contract = PolicyContract.load(ROOT / 'configs' / 'policy-50hz-001.json')
    applied_pan = log['observation'][-1, 18] * contract.action_scale_rad[0] + contract.home_rad[0]
    assert applied_pan == pytest.approx((2972 - 2500) / TICKS_PER_RAD, abs=1e-4)


def test_stale_imu_aborts_and_releases_torque(run, tmp_path):
    sim = SimServos()
    meta = run(sim, make_bundle(tmp_path / 'b', [0.0] * 6), extra=['--allow-unvalidated-physics'],
               imu=FakeImu(stale_after_s=0.15))
    assert 'IMU' in meta['abort'] and not meta['finished']
    assert all(m[40] == 0 for m in sim.memory.values()) and meta['imu'].closed


def test_live_refuses_unvalidated_physics_without_the_override(run, tmp_path):
    sim = SimServos()
    with pytest.raises(SystemExit, match='physics'):
        run(sim, make_bundle(tmp_path / 'b', [0.0] * 6))
    assert sim.writes == []


def test_refuses_when_servo_homing_offsets_differ_from_the_joint_map(run, tmp_path):
    sim = SimServos()   # Homing_Offset 0 on every servo
    map_path = Path(json.loads((tmp_path / 'hw.json').read_text())['joint_map'])
    map_path.write_text(json.dumps({'joints': JOINTS, 'valid_while_eeprom_homing_offset': {'1': 0, '2': 0, '3': 57}}))
    with pytest.raises(SystemExit, match='Homing_Offset'):
        run(sim, make_bundle(tmp_path / 'b', [0.0] * 6), mode='shadow')


def test_target_at_a_contract_limit_survives_tick_rounding(run, tmp_path):
    # lift lower limit -1.65 rad -> 2400 - 1075.6 ticks rounds outward to 1324 (-1.6505 rad) unless clamped
    sim = SimServos()
    meta = run(sim, make_bundle(tmp_path / 'b', [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]), extra=['--allow-unvalidated-physics', '--seconds', '2'])
    assert meta['abort'] is None
    assert meta['log']['target_rad'][:, 1].min() >= -1.65


class LossyTorqueOff(SimServos):
    """Ignores the first Torque_Enable = 0 broadcast, like a lost packet."""
    dropped = False

    def write(self, request):
        if request[4] == 0x83 and request[5] == 40 and request[8] == 0 and not self.dropped:
            self.dropped = True
            return
        super().write(request)


def test_torque_release_is_read_back_and_retried(run, tmp_path):
    sim = LossyTorqueOff()
    meta = run(sim, make_bundle(tmp_path / 'b', [0.0] * 6), extra=['--allow-unvalidated-physics'])
    assert sim.dropped and all(m[40] == 0 for m in sim.memory.values())
    assert meta['torque_release'] == 'confirmed'


def test_improper_imu_rotation_is_refused_before_any_write(run, tmp_path):
    hw = tmp_path / 'hw.json'
    config = json.loads(hw.read_text())
    config['imu_rotation_base_from_sensor'] = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    hw.write_text(json.dumps(config))
    sim = SimServos()
    with pytest.raises(SystemExit, match='rotation'):
        run(sim, make_bundle(tmp_path / 'b', [0.0] * 6), extra=['--allow-unvalidated-physics'])
    assert sim.writes == []


def test_lost_servo_feedback_aborts_and_still_saves_the_log(run, tmp_path):
    sim = SimServos()
    calls = {'n': 0}

    def drop_after_ramp(s):
        calls['n'] += 1
        if calls['n'] > 200 and s.memory[1][40]:
            s.pending = b''
    sim.after_read = drop_after_ramp
    meta = run(sim, make_bundle(tmp_path / 'b', [0.0] * 6), extra=['--allow-unvalidated-physics', '--seconds', '3'])
    assert meta['abort'] and not meta['finished']
    assert all(m[40] == 0 for m in sim.memory.values())


def test_ramp_to_home_is_speed_limited_whatever_ramp_s_says(run, tmp_path):
    sim = SimServos()
    meta = run(sim, make_bundle(tmp_path / 'b', [0.0] * 6), extra=['--allow-unvalidated-physics', '--ramp-s', '0'])
    goals = meta['log']['command_goal_ticks']
    steps = np.abs(np.diff(np.vstack([[2048, 2048, 2048], goals]), axis=0)).max(axis=1)
    assert steps.max() <= 0.5 * TICKS_PER_RAD / 50 + 1           # 0.5 rad/s at 50 Hz, plus tick rounding


def test_joint_map_soft_window_tightens_the_eeprom_window(run, tmp_path):
    sim = SimServos()   # EEPROM window 1124..2972 after the margin
    map_path = Path(json.loads((tmp_path / 'hw.json').read_text())['joint_map'])
    joints = json.loads(json.dumps(JOINTS))
    joints['shoulder_pan']['window_ticks'] = [1000, 2700]      # a hard stop the servo EEPROM does not know about
    map_path.write_text(json.dumps({'joints': joints}))
    meta = run(sim, make_bundle(tmp_path / 'b', [1.0] + [0.0] * 5), extra=['--allow-unvalidated-physics', '--seconds', '1.5'])
    assert meta['window_ticks'][0][0] == 1124 and meta['window_ticks'][1][0] == 2700
    assert meta['log']['command_goal_ticks'][:, 0].max() == 2700


def test_a_slow_first_step_anchors_the_schedule_instead_of_aborting(run, tmp_path, monkeypatch):
    from le_snail import contract as contract_module
    original, calls = contract_module.PolicyContract.observation, {'n': 0}

    def slow_first(self, *a):
        calls['n'] += 1
        if calls['n'] == 1:
            time.sleep(0.04)        # e.g. first-call warm-up on the target computer
        return original(self, *a)
    monkeypatch.setattr(contract_module.PolicyContract, 'observation', slow_first)
    meta = run(SimServos(), make_bundle(tmp_path / 'b', [0.0] * 6), extra=['--allow-unvalidated-physics'])
    assert meta['abort'] is None and meta['policy_steps'] >= 12
    assert meta['log']['lateness_s'][0] > 0.03 and np.all(meta['log']['lateness_s'][1:] < 0.01)


def test_cpu_affinity_from_the_hardware_config_is_applied(run, tmp_path, monkeypatch):
    import os
    hw = tmp_path / 'hw.json'
    config = json.loads(hw.read_text())
    config['cpu_affinity'] = [0]
    hw.write_text(json.dumps(config))
    applied = []
    monkeypatch.setattr(os, 'sched_setaffinity', lambda pid, cpus: applied.append((pid, set(cpus))))
    meta = run(SimServos(), make_bundle(tmp_path / 'b', [0.0] * 6), mode='shadow')
    assert applied == [(0, {0})] and meta['cpu_affinity'] == [0]


class SideImu(FakeImu):
    """Base lying on its side (sim 'left' start gravity)."""
    def read(self, max_age_s):
        sample = super().read(max_age_s)
        return dict(sample, gravity=np.array([0.039, -0.994, 0.101]))


def test_require_upright_refuses_to_start_the_policy_on_a_fallen_robot(run, tmp_path):
    sim = SimServos()
    meta = run(sim, make_bundle(tmp_path / 'b', [0.0] * 6), extra=['--allow-unvalidated-physics', '--require-upright'],
               imu=SideImu())
    assert 'not upright' in meta['abort'] and meta['policy_steps'] == 0
    assert all(m[40] == 0 for m in sim.memory.values())
    assert len(meta['log']['ramp_gravity']) > 0            # the IMU is logged during the ramp and hold


def test_ramp_imu_is_logged_on_normal_runs_too(run, tmp_path):
    meta = run(SimServos(), make_bundle(tmp_path / 'b', [0.0] * 6), extra=['--allow-unvalidated-physics'])
    assert meta['abort'] is None and len(meta['log']['ramp_gravity']) == len(meta['log']['ramp_time_s']) > 5
