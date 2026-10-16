"""Hybrid scripted controller (#14): gait formula, fall detection, get-up sequencing, retries, TorchScript."""
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from le_snail.hybrid import ABORT, CONFIRM, GAIT, GETUP, SETTLE, HybridController, load_controller  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/controllers/hybrid-001.json'
CONTRACT = ROOT / 'configs/policy-50hz-001.json'


def controller():
    return load_controller(CONFIG, CONTRACT)


def observation(n=1, gravity=(0., 0., -1.), applied_rad=None, gyro=(0., 0., 0.)):
    contract = json.loads(CONTRACT.read_text())
    home, scale = torch.tensor(contract['home_rad']), torch.tensor(contract['action_scale_rad'])
    obs = torch.zeros((n, 24))
    obs[:, 12:15] = torch.tensor(gravity)
    obs[:, 15:18] = torch.tensor(gyro)
    if applied_rad is not None:
        obs[:, 18:24] = (torch.as_tensor(applied_rad, dtype=torch.float32) - home) / scale
    return obs


def left_lying(applied_rad=None):
    return observation(gravity=(0.04, -0.994, 0.1), applied_rad=applied_rad)


def test_gait_matches_the_study_formula():
    c = controller()
    c.reset()
    cfg = json.loads(CONFIG.read_text())['gait']
    obs = observation()
    for step in range(120):
        action = c(obs)[0]
        t = step * c.dt
        ramp = min(1., t / cfg['ramp_s'])
        for j in range(6):
            x = 2 * math.pi * t / cfg['T'] * cfg['m'][j] + cfg['psi'][j]
            w = math.tanh(cfg['k'] * math.sin(x)) / math.tanh(cfg['k'])
            assert action[j].item() == pytest.approx(max(-1., min(1., cfg['c'][j] + ramp * cfg['A'][j] * w)), abs=1e-5)


def test_upright_start_gaits_and_side_start_settles_then_picks_the_side():
    c = controller()
    c.reset()
    c(torch.cat((observation(), left_lying())))
    assert c.mode.tolist() == [GAIT, SETTLE]
    c.reset()
    home = json.loads(CONTRACT.read_text())['home_rad']
    for _ in range(round(1. / c.dt) + 1):
        c(left_lying(home))
    assert c.mode.item() == GETUP and c.side.item() == 0
    c.reset()
    right = observation(gravity=(0.01, 0.994, 0.1), applied_rad=home)
    for _ in range(round(1. / c.dt) + 1):
        c(right)
    assert c.mode.item() == GETUP and c.side.item() == 1


def test_roll_fall_interrupts_the_gait_after_the_hold_time():
    c = controller()
    c.reset()
    c(observation())
    tilted = observation(gravity=(0., -math.sin(math.radians(70)), -math.cos(math.radians(70))))
    for _ in range(round(.2 / c.dt) - 1):
        c(tilted)
        assert c.mode.item() == GAIT
    c(tilted)
    assert c.mode.item() == SETTLE


def test_pitch_rocking_does_not_count_as_a_fall():
    c = controller()
    c.reset()
    pitched = observation(gravity=(-math.sin(math.radians(56)), 0., -math.cos(math.radians(56))))
    for _ in range(200):
        c(pitched)
    assert c.mode.item() == GAIT


def test_keys_advance_only_after_the_applied_target_holds_them():
    c = controller()
    c.reset()
    home = json.loads(CONTRACT.read_text())['home_rad']
    for _ in range(round(1. / c.dt) + 1):
        action = c(left_lying(home))
    keys = torch.tensor(json.loads(CONFIG.read_text())['getup']['keys_rad']['left'][0])
    assert c.key.item() == 0
    for _ in range(20):                       # target still at home: key 0 is not reached
        action = c(left_lying(home))
    assert c.key.item() == 0
    contract = json.loads(CONTRACT.read_text())
    home_t, scale, lower, upper = (torch.tensor(contract[k]) for k in ('home_rad', 'action_scale_rad', 'lower_rad', 'upper_rad'))
    for key in range(3):
        # The applied target reaches the key after the action clamp and the joint limits (as in the env).
        reached = torch.clamp(home_t + torch.clamp((keys[key] - home_t) / scale, -1., 1.) * scale, lower, upper)
        for _ in range(c.advance_steps):
            assert c.key.item() == key
            action = c(left_lying(reached))
        assert c.key.item() == key + 1
    for _ in range(c.final_steps):
        action = c(left_lying(home))
    assert c.mode.item() == CONFIRM
    assert torch.equal(action, torch.zeros_like(action))


def test_confirm_resumes_gait_when_upright_and_retries_then_aborts_when_still_fallen():
    c = controller()
    c.reset()
    c.mode.fill_(CONFIRM)
    for _ in range(c.upright_steps):
        c(observation())
    assert c.mode.item() == GAIT and c.gait_time.item() == pytest.approx(c.dt)
    c.reset()
    c(left_lying())
    for attempt in range(c.max_attempts):
        c.mode.fill_(CONFIRM)
        c.timer.zero_()
        for _ in range(c.confirm_steps):
            c(left_lying())
        assert c.attempts.item() == attempt + 1
    assert c.mode.item() == ABORT
    assert torch.equal(c(left_lying()), torch.zeros((1, 6)))


def test_torchscript_round_trip_and_batch_resize(tmp_path):
    scripted = torch.jit.script(controller())
    scripted.save(str(tmp_path / 'policy.pt'))
    loaded = torch.jit.load(str(tmp_path / 'policy.pt'))
    loaded.reset()
    eager = controller()
    eager.reset()
    obs = torch.cat((observation(), left_lying()))
    for _ in range(80):
        assert torch.allclose(loaded(obs), eager(obs))
    assert loaded(observation()).shape == (1, 6)
    assert isinstance(eager, HybridController)
