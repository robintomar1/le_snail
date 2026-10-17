"""Phase-locked hybrid gait (#15): a clock-free teacher for cloning hybrid-001's crawl into a policy."""
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from le_snail.gait_teacher import load_phase_gait  # noqa: E402
from le_snail.hybrid import load_controller  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/controllers/hybrid-001.json'
CONTRACT = ROOT / 'configs/policy-50hz-001.json'
C = json.loads(CONTRACT.read_text())
HOME, SCALE = torch.tensor(C['home_rad']), torch.tensor(C['action_scale_rad'])
LOWER, UPPER = torch.tensor(C['lower_rad']), torch.tensor(C['upper_rad'])
STEP = torch.tensor(C['max_target_speed_rad_s']) / C['policy_hz']


def advance(target, action):
    """The environment's action path: clip, scale about home, joint limits, then the slew."""
    goal = torch.clamp(HOME + action.clamp(-1., 1.) * SCALE, LOWER, UPPER)
    return target + torch.clamp(goal - target, -STEP, STEP)


def observation(target):
    obs = torch.zeros((len(target), 24))
    obs[:, 14] = -1.
    obs[:, 18:24] = (target - HOME) / SCALE
    return obs


def open_loop(steps):
    hybrid = load_controller(CONFIG, CONTRACT)
    t = torch.arange(steps) / C['policy_hz']
    target, out = HOME[None].clone(), []
    for a in hybrid.gait_action(t):
        target = advance(target, a[None])
        out.append(target[0])
    return torch.stack(out)


def test_closed_loop_reproduces_the_open_loop_gait_once_on_the_cycle():
    reference = open_loop(800)
    teacher = load_phase_gait(CONFIG, CONTRACT)
    target = reference[399][None].clone()
    played = []
    for _ in range(300):
        target = advance(target, teacher(observation(target)))
        played.append(target[0])
    error = (torch.stack(played) - reference[400:700]).abs().max()
    assert error < 0.02   # rad, after 6 s locked on the cycle


def test_joins_the_cycle_from_home_and_runs_forward():
    teacher = load_phase_gait(CONFIG, CONTRACT)
    target, phases = HOME[None].clone(), []
    for _ in range(400):
        target = advance(target, teacher(observation(target)))
        phases.append(float(teacher.current_phase()[0]))
    normalized = (target - HOME) / SCALE
    assert (normalized - teacher.cycle_target).square().sum(dim=1).min().sqrt() < 0.05
    steps = torch.tensor(phases[200:]).diff() % 1.
    expected = 1. / (C['policy_hz'] * teacher.period)
    assert steps.median() == pytest.approx(expected, rel=0.05)   # advances one policy step per step
    assert (steps < 0.2).all()                                     # never jumps around the cycle


def test_torchscript_and_per_env_reset():
    teacher = torch.jit.script(load_phase_gait(CONFIG, CONTRACT))
    obs = observation(torch.stack((HOME, HOME + 0.2)))
    first = teacher(obs)
    assert first.shape == (2, 6) and torch.isfinite(first).all() and first.abs().max() <= 1.
    teacher.reset_envs(torch.tensor([1]))
    teacher(obs)
    teacher.reset()
    assert torch.equal(teacher(obs), first)
