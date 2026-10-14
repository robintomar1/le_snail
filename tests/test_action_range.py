"""Full-range policy compatibility and pre-clamp diagnostic behavior."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from le_snail.contract import PolicyContract
from le_snail.action_stats import ActionDiagnostics


ROOT = Path(__file__).resolve().parents[1]


def test_wide_contract_changes_only_scale_and_fingerprint():
    current = PolicyContract.load(ROOT / 'configs/policy.json')
    wide = PolicyContract.load(ROOT / 'configs/policy-wide-001.json')
    before, after = asdict(current), asdict(wide)
    before.pop('action_scale_rad')
    after.pop('action_scale_rad')
    assert before == after
    assert current.fingerprint != wide.fingerprint
    assert np.allclose(wide.position_target(np.zeros(6), wide.home_rad), wide.home_rad)
    for sign, limits in ((-1., wide.lower_rad), (1., wide.upper_rad)):
        target = np.array(wide.home_rad)
        for _ in range(60):
            previous = target.copy()
            target = wide.position_target(np.full(6, sign), target)
            assert np.max(np.abs(target - previous)) <= .05 + 1e-12
        assert np.allclose(target, limits)


def test_wide_portable_tensor_mapping_and_observation_parity():
    contract = PolicyContract.load(ROOT / 'configs/policy-wide-001.json')
    rng = np.random.default_rng(841)
    home, scale, lower, upper, slew = map(torch.tensor, (
        contract.home_rad, contract.action_scale_rad, contract.lower_rad,
        contract.upper_rad, np.asarray(contract.max_target_speed_rad_s) * contract.dt))
    for _ in range(50):
        raw = rng.normal(0., 2., 6)
        previous = rng.uniform(contract.lower_rad, contract.upper_rad)
        tensor_previous = torch.tensor(previous)
        desired = torch.clamp(home + torch.tensor(raw).clamp(-1., 1.) * scale, lower, upper)
        applied = tensor_previous + torch.clamp(desired - tensor_previous, -slew, slew)
        assert np.allclose(applied.numpy(), contract.position_target(raw, previous), atol=1e-6)
        observation = contract.observation(contract.home_rad, np.zeros(6), [0, 0, -1], [0, 0, 0], applied.numpy())
        assert observation.shape == (24,)
        assert np.allclose(observation[-6:], ((applied - home) / scale).numpy(), atol=1e-6)


def test_action_counts_preserve_inputs_and_freeze_initial_window():
    monitor = ActionDiagnostics(2, ['a', 'b'], 'cpu', initial_steps=1)
    raw = torch.tensor([[2., -.5], [-1., -2.]])
    unclipped = torch.tensor([[1., -.5], [-1., -1.]])
    desired = unclipped.clamp(-.75, .75)
    old = torch.zeros_like(raw)
    slew = torch.tensor([.5, .5])
    new = desired.clamp(-slew, slew)
    saved = [v.clone() for v in (raw, unclipped, desired, old, new, slew)]
    monitor.update(raw, unclipped, desired, old, new, slew, -.75, .75)
    first = monitor.report()
    assert first['all']['counts_per_joint'] == {
        'raw_out_of_range': [1, 1], 'joint_limit_clipped': [2, 1],
        'slew_limited': [2, 1], 'controller_at_slew_bound': [2, 2], 'desired_at_joint_bound': [2, 1]}
    assert first['all']['fractions_all_joints']['raw_out_of_range'] == .5
    zeros = torch.zeros_like(raw)
    monitor.update(zeros, zeros, zeros, zeros, zeros, slew, -.75, .75)
    final = monitor.report()
    assert final['initial'] == first['initial']
    assert final['all']['fractions_all_joints']['raw_out_of_range'] == .25
    assert final['peak_abs_raw_action_per_joint'] == [2., 2.]
    for value, snapshot in zip((raw, unclipped, desired, old, new, slew), saved):
        assert torch.equal(value, snapshot)
