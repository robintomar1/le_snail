import torch
from le_snail.speed import PhysicsSpeedDiagnostics


def test_actual_randomized_caps_counts_and_reset_streak():
    monitor = PhysicsSpeedDiagnostics(2, 2, .005, 'cpu')
    velocity = torch.tensor([[4.1, 0.], [3.1, 0.]])
    limits = torch.tensor([[4., 4.], [3., 3.]])
    pd = torch.tensor([[-1., 0.], [1., 0.]])
    for _ in range(8):
        monitor.update(velocity, limits, pd)
    monitor.reset_envs([1])
    for _ in range(3):
        monitor.update(velocity, limits, pd)
    report = monitor.report()
    assert report['environment_time_samples'] == 22
    assert report['joint_time_samples'] == 44
    assert report['environment_over_limit_samples'] == 22
    assert report['joint_over_limit_samples'] == 22
    assert report['driven_proxy_over_limit_joint_samples'] == 11
    assert report['longest_driven_proxy_over_limit_s'] == .04
    assert report['flag_over_one_percent_time']
    assert not report['flag_sustained_driven_proxy']
    for _ in range(7):
        monitor.update(velocity, limits, pd)
    assert monitor.report()['flag_sustained_driven_proxy']


def test_empty_and_tolerance():
    monitor = PhysicsSpeedDiagnostics(1, 1, .005, 'cpu')
    assert monitor.report()['environment_time_samples'] == 0
    monitor.update(torch.tensor([[4.0005]]), torch.tensor([[4.]]), torch.ones(1, 1))
    assert monitor.report()['environment_over_limit_samples'] == 0
