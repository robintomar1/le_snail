"""Scripted-prefix handover curriculum (#6): config validation, masked diagnostics and masked PPO."""
import copy
import json
from pathlib import Path

import pytest
from le_snail.physics import load_physics

ROOT = Path(__file__).resolve().parents[1]
KEYS = [[0.1] * 6, [0.2] * 6, [0.3] * 6]


def handover_config():
    config = copy.deepcopy(json.loads((ROOT / 'configs/experiments/physics-v2-002.json').read_text()))
    config['handover'] = {'fraction_of_side_starts': .75, 'scripts': {
        'left': [{'id': 'p9', 'keyframes_rad': copy.deepcopy(KEYS), 'prefix_steps': [5, 10]}],
        'right': [{'id': 'p48', 'keyframes_rad': copy.deepcopy(KEYS), 'prefix_steps': [7]}]}}
    return config


def write(tmp_path, config):
    path = tmp_path / 'handover.json'
    path.write_text(json.dumps(config))
    return path


def test_valid_handover_config_loads(tmp_path):
    assert load_physics(write(tmp_path, handover_config()))['handover']['fraction_of_side_starts'] == .75
    config = handover_config()
    config['handover']['attempt_seconds'] = 3.
    assert load_physics(write(tmp_path, config))['handover']['attempt_seconds'] == 3.


@pytest.mark.parametrize('key', ['side_start_heading_reset_at_recovery', 'side_start_progress_neutral_until_recovery'])
def test_side_start_reward_flags_must_be_boolean(tmp_path, key):
    config = handover_config()
    config[key] = True
    assert load_physics(write(tmp_path, config))[key] is True
    for value in (1, 'true', None):
        config[key] = value
        with pytest.raises(ValueError):
            load_physics(write(tmp_path, config))


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf'), True, '3'])
def test_invalid_attempt_seconds_is_rejected(tmp_path, value):
    config = handover_config()
    config['handover']['attempt_seconds'] = value
    with pytest.raises(ValueError):
        load_physics(write(tmp_path, config))


def test_longest_driven_streak_records_its_location_and_context():
    torch = pytest.importorskip('torch')
    from le_snail.speed import PhysicsSpeedDiagnostics
    monitor = PhysicsSpeedDiagnostics(2, 2, .005, 'cpu')
    limits = torch.full((2, 2), 4.)
    context = torch.tensor([[0., .1], [2., .4]])
    driven = torch.tensor([[0., 0.], [5., 0.]])
    for _ in range(3):
        monitor.update(driven, limits, torch.ones(2, 2), context=context)
    monitor.update(torch.zeros(2, 2), limits, torch.ones(2, 2), context=context)
    report = monitor.report()
    assert report['longest_driven_proxy_over_limit_s'] == pytest.approx(.015)
    assert report['longest_driven_proxy_location'] == {'env': 1, 'joint_index': 0, 'physics_sample': 3, 'context': [2., pytest.approx(.4)]}
    assert PhysicsSpeedDiagnostics(1, 1, .005, 'cpu').report()['longest_driven_proxy_location'] is None


@pytest.mark.parametrize('change', ['zero_fraction', 'excess_fraction', 'missing_side', 'empty_side', 'short_keys',
                                    'nonfinite_key', 'official_prefix', 'float_prefix', 'empty_prefix', 'no_side_starts',
                                    'no_settled_starts'])
def test_invalid_handover_config_is_rejected(tmp_path, change):
    config = handover_config()
    handover = config['handover']
    left = handover['scripts']['left'][0]
    if change == 'zero_fraction':
        handover['fraction_of_side_starts'] = 0.
    elif change == 'excess_fraction':
        handover['fraction_of_side_starts'] = 1.5
    elif change == 'missing_side':
        del handover['scripts']['right']
    elif change == 'empty_side':
        handover['scripts']['right'] = []
    elif change == 'short_keys':
        left['keyframes_rad'] = KEYS[:2]
    elif change == 'nonfinite_key':
        left['keyframes_rad'][1][2] = float('nan')
    elif change == 'official_prefix':
        left['prefix_steps'] = [0, 5]
    elif change == 'float_prefix':
        left['prefix_steps'] = [5.]
    elif change == 'empty_prefix':
        left['prefix_steps'] = []
    elif change == 'no_side_starts':
        config['recovery_start_fraction'] = 0.
    elif change == 'no_settled_starts':
        for key in ('settled_starts', 'side_upright_progress_reward_cap'):
            del config[key]
    with pytest.raises(ValueError):
        load_physics(write(tmp_path, config))


def test_teacher_rows_are_excluded_from_speed_and_action_diagnostics():
    torch = pytest.importorskip('torch')
    from le_snail.action_stats import ActionDiagnostics
    from le_snail.speed import PhysicsSpeedDiagnostics
    speed = PhysicsSpeedDiagnostics(2, 1, .005, 'cpu')
    learner = torch.tensor([True, False])
    for _ in range(4):
        speed.update(torch.tensor([[5.], [9.]]), torch.full((2, 1), 4.), torch.ones(2, 1), mask=learner)
    report = speed.report()
    assert report['environment_time_samples'] == 4
    assert report['environment_over_limit_samples'] == 4
    assert report['peak_abs_velocity_rad_s'] == 5.
    actions = ActionDiagnostics(2, ['j'], 'cpu')
    raw = torch.tensor([[2.], [3.]])
    target = torch.zeros(2, 1)
    actions.update(raw, target, target, target, target, torch.ones(1), -torch.ones(1), torch.ones(1), mask=learner)
    window = actions.report()['all']
    assert window['samples_per_joint'] == 1
    assert window['counts_per_joint']['raw_out_of_range'] == [1]
    assert actions.report()['peak_abs_raw_action_per_joint'] == [2.]


def rollout(torch, alg, steps, envs, masks, seed=0):
    from tensordict import TensorDict
    generator = torch.Generator().manual_seed(seed)
    alg.init_storage('rl', envs, steps, TensorDict({'policy': torch.zeros(envs, 24)}, batch_size=[envs]), [6])
    for step in range(steps):
        obs = TensorDict({'policy': torch.randn(envs, 24, generator=generator)}, batch_size=[envs])
        alg.act(obs)
        rewards = torch.randn(envs, generator=generator)
        dones = (torch.rand(envs, generator=generator) < .1).long()
        extras = {} if masks is None else {'learner_mask': masks[step], 'next_observation_is_learner': torch.ones(envs)}
        alg.process_env_step(obs, rewards, dones, extras)
    last = TensorDict({'policy': torch.randn(envs, 24, generator=generator)}, batch_size=[envs])
    alg.compute_returns(last)


def make_algorithms(torch, masked_class):
    pytest.importorskip('rsl_rl')
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    obs = TensorDict({'policy': torch.zeros(1, 24)}, batch_size=[1])
    groups = {'policy': ['policy'], 'critic': ['policy']}
    settings = dict(init_noise_std=.5, actor_obs_normalization=True, critic_obs_normalization=True,
                    actor_hidden_dims=[32, 32], critic_hidden_dims=[32, 32], activation='elu')
    algorithm = dict(value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2, entropy_coef=.01,
                     num_learning_epochs=2, num_mini_batches=4, learning_rate=3e-4, schedule='adaptive',
                     gamma=.995, lam=.95, desired_kl=.01, max_grad_norm=1.)
    torch.manual_seed(1)
    stock = PPO(ActorCritic(obs, groups, 6, **settings), **algorithm)
    torch.manual_seed(1)
    masked = masked_class(ActorCritic(obs, groups, 6, **settings), **algorithm)
    masked.policy.load_state_dict(stock.policy.state_dict())
    return stock, masked


def test_masked_ppo_matches_stock_ppo_when_every_step_is_learner():
    torch = pytest.importorskip('torch')
    pytest.importorskip('rsl_rl')
    from le_snail.masked_ppo import MaskedPPO
    stock, masked = make_algorithms(torch, MaskedPPO)
    for alg in (stock, masked):
        torch.manual_seed(7)
        rollout(torch, alg, 16, 8, None)
    torch.testing.assert_close(masked.storage.advantages, stock.storage.advantages, rtol=1e-5, atol=1e-6)
    torch.manual_seed(11)
    stock_losses = stock.update()
    torch.manual_seed(11)
    masked_losses = masked.update()
    for name in ('value_function', 'surrogate', 'entropy'):
        assert masked_losses[name] == pytest.approx(stock_losses[name], rel=1e-4, abs=1e-6)
    assert masked_losses['learner_fraction'] == 1.
    for key, value in stock.policy.state_dict().items():
        torch.testing.assert_close(masked.policy.state_dict()[key], value, rtol=1e-4, atol=1e-6)


def test_teacher_transitions_do_not_influence_the_update():
    torch = pytest.importorskip('torch')
    pytest.importorskip('rsl_rl')
    from le_snail.masked_ppo import MaskedPPO
    steps, envs = 16, 8
    masks = torch.ones(steps, envs)
    masks[:3, :4] = 0.  # teacher prefix at the start of the rollout for half of the environments
    results = []
    for corrupt in (False, True):
        _, alg = make_algorithms(torch, MaskedPPO)
        torch.manual_seed(7)
        rollout(torch, alg, steps, envs, masks)
        teacher = alg.learner_mask == 0
        assert bool((alg.storage.advantages[teacher] == 0).all())
        learner = alg.storage.advantages[~teacher]
        assert float(learner.mean()) == pytest.approx(0., abs=1e-5)
        assert float(learner.std()) == pytest.approx(1., rel=1e-4)
        if corrupt:
            # Arbitrary teacher actions, log-probabilities and value targets must not matter.
            alg.storage.actions[:3, :4] = 9.
            alg.storage.actions_log_prob[:3, :4] = -50.
            alg.storage.returns[:3, :4] = 1e3
        torch.manual_seed(11)
        losses = alg.update()
        assert losses['learner_fraction'] == pytest.approx(1 - 12 / (steps * envs), abs=.2)
        results.append({k: v.clone() for k, v in alg.policy.state_dict().items()})
    for key, value in results[0].items():
        torch.testing.assert_close(results[1][key], value)


def test_teacher_observations_do_not_update_the_normalizer():
    torch = pytest.importorskip('torch')
    pytest.importorskip('rsl_rl')
    from tensordict import TensorDict
    from le_snail.masked_ppo import MaskedPPO
    _, alg = make_algorithms(torch, MaskedPPO)
    alg.init_storage('rl', 2, 4, TensorDict({'policy': torch.zeros(2, 24)}, batch_size=[2]), [6])
    normalizer = alg.policy.actor_obs_normalizer
    before = int(normalizer.count)
    obs = TensorDict({'policy': torch.stack((torch.zeros(24), torch.full((24,), 1e6)))}, batch_size=[2])
    alg.act(obs)
    alg.process_env_step(obs, torch.zeros(2), torch.zeros(2, dtype=torch.long),
                         {'learner_mask': torch.ones(2), 'next_observation_is_learner': torch.tensor([1., 0.])})
    assert int(normalizer.count) == before + 1
    assert float(normalizer._mean.abs().max()) < 1.


SCHEDULE = {'type': 'reverse', 'uniform_share': .1, 'frontier_share': .5, 'promote_success_rate': .5,
            'promote_min_attempts': 50, 'ema_alpha': .02}


def test_valid_schedule_config_loads(tmp_path):
    config = handover_config()
    config['handover']['schedule'] = dict(SCHEDULE)
    config['handover']['scripts']['left'][0]['handover_upright'] = [-.1, .3]
    assert load_physics(write(tmp_path, config))['handover']['schedule']['type'] == 'reverse'


@pytest.mark.parametrize('change', ['type', 'share', 'zero_rate', 'zero_alpha', 'attempts', 'unsorted', 'upright_length'])
def test_invalid_schedule_config_is_rejected(tmp_path, change):
    config = handover_config()
    schedule = config['handover']['schedule'] = dict(SCHEDULE)
    if change == 'type':
        schedule['type'] = 'forward'
    elif change == 'share':
        schedule['uniform_share'] = 1.5
    elif change == 'zero_rate':
        schedule['promote_success_rate'] = 0.
    elif change == 'zero_alpha':
        schedule['ema_alpha'] = 0.
    elif change == 'attempts':
        schedule['promote_min_attempts'] = 0
    elif change == 'unsorted':
        config['handover']['scripts']['left'][0]['prefix_steps'] = [10, 5]
    elif change == 'upright_length':
        config['handover']['scripts']['left'][0]['handover_upright'] = [.1]
    with pytest.raises(ValueError):
        load_physics(write(tmp_path, config))


def test_reverse_schedule_starts_at_latest_point_and_samples_its_window():
    torch = pytest.importorskip('torch')
    from le_snail.curriculum import ReverseSchedule
    torch.manual_seed(0)
    schedule = ReverseSchedule([5, 3], 'cpu', uniform_share=0., frontier_share=1.)
    assert schedule.frontier.tolist() == [4, 2]
    assert set(schedule.sample(torch.zeros(200, dtype=torch.long)).tolist()) == {4}
    schedule = ReverseSchedule([5, 3], 'cpu', uniform_share=.1, frontier_share=.5)
    points = schedule.sample(torch.zeros(4000, dtype=torch.long))
    assert float((points == 4).float().mean()) == pytest.approx(.9 + .1 / 5, abs=.03)
    assert set(points.tolist()) == {0, 1, 2, 3, 4}


def test_reverse_schedule_promotes_only_after_enough_successes_and_never_retreats():
    torch = pytest.importorskip('torch')
    from le_snail.curriculum import ReverseSchedule
    schedule = ReverseSchedule([3], 'cpu', min_attempts=50, promote_at=.5, alpha=.02)
    v, p = torch.zeros(49, dtype=torch.long), torch.full((49,), 2)
    assert not bool(schedule.record(v, p, torch.ones(49, dtype=torch.bool)).any())
    assert float(schedule.rolling_success()[0]) == pytest.approx(1.)
    assert bool(schedule.record(v[:1], p[:1], torch.ones(1, dtype=torch.bool)).all())
    assert schedule.frontier.tolist() == [1] and schedule.attempts.tolist() == [0] and schedule.promotions.tolist() == [1]
    # Outcomes at points other than the frontier do not count; a 40% rate never promotes.
    schedule.record(torch.zeros(100, dtype=torch.long), torch.full((100,), 2), torch.ones(100, dtype=torch.bool))
    assert schedule.attempts.tolist() == [0]
    wins = torch.tensor([True, True, False, False, False] * 40)
    for chunk in wins.split(10):
        schedule.record(torch.zeros(len(chunk), dtype=torch.long), torch.ones(len(chunk), dtype=torch.long), chunk)
    assert schedule.frontier.tolist() == [1]
    assert float(schedule.rolling_success()[0]) == pytest.approx(.4, abs=.05)
    for _ in range(3):
        schedule.record(torch.zeros(60, dtype=torch.long), schedule.frontier.repeat(60), torch.ones(60, dtype=torch.bool))
    assert schedule.frontier.tolist() == [0]
    state = schedule.state()
    restored = ReverseSchedule([3], 'cpu')
    restored.load(state)
    assert restored.state() == state
    with pytest.raises(ValueError):
        ReverseSchedule([4], 'cpu').load(state)
