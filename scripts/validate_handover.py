#!/usr/bin/env python3
"""Phase 0 handover implementation checks on the real environment; zero training (#6).

On <=64 environments: teacher steps earn exactly zero reward and carry learner mask 0; each prefix
runs exactly its assigned length; episode accounting restarts at handover; actor observations keep
the 24-value contract; MaskedPPO rollout storage, advantages and normalizer exclude teacher steps;
the in-environment teacher reproduces the scripted get-ups; evaluation resets never use a teacher.
No optimizer step is taken. Writes only under --output.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--config', default='configs/experiments/handover-001.json')
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--contract', default='configs/policy.json')
p.add_argument('--checkpoint', default='runs/recover-shaped-051-100/model_final.pt')
p.add_argument('--crawl-policy', default='runs/recover-shaped-051-100/exported/policy.pt')
p.add_argument('--sweep', default='runs/slew-handover-sweep-001/results.json')
p.add_argument('--rollout-windows', type=int, default=3)
p.add_argument('--validation-fraction', type=float, default=1., help='Share of side starts handed over in the rollout checks')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new; evidence is never overwritten')
args.output.mkdir(parents=True)
app = AppLauncher(args).app
env = None

try:
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.modules import ActorCritic
    from le_snail.bundle import file_hash
    from le_snail.evaluation import Trial
    from le_snail.masked_ppo import MaskedPPO
    from le_snail.sim.env import SnailEnv, make_config

    N, STEPS = 64, 64
    base = json.loads(Path(args.config).read_text())
    validation = dict(base, recovery_start_fraction=1.)
    validation['handover'] = dict(base['handover'], fraction_of_side_starts=args.validation_fraction)
    validation_path = args.output / 'validation-config.json'
    validation_path.write_text(json.dumps(validation, indent=1) + '\n')
    cfg = make_config(args.asset, args.contract, validation_path, num_envs=N, stage='recover',
                      evaluation=False, device=args.device or 'cuda:0', seed=4242)
    env = SnailEnv(cfg)
    assert env.teacher is not None
    device, dt = env.device, env.step_dt
    wrapper = RslRlVecEnvWrapper(env)
    obs = wrapper.get_observations().to(device)
    runner = json.loads((Path(args.checkpoint).parent / 'run.json').read_text())['runner']
    policy_cfg = dict(runner['policy'])
    assert policy_cfg.pop('class_name') == 'ActorCritic'
    network = ActorCritic(obs, runner['obs_groups'], 6, **policy_cfg).to(device)
    network.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)['model_state_dict'])
    algorithm_cfg = {k: v for k, v in runner['algorithm'].items() if k != 'class_name'}
    alg = MaskedPPO(network, device=device, **algorithm_cfg)
    alg.init_storage('rl', N, STEPS, obs, [6])
    initial_parameters = {k: v.clone() for k, v in network.state_dict().items() if 'normalizer' not in k}
    crawl = torch.jit.load(args.crawl_policy, map_location=device).eval()
    failures = []
    report = {'schema_version': 1, 'training_performed': False, 'num_envs': N,
              'provenance': {str(f): file_hash(f) for f in (args.config, args.asset, args.contract, args.checkpoint,
                                                            args.crawl_policy, args.sweep, __file__,
                                                            'src/le_snail/sim/env.py', 'src/le_snail/masked_ppo.py')},
              'validation_config': f'Derived from --config: every reset a side start; {args.validation_fraction:g} of them use a teacher prefix.',
              'checks': {}}

    def check(name, ok, **details):
        entry = report['checks'].setdefault(name, {'passed': True, 'failures': 0})
        if not ok:
            entry['passed'] = False
            entry['failures'] += 1
            if entry['failures'] <= 5:
                failures.append({'check': name, **details})
        return ok

    neutral = bool(validation.get('side_start_progress_neutral_until_recovery', False))
    heading_reset = bool(validation.get('side_start_heading_reset_at_recovery', False))
    reward_stats = {'progress_counted_after_recovery_steps': 0, 'first_recoveries': 0}

    def reward_rules(ended, reached_before, phase):
        """Side-start reward fixes: neutral progress before recovery; heading reset at first recovery."""
        learner_side = (env.initial_side > 0) & ~ended & ~env.teacher_step
        progress = env.reward_components['progress']
        if neutral:
            before = learner_side & ~env.reached_upright
            check('progress_neutral_before_recovery', bool((progress[before] == 0).all()), phase=phase)
            after = learner_side & env.reached_upright
            reward_stats['progress_counted_after_recovery_steps'] += int((progress[after] != 0).sum())
        if heading_reset:
            first = learner_side & env.reached_upright & ~reached_before
            reward_stats['first_recoveries'] += int(first.sum())
            heading = env.yaw(env.last_transition_quat)
            check('heading_reset_at_first_recovery', bool(torch.equal(env.start_yaw[first], heading[first]))
                  and bool((env.reward_components['yaw'][first] == 0).all()), phase=phase)

    def save():
        report['failures'] = failures
        report['reward_rule_stats'] = reward_stats
        report['passes'] = all(c['passed'] for c in report['checks'].values())
        (args.output / 'report.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')

    # --- A: rollouts through MaskedPPO without any update -------------------------------------
    cap = env.physics['side_upright_progress_reward_cap']
    teacher_count = torch.zeros(N, dtype=torch.long, device=device)
    assigned = env.handover_prefix.clone()
    handed_previous = torch.zeros(N, dtype=torch.bool, device=device)
    stats = {'teacher_steps': 0, 'learner_steps': 0, 'handovers': 0, 'prefix_length_matches': 0,
             'handover_cos': [], 'assignments': {}, 'attempt_steps': env.attempt_steps, 'attempt_truncations': 0}
    logged = torch.zeros((4, 2, 7), device=device)
    bin_names, side_names = ('official', 'below_0.2', '0.2_to_0.5', 'at_least_0.5'), ('left', 'right')
    event_columns = ((0, 'started'), (1, 'completed'), (2, 'stable_get_ups'), (6, 'attempt_truncations'))
    with torch.inference_mode():
        for window in range(args.rollout_windows):
            recorded = []
            for step in range(STEPS):
                observation = obs['policy']
                q = env.robot.data.joint_pos[:, env.joint_ids]
                contract_ok = (observation.shape == (N, 24)
                               and torch.allclose(observation[:, :6], q - env.home, atol=1e-5)
                               and torch.allclose(observation[:, 6:12], env.filtered_velocity, atol=1e-5)
                               and torch.allclose(observation[:, 12:15], env.robot.data.projected_gravity_b, atol=1e-5)
                               and torch.allclose(observation[:, 15:18], env.robot.data.root_ang_vel_b, atol=1e-5)
                               and torch.allclose(observation[:, 18:], (env.applied_target - env.home) / env.scale, atol=1e-5))
                check('observation_contract_unchanged', contract_ok, window=window, step=step)
                teacher = env.teacher_left > 0
                new_episode = env.episode_length_buf == 0
                assigned = torch.where(new_episode & teacher, env.handover_prefix, assigned)
                bin_before, reached_before = env.handover_bin.clone(), env.reached_upright.clone()
                length_before, side_before = env.episode_length_buf.clone(), env.initial_side.clone()
                actions = alg.act(obs)
                count_before = int(network.actor_obs_normalizer.count)
                obs, rewards, dones, extras = wrapper.step(actions.to(env.device))
                obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
                mask = extras['learner_mask']
                for b, bin_name in enumerate(bin_names):
                    for s, side_name in enumerate(side_names):
                        for c, event in event_columns:
                            logged[b, s, c] += float(extras['log'][f'Handover/{bin_name}_{side_name}_{event}'])
                reward_rules(dones > 0, reached_before, 'rollout')
                if env.attempt_steps > 0:
                    ended, time_outs = dones > 0, extras['time_outs'].to(device).bool()
                    truncated = ended & time_outs & (bin_before > 0)
                    stats['attempt_truncations'] += int(truncated.sum())
                    check('attempt_truncation_is_unrecovered_time_out',
                          bool((~reached_before[truncated]).all() and (length_before[truncated] + 1 >= env.attempt_steps).all()),
                          window=window, step=step)
                    official = ended & time_outs & (bin_before == 0) & (side_before > 0)
                    check('official_side_starts_not_truncated',
                          bool((length_before[official] + 1 >= env.max_episode_length).all()), window=window, step=step)
                    open_attempt = ~ended & (env.handover_bin > 0) & ~env.reached_upright & (env.stable_upright_steps == 0)
                    check('no_overlong_unrecovered_attempt', bool((env.episode_length_buf[open_attempt] < env.attempt_steps).all()),
                          window=window, step=step)
                check('learner_mask_matches_teacher_steps', bool(torch.equal(mask, (~teacher).float())), window=window, step=step)
                check('teacher_reward_exactly_zero', bool((rewards[teacher] == 0).all()), window=window, step=step)
                components_zero = all(bool((value[teacher] == 0).all()) for value in env.reward_components.values())
                check('teacher_reward_components_exactly_zero', components_zero, window=window, step=step)
                # Handover accounting: the prefix ended during this step and the episode continues.
                handed = teacher & (env.teacher_left == 0) & (dones == 0)
                pos, gravity, _ = env.last_transition
                up = -gravity[:, 2]
                for i in torch.nonzero(handed).flatten().tolist():
                    stats['handovers'] += 1
                    stats['handover_cos'].append(float(up[i]))
                    length_ok = int(teacher_count[i]) + 1 == int(assigned[i]) or int(env.teacher_key_i[i]) > 3
                    stats['prefix_length_matches'] += int(int(teacher_count[i]) + 1 == int(assigned[i]))
                    check('prefix_runs_assigned_length', length_ok, env=i, counted=int(teacher_count[i]) + 1, assigned=int(assigned[i]))
                    bin_expected = 1 + int(float(up[i]) >= .2) + int(float(up[i]) >= .5)
                    fallen = float(env.fallen_time[i])
                    reset_ok = (torch.equal(env.start_pos[i], pos[i]) and float(env.reset_upright[i]) == float(up[i])
                                and int(env.episode_length_buf[i]) == 0 and float(env.best_upright_progress[i]) == 0.
                                and min(abs(fallen), abs(fallen - dt)) < 1e-6 and int(env.handover_bin[i]) == bin_expected
                                and not bool(env.reached_upright[i]))
                    check('accounting_restarts_at_handover', reset_ok, env=i)
                    key = env.teacher['names'][int(env.handover_variant[i])]
                    stats['assignments'][key] = stats['assignments'].get(key, 0) + 1
                # First learner step after handover: shaping credit is relative to the handover posture.
                first = handed_previous & ~teacher & (dones == 0)
                for i in torch.nonzero(first).flatten().tolist():
                    base_cos = float(env.handover_cos[i])
                    expected = cap * min(max((float(up[i]) - base_cos) / max(.8 - base_cos, 1e-6), 0.), 1.)
                    check('first_learner_credit_relative_to_handover',
                          abs(float(env.reward_components['upright_progress'][i]) - expected) < 1e-5, env=i)
                handed_previous = handed
                teacher_count = torch.where(teacher & ~handed, teacher_count + 1, teacher_count)
                teacher_count = torch.where(handed | (dones > 0), 0, teacher_count)
                following = extras['next_observation_is_learner']
                alg.process_env_step(obs, rewards, dones, extras)
                check('normalizer_counts_learner_rows_only',
                      int(network.actor_obs_normalizer.count) - count_before == int(following.sum()), window=window, step=step)
                recorded.append(mask.clone())
                stats['teacher_steps'] += int(teacher.sum())
                stats['learner_steps'] += int((~teacher).sum())
            check('storage_mask_matches_env', bool(torch.equal(alg.learner_mask.squeeze(-1), torch.stack(recorded))), window=window)
            alg.compute_returns(obs)
            teacher_rows = alg.learner_mask == 0
            learner_adv = alg.storage.advantages[~teacher_rows]
            check('teacher_advantages_zero', bool((alg.storage.advantages[teacher_rows] == 0).all()), window=window)
            check('learner_advantages_normalized', abs(float(learner_adv.mean())) < 1e-4 and abs(float(learner_adv.std()) - 1.) < 1e-3,
                  window=window, mean=float(learner_adv.mean()), std=float(learner_adv.std()))
            alg.storage.clear()  # Discard the rollout: zero training.
            save()
    check('teacher_and_learner_steps_present', stats['teacher_steps'] > 0 and stats['learner_steps'] > 0 and stats['handovers'] > 0)
    columns = [c for c, _ in event_columns]
    check('per_step_log_sums_to_counters', bool(torch.equal(logged[:, :, columns], env._logged_counts[:, :, columns].float())))
    diagnostics = env.recovery_diagnostics()
    closed = sum(side['closed_episodes'] for bins in diagnostics['side_start_before_first_upright']['bins'].values() for side in bins.values())
    check('prerecovery_and_fall_diagnostics_present', closed > 0 and 'upright_start_falls' in diagnostics, closed=closed)
    report['recovery_diagnostics_after_rollout'] = diagnostics
    if env.attempt_steps > 0:
        check('attempt_truncations_observed', stats['attempt_truncations'] > 0, truncations=stats['attempt_truncations'])
    check('every_variant_assigned', set(stats['assignments']) == set(env.teacher['names']), seen=sorted(stats['assignments']))
    unchanged = all(torch.equal(network.state_dict()[k], v) for k, v in initial_parameters.items())
    check('no_parameter_update', unchanged)
    cosines = stats['handover_cos']
    stats['handover_cos'] = {'n': len(cosines), 'min': min(cosines) if cosines else None, 'max': max(cosines) if cosines else None}
    report['rollout'] = stats
    report['handover_diagnostics_after_rollout'] = env.handover_diagnostics()
    save()
    print(json.dumps({'phase': 'rollout', 'checks': {k: v['passed'] for k, v in report['checks'].items()}, 'stats': stats}), flush=True)

    # --- C: the in-environment teacher reproduces the full scripted get-up ---------------------
    sweep = json.loads(Path(args.sweep).read_text())
    confirmed = {b['side']: b for b in sweep['batches'] if b['kind'] == 'confirmed' and b['slew_rad_s'] == 1.}
    full_lengths = {}
    for code, side in ((1, 'left'), (2, 'right')):
        table = env.teacher['sides'][code]
        lengths = []
        for name in env.teacher['names']:
            if not name.startswith(side):
                continue
            label = name.split(':')[1]
            ends = [r['script_end_step'] for r in confirmed[side]['records'] if r['label'] == label and r['script_end_step']]
            lengths.append(int(np.median(ends)))
        table['points'] = torch.tensor(lengths, device=device)[:, None]
        table['counts'] = torch.ones(len(lengths), dtype=torch.long, device=device)
        full_lengths[side] = lengths
    env.teacher['fraction'] = 1.  # Reproduction concerns the teacher itself; every reset gets one.
    with torch.inference_mode():
        observations, _ = env.reset()
        start = env.robot.data.root_pos_w.clone()
        check('reproduction_every_env_has_teacher', bool((env.handover_prefix > 0).all()))
        sides = ['left' if int(s) == 1 else 'right' for s in env.initial_side.tolist()]
        trials = [Trial(kind=side) for side in sides]
        alive = torch.ones(N, dtype=torch.bool, device=device)
        script_completed = torch.zeros(N, dtype=torch.bool, device=device)
        for step in range(1, round(30 / dt) + 1):
            actions = crawl(observations['policy'])
            live = alive.clone()
            was_teacher = env.teacher_left > 0
            recovered_before = env.reached_upright.clone()
            observations, _, terminated, truncated, _ = env.step(actions)
            check('recovered_handover_not_truncated', not bool((truncated & recovered_before & live).any()), step=step)
            reward_rules(terminated | truncated, recovered_before, 'reproduction')
            script_completed |= was_teacher & (env.teacher_key_i > 3)
            pos, gravity, _ = env.last_transition
            disp = (pos - start).cpu().numpy()
            upc = (-gravity[:, 2]).cpu().numpy()
            ended = (terminated | truncated).cpu().numpy()
            livec = live.cpu().numpy()
            for i, trial in enumerate(trials):
                if livec[i]:
                    trial.update(step * dt, float(disp[i, 0]), float(disp[i, 1]), float(upc[i]), fault=bool(ended[i]))
            alive &= ~(terminated | truncated)
            if all(t.finished for t in trials):
                break
    reproduction = {}
    for side in ('left', 'right'):
        rows = [t for t, s in zip(trials, sides) if s == side]
        reproduction[side] = {'episodes': len(rows), 'stable_upright': sum(t.recovery_time_s is not None for t in rows),
                              'successes': sum(t.success for t in rows), 'full_prefix_steps': full_lengths[side]}
        check('in_env_teacher_reproduces_get_up', reproduction[side]['stable_upright'] >= .9 * len(rows), side=side, **reproduction[side])
    reproduction['script_completed_in_prefix'] = int(script_completed.sum())
    report['script_reproduction'] = {'criterion': 'stable upright in >=90% of full-script handover episodes per side; '
                                                  'sweep confirmed batches reached 64/64 per side', **reproduction}
    if neutral:
        check('progress_counted_after_recovery', reward_stats['progress_counted_after_recovery_steps'] > 0)
    if heading_reset:
        check('first_recoveries_observed', reward_stats['first_recoveries'] > 0)
    save()
    print(json.dumps({'phase': 'reproduction', **reproduction}), flush=True)

    # --- B: evaluation resets never use a teacher ----------------------------------------------
    before = env.handover_counts[1:].clone()
    env.cfg.evaluation = True
    with torch.inference_mode():
        for side in ('left', 'right'):
            env.cfg.initial_side = side
            observations, _ = env.reset()
            check('evaluation_reset_has_no_teacher', bool((env.teacher_left == 0).all()), side=side)
            check('evaluation_reset_uses_official_side', bool((env.initial_side == (1 if side == 'left' else 2)).all()), side=side)
            for _ in range(100):
                observations, _, _, _, extras = env.step(crawl(observations['policy']))
                check('evaluation_steps_have_no_teacher', not bool(env.teacher_step.any()) and bool((env.teacher_left == 0).all())
                      and bool((extras['learner_mask'] == 1).all()), side=side)
    check('evaluation_leaves_handover_counts_unchanged', bool(torch.equal(env.handover_counts[1:], before)))
    report['complete'] = True
    save()
    print(json.dumps({'phase0_implementation_checks_pass': report['passes'], 'failures': failures[:5]}), flush=True)
    print('Handover validation complete', flush=True)
except BaseException:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    if env is not None:
        env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
