#!/usr/bin/env python3
"""Verify the two-portion reward in simulation against an independent offline computation; zero training (#6).

Batches (64 envs, from settled starts, after the usual 2 s zero-action settle): per side, the 64 perturbed
and the confirmed scripted get-ups, lying still and random actions; from the upright start, standing
still, random actions and the frozen 20 Hz crawl policy (run at 50 Hz purely as a source of real motion).
Every step captures the exact inputs the environment used for its reward; an independent numpy
implementation recomputes each component from them. Behaviour checks repeat the offline evidence.
Writes only under --output.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--config', default='configs/experiments/two-portion-002.json')
p.add_argument('--contract', default='configs/policy-50hz-001.json')
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--confirmation', default='runs/feasibility-side-recovery-20261014T111801Z/results.json')
p.add_argument('--crawl-policy', default='runs/recover-shaped-051-100/exported/policy.pt')
p.add_argument('--seconds', type=float, default=8.)
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
    from le_snail.bundle import file_hash
    from le_snail.evaluation import Trial
    from le_snail.sim.env import SnailEnv, make_config

    class VerifyEnv(SnailEnv):
        def _two_portion_rewards(self):
            self.captured = {'gravity': self.robot.data.projected_gravity_b.clone(),
                             'joint': self.robot.data.joint_pos[:, self.joint_ids].clone(),
                             'x': self.robot.data.root_pos_w[:, 0].clone(), 'x_record': self.x_record.clone(),
                             'rate': ((self.target - self.old_target) / self.step_dt).clone(),
                             'terminated': self.reset_terminated.clone(), 'previous_phi': self.previous_phi.clone(),
                             'crawling': self.crawling.clone()}
            return super()._two_portion_rewards()

    N, SEED = 64, 4242
    cfg = make_config(args.asset, args.contract, args.config, num_envs=N, stage='recover', evaluation=True,
                      initial_side='left', device=args.device or 'cuda:0', seed=SEED)
    env = VerifyEnv(cfg)
    assert env.two_portion is not None
    device, dt = env.device, env.step_dt
    reward_cfg, gamma = env.two_portion['config'], env.two_portion['gamma']
    bank = np.load(reward_cfg['getup_progress_bank'])
    b_states, b_phase = bank['bank_states'].astype(np.float64), bank['phases'].astype(np.float64)
    b_mean, b_std, sigma, k, shrink = bank['mean'], bank['std'], float(bank['sigma']), int(bank['k']), float(bank['shrink'])
    crawl = torch.jit.load(args.crawl_policy, map_location=device).eval()
    source = json.loads(Path(args.confirmation).read_text())
    perturb = np.random.default_rng(SEED).uniform(-.1, .1, (64, 3, 6))
    ar = torch.arange(N, device=device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    report = {'schema_version': 1, 'training_performed': False, 'num_envs': N, 'policy_dt': dt, 'gamma': gamma,
              'provenance': {str(f): file_hash(f) for f in (args.config, args.contract, reward_cfg['getup_progress_bank'], args.asset,
                                                            __file__, 'src/le_snail/sim/env.py', 'src/le_snail/rewards.py')},
              'note': 'The upright crawl batch runs a 20 Hz policy at 50 Hz only to produce real motion for the forward term.',
              'batches': [], 'checks': {}}
    failures, mismatches = [], []

    def check(name, ok, **details):
        entry = report['checks'].setdefault(name, {'passed': True, 'failures': 0})
        if not ok:
            entry['passed'] = False
            entry['failures'] += 1
            if entry['failures'] <= 3:
                failures.append({'check': name, **details})

    def ramp(value, low, high):
        return np.clip((value - low) / (high - low), 0., 1.)

    def offline(c):
        """Independent float64 numpy recomputation of every component from the captured inputs."""
        upright = -c['gravity'][:, 2]
        z = (np.concatenate((c['gravity'], c['joint']), axis=1) - b_mean) / b_std
        d2 = ((z[:, None, :] - b_states[None, :, :]) ** 2).sum(2)
        index = np.argpartition(d2, k - 1, axis=1)[:, :k]   # the k nearest; their order does not matter
        near = np.take_along_axis(d2, index, axis=1)
        weight = np.exp(-near / (2 * sigma ** 2))
        progress = (weight * b_phase[index]).sum(1) / (weight.sum(1) + shrink)
        stand = ramp(upright, *reward_cfg['standing_ramp_cos']) ** 2
        gate = ramp(upright, *reward_cfg['forward_gate_cos'])
        forward_gate = gate * c['crawling'] if reward_cfg.get('forward_requires_crawl_mode', False) else gate
        phi = np.maximum(progress, stand)
        following = np.where(c['terminated'], 0., phi)
        shaping = np.where(np.isfinite(c['previous_phi']), reward_cfg['getup_progress_weight'] * (gamma * following - c['previous_phi']), 0.)
        return {'getup_progress': shaping, 'standing': reward_cfg['standing_per_s'] * dt * stand,
                'forward': reward_cfg['forward_per_m'] * forward_gate * np.maximum(c['x'] - c['x_record'], 0.),
                'effort': -reward_cfg['effort_weight'] * dt * (c['rate'] ** 2).sum(1) * gate}, phi, gate

    def confirmed_ids(side):
        variants = {'left': (9, 34), 'right': (48, 62)}[side]
        ids = [variants[i % 2] for i in range(N)]
        for v in variants:
            match = [r for r in source['results'] if r['stage'] == 'recover' and r['side'] == side and r['variant'] == f'perturbed{v}'
                     and r['effort_nm'] == .8 and r['vel_limit'] == 2. and r['success']]
            ids[match[0]['env']] = v
        return ids

    def run(start, kind, variants=None):
        env.cfg.initial_side = start
        obs, _ = env.reset(seed=SEED)
        zero = torch.zeros((N, 6), device=device)
        for _ in range(round(2 / dt)):
            obs, *_ = env.step(zero)
        begin = env.robot.data.root_pos_w.clone()
        keys = None
        if variants is not None:
            base = np.asarray(source['keys'][start])
            keys = env.tensor(np.array([np.vstack((base + perturb[v], env.contract.home_rad)) for v in variants]))
        key_i = torch.zeros(N, dtype=torch.long, device=device)
        held = torch.zeros_like(key_i)
        trials = [Trial(kind='crawl' if start == 'upright' else start) for _ in range(N)]
        live = torch.ones(N, dtype=torch.bool, device=device)
        steps = round(args.seconds / dt)
        totals = np.zeros((steps, N))
        max_error = {name: 0. for name in ('getup_progress', 'standing', 'forward', 'effort')}
        previous_offline_phi, previous_x_record = None, None
        standing_sum, shaping_sum = np.zeros(N), np.zeros(N)
        upright_trace = np.zeros((steps, N))
        phi_trace = np.zeros((steps, N))
        for step in range(steps):
            if kind == 'random':
                actions = torch.randn((N, 6), device=device, generator=generator)
            elif kind == 'crawl':
                actions = crawl(obs['policy'])
            elif keys is None:
                actions = zero
            else:
                goal = keys[ar, key_i.clamp(max=3)]
                actions = ((goal - env.home) / env.scale).clamp(-1, 1)
            alive = live.clone()
            obs, reward, terminated, truncated, _ = env.step(actions)
            if keys is not None:
                clipped = torch.clamp(env.home + actions * env.scale, env.lower, env.upper)
                reached = (env.target - clipped).abs().amax(dim=1) < 1e-3
                running = key_i <= 3
                held = torch.where(reached & running, held + 1, 0)
                advance = running & (held >= torch.where(key_i == 3, round(1. / dt), round(.1 / dt)))
                key_i += advance.long()
                held = torch.where(advance, 0, held)
            captured = {name: value.double().cpu().numpy() if value.dtype != torch.bool else value.cpu().numpy()
                        for name, value in env.captured.items()}
            expected, phi, gate = offline(captured)
            for name in max_error:
                max_error[name] = max(max_error[name], float(np.abs(env.reward_components[name].double().cpu().numpy() - expected[name]).max()))
            gap = np.abs(env.reward_components['getup_progress'].double().cpu().numpy() - expected['getup_progress'])
            env_phi = env.getup_potential.double().cpu().numpy()
            for i in np.nonzero(gap > 1e-4)[0][:max(0, 20 - len(mismatches))]:
                mismatches.append({'start': start, 'kind': kind, 'step': step, 'env': int(i), 'features': np.concatenate((captured['gravity'][i], captured['joint'][i])).tolist(),
                                   'previous_phi': float(captured['previous_phi'][i]), 'terminated': bool(captured['terminated'][i]),
                                   'env_phi': float(env_phi[i]), 'offline_phi': float(phi[i]),
                                   'env_shaping': float(env.reward_components['getup_progress'][i]), 'offline_shaping': float(expected['getup_progress'][i])})
            standing_sum += env.reward_components['standing'].double().cpu().numpy() * alive.cpu().numpy()
            shaping_sum += env.reward_components['getup_progress'].double().cpu().numpy() * alive.cpu().numpy()
            components = sum(env.reward_components[name].double().cpu().numpy() for name in env.reward_components)
            check('total_equals_sum_of_components', bool(np.allclose(components, reward.double().cpu().numpy(), atol=1e-5)), kind=kind)
            first = ~np.isfinite(captured['previous_phi'])
            check('no_shaping_on_first_step', bool((env.reward_components['getup_progress'].cpu().numpy()[first] == 0).all()), kind=kind)
            check('effort_zero_while_fallen', bool((env.reward_components['effort'].cpu().numpy()[gate == 0] == 0).all()), kind=kind)
            check('forward_never_negative', bool((env.reward_components['forward'].cpu().numpy() >= 0).all()), kind=kind)
            if reward_cfg.get('forward_requires_crawl_mode', False):
                check('forward_only_while_crawling', bool((env.reward_components['forward'].cpu().numpy()[~captured['crawling']] == 0).all()), kind=kind)
            if previous_offline_phi is not None:
                carried = np.isfinite(captured['previous_phi'])
                check('potential_carried_between_steps', bool(np.allclose(captured['previous_phi'][carried], previous_offline_phi[carried], atol=1e-4)), kind=kind)
                check('x_record_carried_between_steps', bool((captured['x_record'][carried] >= previous_x_record[carried] - 1e-9).all()), kind=kind)
            previous_offline_phi, previous_x_record = phi, np.maximum(captured['x_record'], captured['x'])
            pos, gravity_now, _ = env.last_transition
            up = -gravity_now[:, 2]
            disp, upc, endc, livec = (pos - begin).cpu().numpy(), up.cpu().numpy(), (terminated | truncated).cpu().numpy(), alive.cpu().numpy()
            for i, trial in enumerate(trials):
                if livec[i]:
                    trial.update((step + 1) * dt, float(disp[i, 0]), float(disp[i, 1]), float(upc[i]), fault=bool(endc[i]))
            live &= ~(terminated | truncated)
            totals[step] = reward.cpu().numpy() * livec
            upright_trace[step], phi_trace[step] = upc, phi
        for name, error in max_error.items():
            check(f'{name}_matches_offline', error < (1e-4 if name == 'getup_progress' else 1e-5), kind=kind, start=start, error=error)
        cumulative = np.cumsum(totals, axis=0)
        at = {f'{s:g}s': cumulative[round(s / dt) - 1] for s in (1, 2, 4, args.seconds)}
        success = np.array([t.recovery_time_s is not None for t in trials])
        batch = {'start': start, 'kind': kind, 'stable_upright': int(success.sum()), 'max_component_error_vs_offline': max_error,
                 'cumulative_median': {h: float(np.median(v)) for h, v in at.items()},
                 'cumulative_min': {h: float(np.min(v)) for h, v in at.items()}, 'cumulative_max': {h: float(np.max(v)) for h, v in at.items()},
                 'phi_end_median': float(np.median(phi_trace[-1]))}
        if start != 'upright' and variants is not None:
            if success.any():
                batch['successes_min'] = {h: float(v[success].min()) for h, v in at.items()}
                check('scripted_successes_positive_from_1s', all(float(v[success].min()) > 0 for v in at.values()), start=start, kind=kind)
            if (~success).any():
                # Judge failures by how they end: one that ends lying down must not keep reward; one that is
                # upright when the window closes (not yet 0.5 s stable) is a late success; half-up is reported.
                end = at[f'{args.seconds:g}s']
                final = upright_trace[-1]
                lying, tilted, up_late = (~success) & (final < .1), (~success) & (final >= .1) & (final < .8), (~success) & (final >= .8)
                batch['failures_by_final_posture'] = {
                    'lying': {'n': int(lying.sum()), 'end_reward_max': float(end[lying].max()) if lying.any() else None},
                    'tilted': {'n': int(tilted.sum()), 'end_reward': end[tilted].round(3).tolist(), 'final_cos': final[tilted].round(3).tolist()},
                    'upright_not_yet_stable': {'n': int(up_late.sum()), 'end_reward': end[up_late].round(3).tolist()}}
                batch['failure_details'] = [
                    {'env': int(i), 'end_reward': float(end[i]), 'standing_sum': float(standing_sum[i]), 'shaping_sum': float(shaping_sum[i]),
                     'max_cos': float(upright_trace[:, i].max()), 'final_cos': float(final[i]), 'phi_end': float(phi_trace[-1, i]),
                     'seconds_above_0.5': float((upright_trace[:, i] >= .5).sum() * dt), 'reason': trials[i].reason}
                    for i in np.nonzero(~success)[0] if end[i] > .1]
                if lying.any():
                    check('failures_ending_lying_keep_no_reward', float(end[lying].max()) <= .1, start=start, kind=kind, end=float(end[lying].max()))
                if tilted.any() and success.any():
                    check('tilted_endings_earn_less_than_successes', float(end[tilted].max()) < float(np.median(end[success])), start=start, kind=kind)
        if start != 'upright' and kind == 'hold':
            check('lying_still_earns_zero', float(np.abs(at[f'{args.seconds:g}s']).max()) < .02, start=start)
        if start != 'upright' and kind == 'random':
            check('flailing_earns_about_zero', float(np.abs(at[f'{args.seconds:g}s']).max()) < .15, start=start)
        if start == 'upright' and kind == 'hold':
            per_step = totals[-1]
            expected_hold = reward_cfg['standing_per_s'] * dt + reward_cfg['getup_progress_weight'] * (gamma - 1.)
            check('upright_hold_rate', bool(np.allclose(per_step, expected_hold, atol=2e-4)), expected=expected_hold, got=float(np.median(per_step)))
        report['batches'].append(batch)
        report['failures'] = failures
        report['shaping_mismatch_examples'] = mismatches
        report['passes'] = all(c['passed'] for c in report['checks'].values())
        (args.output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
        print(json.dumps({'start': start, 'kind': kind, 'stable_upright': batch['stable_upright'], 'median': batch['cumulative_median'],
                          'max_error': {k: f'{v:.1e}' for k, v in max_error.items()}}), flush=True)

    with torch.no_grad():
        for side in ('left', 'right'):
            run(side, 'perturbed', list(range(N)))
            run(side, 'confirmed', confirmed_ids(side))
            run(side, 'hold')
            run(side, 'random')
        run('upright', 'hold')
        run('upright', 'random')
        run('upright', 'crawl')
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'passes': report['passes'], 'failed': [k for k, v in report['checks'].items() if not v['passed']]}), flush=True)
    print('Two-portion reward verification complete', flush=True)
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
