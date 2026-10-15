#!/usr/bin/env python3
"""Zero-training target-slew sweep and handover-point audit on nominal side starts; never trains.

Slew Part 1 and the handover-curriculum Phase 0 physics grid (laptop decision
2026-10-15T02:36:00Z, gate amendment 02:38:16Z, #6). For each slew value:

- basin: the seed-4242 +-0.1 rad perturbations of the confirmed keyframe plan, one per env,
  run through the normal action path, then the frozen policy; acceptance-trial scoring.
- confirmed / hold: the exact confirmed scripts (64 replicas) versus holding home, with
  audit-002 return accounting (30 s, gamma .995, termination absorbing, settling excluded).
- passive grid: a script prefix of k steps, then hold the last commanded target for 3 s with
  curriculum accounting reset at handover (shaping baseline, origin, yaw, timers).
- policy grid: the same prefixes, then the frozen policy, scored as an official trial.

The slew is overridden at runtime for this sweep only; the contract file is unchanged.
Writes only under --output.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--slews', type=float, nargs='+', default=[1., 2., 3.])
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--contract', default='configs/policy.json')
p.add_argument('--physics', default='configs/experiments/physics-v2-002.json')
p.add_argument('--confirmation', default='runs/feasibility-side-recovery-20261014T111801Z/results.json')
p.add_argument('--policy', default='runs/recover-shaped-051-100/exported/policy.pt')
p.add_argument('--grid-points', type=int, default=16)
p.add_argument('--max-process-seconds', type=float, default=2400.)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new; evidence is never overwritten')
if sorted(args.slews) != args.slews or args.slews[0] != 1.:
    p.error('--slews must be ascending and start at the 1.0 rad/s baseline')
started = time.monotonic()
args.output.mkdir(parents=True)
app = AppLauncher(args).app
env = None

try:
    import numpy as np
    import torch
    from le_snail.action_stats import ActionDiagnostics
    from le_snail.bundle import file_hash
    from le_snail.evaluation import Trial
    from le_snail.sim.env import SnailEnv, make_config
    from le_snail.speed import PhysicsSpeedDiagnostics

    N, SEED, GAMMA = 64, 4242, .995
    HOLD_STEPS_S, CAP = 3., 1.
    PASSIVE_GATES = {'max_passive_get_ups': 0, 'max_passive_return': 0., 'max_passive_shaping_credit': .1}
    VARIANTS = {'left': (9, 34), 'right': (48, 62)}
    cfg = make_config(args.asset, args.contract, args.physics, num_envs=N, stage='recover',
                      evaluation=True, initial_side='left', device=args.device or 'cuda:0', seed=SEED)
    env = SnailEnv(cfg)
    device, dt = env.device, env.step_dt
    assert env.physics['effort_limit_nm'] == 1.5 and env.physics['velocity_limit_rad_s'] == 4.
    assert all(v == 1. for v in env.contract.max_target_speed_rad_s)
    assert env.physics['side_upright_progress_reward_cap'] == CAP
    policy = torch.jit.load(args.policy, map_location=device).eval()
    source = json.loads(Path(args.confirmation).read_text())
    perturb = np.random.default_rng(SEED).uniform(-.1, .1, (64, 3, 6))
    ar = torch.arange(N, device=device)
    home_np = np.asarray(env.contract.home_rad)
    trial_steps = round(30 / dt)
    hold_steps = round(HOLD_STEPS_S / dt)

    def git_head():
        head = Path('.git/HEAD').read_text().strip()
        if head.startswith('ref: '):
            ref = Path('.git') / head[5:]
            return ref.read_text().strip() if ref.exists() else head
        return head

    report = {
        'schema_version': 1, 'training_performed': False, 'physics_validated': False,
        'decision': 'LAPTOP 2026-10-15T02:36:00Z [DECISION robin] A + C Phase 0 grid; gate amendment 02:38:16Z',
        'num_envs': N, 'seed': SEED, 'gamma': GAMMA, 'policy_dt': dt, 'slews_rad_s': args.slews,
        'slew_override': 'Runtime override of env.slew for every joint; contract file and fingerprint unchanged.',
        'scope_note': 'Speed/action diagnostics cover the scored period only (after the 2 s zero-action settle).',
        'continuation_policy_note': 'The frozen policy was trained at 1 rad/s; at higher slews it runs faster than trained.',
        'passive_gates': PASSIVE_GATES, 'passive_window_s': HOLD_STEPS_S,
        'provenance': {'git_head': git_head(),
                       'files': {str(f): file_hash(f) for f in (args.asset, args.contract, args.physics, args.confirmation,
                                                                args.policy, __file__, 'src/le_snail/sim/env.py')}},
        'batches': [], 'summary': {},
    }

    def save():
        report['process_elapsed_seconds'] = time.monotonic() - started
        (args.output / 'results.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')

    def check_budget():
        assert time.monotonic() - started < args.max_process_seconds, 'Process budget exhausted'

    def set_slew(value):
        env.slew = torch.full((6,), value * dt, device=device)

    def confirmed_ids(side):
        variants = VARIANTS[side]
        ids = [variants[i % 2] for i in range(N)]
        for v in variants:
            matches = [r for r in source['results'] if r['stage'] == 'recover' and r['side'] == side
                       and r['variant'] == f'perturbed{v}' and r['effort_nm'] == .8 and r['vel_limit'] == 2. and r['success']]
            assert matches, (side, v)
            ids[matches[0]['env']] = v
        return ids

    def key_tensor(side, perturbation_ids):
        base = np.asarray(source['keys'][side])
        return env.tensor(np.array([np.vstack((base + perturb[i], home_np)) for i in perturbation_ids]))

    class Script:
        """Keyframe sequencing identical to the feasibility probe and reward audits."""
        def __init__(self, keys):
            self.keys = keys
            self.key_i = torch.zeros(N, dtype=torch.long, device=device)
            self.held = torch.zeros_like(self.key_i)

        def running(self):
            return self.key_i <= 3

        def actions(self):
            goal = self.keys[ar, self.key_i.clamp(max=3)]
            return ((goal - env.home) / env.scale).clamp(-1, 1)

        def advance(self, script_actions, mask):
            clipped = torch.clamp(env.home + script_actions * env.scale, env.lower, env.upper)
            reached = (env.target - clipped).abs().amax(dim=1) < 1e-3
            moving = self.running() & mask
            self.held = torch.where(reached & moving, self.held + 1, 0)
            advance = moving & (self.held >= torch.where(self.key_i == 3, 20, 2))
            self.key_i += advance.long()
            self.held = torch.where(advance, 0, self.held)
            return advance & (self.key_i > 3)

    def begin(side, reset_seed):
        env.cfg.initial_side = side
        obs, _ = env.reset(seed=reset_seed)
        zero = torch.zeros((N, 6), device=device)
        for _ in range(round(2 / dt)):
            obs, _, term, trunc, _ = env.step(zero)
            assert not bool((term | trunc).any()), 'Settling fault'
        start = env.robot.data.root_pos_w.clone()
        env.start_pos.copy_(start)
        env.episode_length_buf.zero_()
        env.fallen_time.zero_()
        env.speed_monitor = PhysicsSpeedDiagnostics(N, 6, env.physics_dt, device)
        env.action_monitor = ActionDiagnostics(N, env.contract.joint_names, device)
        return obs, start

    def stats(values):
        values = [v for v in values if v is not None]
        return {'n': len(values), 'mean': float(np.mean(values)) if values else None,
                'median': float(np.median(values)) if values else None,
                'min': float(np.min(values)) if values else None, 'max': float(np.max(values)) if values else None}

    def run_trials(slew, side, kind, keys, prefix, after, labels, reset_seed=SEED, full_horizon=False):
        """Acceptance-trial batch. prefix: per-env script steps before handover (None: whole script)."""
        obs, start = begin(side, reset_seed)
        script = Script(keys) if keys is not None else None
        limit = torch.full((N,), 10 ** 9, dtype=torch.long, device=device) if prefix is None else prefix
        active = torch.ones(N, dtype=torch.bool, device=device)
        trials = [Trial(kind=side) for _ in range(N)]
        names = None
        disc = undisc = disc4 = None
        first80 = torch.full((N,), -1, dtype=torch.long, device=device)
        script_end = torch.full((N,), -1, dtype=torch.long, device=device)
        max_up = torch.full((N,), -2., device=device)
        trace = []
        for step in range(1, trial_steps + 1):
            check_budget()
            in_prefix = (script.running() & (step <= limit)) if script is not None else torch.zeros(N, dtype=torch.bool, device=device)
            scripted = script.actions() if script is not None else None
            if after == 'policy':
                later = policy(obs['policy'])
            else:
                later = torch.zeros((N, 6), device=device)
            actions = torch.where(in_prefix[:, None], scripted, later) if script is not None else later
            live = active.clone()
            obs, reward, terminated, truncated, _ = env.step(actions)
            if names is None:
                names = list(env.reward_components)
                disc = torch.zeros((N, len(names)), device=device)
                undisc, disc4 = torch.zeros_like(disc), torch.zeros_like(disc)
            comps = torch.stack([env.reward_components[name] for name in names], dim=1)
            torch.testing.assert_close(comps.sum(dim=1), reward)
            assert torch.isfinite(comps).all() and torch.isfinite(actions).all()
            if script is not None:
                done = script.advance(scripted, in_prefix & live)
                script_end = torch.where(done & (script_end < 0), step, script_end)
            weighted = comps * live[:, None]
            disc += weighted * GAMMA ** (step - 1)
            undisc += weighted
            if step <= round(4 / dt):
                disc4 += weighted * GAMMA ** (step - 1)
            pos, gravity, _ = env.last_transition
            disp = (pos - start).cpu().numpy()
            up = -gravity[:, 2]
            ended = terminated | truncated
            first80 = torch.where(live & (up >= .8) & (first80 < 0), step, first80)
            max_up = torch.where(live, torch.maximum(max_up, up), max_up)
            upc, endc, livec = up.cpu().numpy(), ended.cpu().numpy(), live.cpu().numpy()
            for i, trial in enumerate(trials):
                if livec[i]:
                    trial.update(step * dt, float(disp[i, 0]), float(disp[i, 1]), float(upc[i]), fault=bool(endc[i]))
            active &= ~ended
            trace.append(upc)
            if not full_horizon and all(t.finished for t in trials):
                break
        cr = names.index('command_rate')
        totals = disc.sum(dim=1)
        records = []
        for i, trial in enumerate(trials):
            records.append({'env': i, 'label': labels[i], 'success': trial.success, 'reason': trial.reason,
                            'stable_upright_time_s': trial.recovery_time_s, 'acceptance_end_s': trial.last_time_s,
                            'first_upright80_step': int(first80[i]) if first80[i] >= 0 else None,
                            'script_end_step': int(script_end[i]) if script_end[i] >= 0 else None,
                            'prefix_steps': None if prefix is None else int(prefix[i]),
                            'max_upright': float(max_up[i]),
                            'discounted30_total': float(totals[i]), 'discounted30_command_rate': float(disc[i, cr]),
                            'discounted4_total': float(disc4[i].sum()),
                            'discounted30_components': dict(zip(names, disc[i].tolist())),
                            'undiscounted30_components': dict(zip(names, undisc[i].tolist()))})
        batch = {'slew_rad_s': slew, 'side': side, 'kind': kind, 'reset_seed': reset_seed, 'policy_steps': len(trace),
                 'full_horizon': full_horizon, 'records': records,
                 'stable_upright': sum(r['stable_upright_time_s'] is not None for r in records),
                 'successes': sum(r['success'] for r in records),
                 'speed_diagnostics': env.speed_diagnostics(), 'action_diagnostics': env.action_diagnostics()}
        np.savez_compressed(args.output / f'slew{slew:g}-{side}-{kind}-upright.npz', upright=np.asarray(trace))
        report['batches'].append(batch)
        save()
        print(json.dumps({'slew': slew, 'side': side, 'kind': kind, 'stable_upright': batch['stable_upright'],
                          'successes': batch['successes'], 'steps': len(trace),
                          'over_cap_fraction': batch['speed_diagnostics']['over_limit_environment_time_fraction'],
                          'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)
        return batch

    def run_passive(slew, side, keys, prefix, labels, reset_seed):
        """Script prefix of k steps, then hold the commanded target for 3 s with handover accounting."""
        obs, start = begin(side, reset_seed)
        script = Script(keys)
        active = torch.ones(N, dtype=torch.bool, device=device)
        handed = torch.zeros(N, dtype=torch.bool, device=device)
        hold_target = env.target.clone()
        handover_cos = torch.full((N,), float('nan'), device=device)
        handover_xy = torch.zeros((N, 2), device=device)
        ret = torch.zeros(N, device=device)
        credit = torch.zeros(N, device=device)
        get_up = torch.zeros(N, dtype=torch.bool, device=device)
        fault = torch.zeros(N, dtype=torch.bool, device=device)
        excursion = torch.zeros(N, device=device)
        max_cos = torch.full((N,), -2., device=device)
        end_cos = torch.full((N,), float('nan'), device=device)
        prefix_fault = torch.zeros(N, dtype=torch.bool, device=device)
        last = int(prefix.max()) + hold_steps
        for step in range(1, last + 1):
            check_budget()
            # Handover happens before the first post-prefix step: reset the curriculum accounting.
            now = active & ~handed & (prefix == step - 1)
            if bool(now.any()):
                upright = -env.robot.data.projected_gravity_b[:, 2]
                hold_target = torch.where(now[:, None], env.target, hold_target)
                handover_cos = torch.where(now, upright, handover_cos)
                handover_xy = torch.where(now[:, None], env.robot.data.root_pos_w[:, :2], handover_xy)
                env.start_pos[now] = env.robot.data.root_pos_w[now]
                env.start_yaw[now] = env.yaw(env.robot.data.root_quat_w[now])
                env.reset_upright[now] = upright[now]
                env.best_upright_progress[now] = 0.
                env.fallen_time[now] = 0.
                env.stable_upright_steps[now] = 0
                env.reached_upright[now] = False
                env.episode_length_buf[now] = 0
                handed |= now
            in_prefix = ~handed
            scripted = script.actions()
            held_actions = (hold_target - env.home) / env.scale
            actions = torch.where(in_prefix[:, None], scripted, held_actions)
            live = active.clone()
            obs, reward, terminated, truncated, _ = env.step(actions)
            script.advance(scripted, in_prefix & live)
            ended = terminated | truncated
            window = live & handed & (step <= prefix + hold_steps)
            ret += reward * window
            credit += env.reward_components['upright_progress'] * window
            get_up |= env.reached_upright & window
            pos, gravity, _ = env.last_transition
            up = -gravity[:, 2]
            excursion = torch.where(window, torch.maximum(excursion, (pos[:, :2] - handover_xy).norm(dim=1)), excursion)
            max_cos = torch.where(window, torch.maximum(max_cos, up), max_cos)
            end_cos = torch.where(window & (step == prefix + hold_steps), up, end_cos)
            fault |= ended & window
            prefix_fault |= ended & live & ~handed
            active &= ~ended
        records = []
        for i in range(N):
            records.append({'env': i, 'label': labels[i], 'prefix_steps': int(prefix[i]),
                            'handover_upright': float(handover_cos[i]) if bool(handed[i]) else None,
                            'passive_get_up': bool(get_up[i]), 'passive_return': float(ret[i]),
                            'passive_shaping_credit': float(credit[i]), 'fault_in_window': bool(fault[i]),
                            'fault_in_prefix': bool(prefix_fault[i]), 'max_planar_excursion_m': float(excursion[i]),
                            'max_upright_in_window': float(max_cos[i]),
                            'end_upright': float(end_cos[i]) if bool(torch.isfinite(end_cos[i])) else None})
        batch = {'slew_rad_s': slew, 'side': side, 'kind': 'passive_grid', 'reset_seed': reset_seed, 'records': records,
                 'speed_diagnostics': env.speed_diagnostics(), 'action_diagnostics': env.action_diagnostics()}
        report['batches'].append(batch)
        save()
        print(json.dumps({'slew': slew, 'side': side, 'kind': 'passive_grid', 'seed': reset_seed,
                          'passive_get_ups': int(get_up.sum()), 'faults': int((fault | prefix_fault).sum()),
                          'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)
        return batch

    def grid_layout(points, batch_index):
        """64 envs = 2 variants x 16 grid points x 2 replica slots, rotated per batch."""
        layout = [None] * N
        for slot in range(2):
            for vi, (variant, ks) in enumerate(points):
                for g in range(16):
                    k = ks[g % len(ks)]
                    base = (slot * 2 + vi) * 16 + g
                    layout[(base + 17 * batch_index) % N] = (variant, k)
        return layout

    summary = report['summary']
    with torch.no_grad():
        for slew in args.slews:
            set_slew(slew)
            entry = summary[f'{slew:g}'] = {}
            for side in ('left', 'right'):
                side_entry = entry[side] = {}
                ids = confirmed_ids(side)
                confirmed = run_trials(slew, side, 'confirmed', key_tensor(side, ids), None, 'policy',
                                       [f'p{v}' for v in ids], full_horizon=True)
                hold = run_trials(slew, side, 'hold', None, None, 'zero', ['hold'] * N, full_horizon=True)
                basin = run_trials(slew, side, 'basin', key_tensor(side, range(N)), None, 'policy',
                                   [f'p{i}' for i in range(N)])
                positive = {f'p{v}': next(r for r in basin['records'] if r['label'] == f'p{v}')['success'] for v in VARIANTS[side]}

                def returns(batch, field):
                    return float(np.mean([r[field] for r in batch['records']]))
                side_entry['basin'] = {
                    'n': N, 'stable_upright_fraction': basin['stable_upright'] / N, 'success_fraction': basin['successes'] / N,
                    'time_to_stable_upright_s': stats([r['stable_upright_time_s'] for r in basin['records']]),
                    'confirmed_perturbations_in_basin_success': positive,
                    'over_cap_environment_time_fraction': basin['speed_diagnostics']['over_limit_environment_time_fraction'],
                    'peak_abs_velocity_rad_s': basin['speed_diagnostics']['peak_abs_velocity_rad_s'],
                    'longest_driven_proxy_s': basin['speed_diagnostics']['longest_driven_proxy_over_limit_s']}
                script_total, hold_total = returns(confirmed, 'discounted30_total'), returns(hold, 'discounted30_total')
                script_cr, hold_cr = returns(confirmed, 'discounted30_command_rate'), returns(hold, 'discounted30_command_rate')
                side_entry['returns'] = {
                    'script_mean_discounted30': script_total, 'hold_mean_discounted30': hold_total,
                    'script_mean_command_rate': script_cr, 'hold_mean_command_rate': hold_cr,
                    'script_without_command_rate': script_total - script_cr, 'hold_without_command_rate': hold_total - hold_cr,
                    'script_normalized_command_rate': script_total - script_cr + script_cr / slew ** 2,
                    'hold_normalized_command_rate': hold_total - hold_cr + hold_cr / slew ** 2,
                    'script_beats_hold': script_total > hold_total,
                    'script_mean_discounted4': returns(confirmed, 'discounted4_total'),
                    'hold_mean_discounted4': returns(hold, 'discounted4_total'),
                    'confirmed_stable_upright': confirmed['stable_upright'], 'confirmed_successes': confirmed['successes'],
                    'confirmed_over_cap_environment_time_fraction': confirmed['speed_diagnostics']['over_limit_environment_time_fraction']}
                # Handover grid from this slew's confirmed timeline: points before the base first reaches 0.8.
                points = []
                for variant in VARIANTS[side]:
                    recs = [r for r in confirmed['records'] if r['label'] == f'p{variant}' and r['first_upright80_step'] is not None]
                    assert recs, f'No confirmed replica of p{variant} reached upright at slew {slew}'
                    k80 = int(np.median([r['first_upright80_step'] for r in recs]))
                    k_end = int(np.median([r['script_end_step'] for r in recs if r['script_end_step'] is not None] or [k80]))
                    last_k = max(min(k80, k_end) - 1, 0)
                    ks = sorted(set(int(round(x)) for x in np.linspace(0, last_k, args.grid_points)))
                    points.append((variant, ks))
                side_entry['grid_points'] = {f'p{v}': ks for v, ks in points}
                passive_records = []
                for batch_index in range(2):
                    layout = grid_layout(points, batch_index)
                    keys = key_tensor(side, [v for v, _ in layout])
                    prefix = torch.tensor([k for _, k in layout], device=device)
                    labels = [f'p{v}@{k}' for v, k in layout]
                    passive_records += run_passive(slew, side, keys, prefix, labels, SEED + batch_index)['records']
                layout = grid_layout(points, 0)
                active_batch = run_trials(slew, side, 'policy_grid', key_tensor(side, [v for v, _ in layout]),
                                          torch.tensor([k for _, k in layout], device=device), 'policy',
                                          [f'p{v}@{k}' for v, k in layout])
                grid = {}
                for variant, ks in points:
                    for k in ks:
                        label = f'p{variant}@{k}'
                        rows = [r for r in passive_records if r['label'] == label]
                        active_rows = [r for r in active_batch['records'] if r['label'] == label]
                        eligible = (sum(r['passive_get_up'] for r in rows) <= PASSIVE_GATES['max_passive_get_ups']
                                    and max(r['passive_return'] for r in rows) <= PASSIVE_GATES['max_passive_return']
                                    and max(r['passive_shaping_credit'] for r in rows) <= PASSIVE_GATES['max_passive_shaping_credit']
                                    and not any(r['fault_in_window'] or r['fault_in_prefix'] for r in rows))
                        grid[label] = {'variant': variant, 'prefix_steps': k, 'replicas': len(rows),
                                       'handover_upright': stats([r['handover_upright'] for r in rows]),
                                       'passive_get_ups': sum(r['passive_get_up'] for r in rows),
                                       'max_passive_return': max(r['passive_return'] for r in rows),
                                       'max_passive_shaping_credit': max(r['passive_shaping_credit'] for r in rows),
                                       'faults': sum(r['fault_in_window'] or r['fault_in_prefix'] for r in rows),
                                       'eligible': eligible,
                                       'frozen_policy_successes': sum(r['success'] for r in active_rows),
                                       'frozen_policy_stable_upright': sum(r['stable_upright_time_s'] is not None for r in active_rows),
                                       'frozen_policy_replicas': len(active_rows)}
                side_entry['grid'] = grid
                side_entry['eligible_handover_points_after_start'] = sum(g['eligible'] and g['prefix_steps'] > 0 for g in grid.values())
                save()

        # Pre-registered Part 2 gate with the confirmed amendment.
        base = summary['1']
        gate = {'rule': 'Smallest k in {2,3}: (i) stable-upright fraction under perturbation >= +15 pp vs 1.0 on one side '
                        'and no drop > 5 pp on the other; (ii) confirmed-script mean discounted 30 s return beats hold on both '
                        'sides. Amendment: if (i) passes and (ii) fails only because of command_rate (both without that term '
                        'and with (rate/slew)^2 normalization the script beats hold on both sides), Part 2 runs normalized.',
                'candidates': {}, 'part2_slew_rad_s': None, 'normalize_command_rate': False}
        for slew in args.slews[1:]:
            here = summary[f'{slew:g}']
            diff = {s: here[s]['basin']['stable_upright_fraction'] - base[s]['basin']['stable_upright_fraction'] for s in ('left', 'right')}
            i_pass = any(diff[s] >= .15 and diff[o] >= -.05 for s, o in (('left', 'right'), ('right', 'left')))
            ii_pass = all(here[s]['returns']['script_beats_hold'] for s in ('left', 'right'))
            without = all(here[s]['returns']['script_without_command_rate'] > here[s]['returns']['hold_without_command_rate'] for s in ('left', 'right'))
            normalized = all(here[s]['returns']['script_normalized_command_rate'] > here[s]['returns']['hold_normalized_command_rate'] for s in ('left', 'right'))
            amended = i_pass and not ii_pass and without and normalized
            gate['candidates'][f'{slew:g}'] = {'stable_upright_difference': diff, 'i_pass': i_pass, 'ii_pass': ii_pass,
                                               'ii_without_command_rate': without, 'ii_normalized': normalized,
                                               'passes': i_pass and (ii_pass or amended), 'requires_normalization': amended}
            if gate['part2_slew_rad_s'] is None and i_pass and (ii_pass or amended):
                gate['part2_slew_rad_s'], gate['normalize_command_rate'] = slew, amended
        gate['decision'] = ('Part 2 runs at the selected slew.' if gate['part2_slew_rad_s'] is not None
                            else 'No slew passes: keep 1.0 rad/s and drop Part 2.')
        report['part2_gate'] = gate
        phase0 = {side: summary['1'][side]['eligible_handover_points_after_start'] for side in ('left', 'right')}
        report['phase0_physics_grid'] = {'contract_slew_rad_s': 1., 'eligible_points_after_start': phase0,
                                         'passes': all(v > 0 for v in phase0.values()),
                                         'note': 'Physics grid only; mask/accounting implementation checks are separate.'}
        report['complete'] = True
        save()
        print(json.dumps({'part2_gate': gate['decision'], 'part2_slew': gate['part2_slew_rad_s'],
                          'normalize': gate['normalize_command_rate'], 'phase0_physics_grid': report['phase0_physics_grid']}), flush=True)
    print('Slew/handover sweep complete', flush=True)
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
