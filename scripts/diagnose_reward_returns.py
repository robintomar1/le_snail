#!/usr/bin/env python3
"""Read-only reward/critic audit on nominal original side starts; never train."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--capture-physics-steps', action='store_true', help='Read-only200Hz joint/drive/contact diagnostic; no changes to dynamics')
p.add_argument('--solver-velocity-iterations', type=int, choices=range(1, 33), help='Diagnostic-only numerical override; never changes production config')
p.add_argument('--modes', nargs='+', choices=('hold', 'script', 'deterministic', 'stochastic'), default=['hold', 'script', 'deterministic', 'stochastic'])
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--contract', default='configs/policy.json')
p.add_argument('--physics', default='configs/experiments/recover-upright-progress-001.json')
p.add_argument('--confirmation', default='runs/feasibility-side-recovery-20261014T111801Z/results.json')
p.add_argument('--checkpoint', default='runs/recover-shaped-051-100/model_final.pt')
p.add_argument('--crawl-policy', default='runs/scale-treatment-051-100/exported/policy.pt')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new')
args.output.mkdir(parents=True)
app = AppLauncher(args).app
env = None
try:
    import numpy as np
    import torch
    from rsl_rl.modules import ActorCritic
    from le_snail.bundle import file_hash
    from le_snail.evaluation import Trial
    from le_snail.sim.env import SnailEnv, make_config

    seed, N, gamma = 4242, 64, .995
    source = json.loads(Path(args.confirmation).read_text())
    perturb = np.random.default_rng(seed).uniform(-.1, .1, (64, 3, 6))
    class AuditEnv(SnailEnv):
        capture_enabled = False
        def _setup_scene(self):
            super()._setup_scene()
            if args.capture_physics_steps:
                from isaaclab.sensors import ContactSensor, ContactSensorCfg
                self.audit_contacts = ContactSensor(ContactSensorCfg(
                    prim_path='/World/envs/env_.*/Robot/.*', update_period=0.))
                self.scene.sensors['audit_contacts'] = self.audit_contacts
        def _apply_action(self):
            super()._apply_action()
            if self.capture_enabled:
                q = self.robot.data.joint_pos[:, self.joint_ids]
                qd = self.robot.data.joint_vel[:, self.joint_ids]
                pd = self.physics['stiffness_nm_rad'] * (self.applied_target-q) - self.physics['damping_nm_s_rad'] * qd
                contacts = self.audit_contacts.data.net_forces_w.norm(dim=-1)
                states = torch.cat((q, qd, self.applied_target, pd, contacts,
                                    -self.robot.data.projected_gravity_b[:, 2:3],
                                    self.audit_active[:, None], self.audit_script[:, None],
                                    self.audit_accepted[:, None]), dim=1)
                self.audit_rows.append(states.cpu().numpy())
    cfg = make_config(args.asset, args.contract, args.physics, num_envs=N,
                      stage='recover', device=args.device or 'cuda:0',
                      evaluation=True, initial_side='left', seed=seed)
    if args.capture_physics_steps:
        cfg.robot_cfg.spawn.activate_contact_sensors = True
    if args.solver_velocity_iterations is not None:
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = args.solver_velocity_iterations
    env = AuditEnv(cfg)
    device, dt = env.device, env.step_dt
    assert torch.allclose(env.robot.data.joint_effort_limits, torch.full_like(env.robot.data.joint_effort_limits, env.physics['effort_limit_nm']))
    assert torch.allclose(env.robot.data.joint_vel_limits, torch.full_like(env.robot.data.joint_vel_limits, env.physics['velocity_limit_rad_s']))
    runner = json.loads((Path(args.checkpoint).parent / 'run.json').read_text())['runner']
    pc = dict(runner['policy'])
    assert pc.pop('class_name') == 'ActorCritic'
    net = ActorCritic({'policy': torch.zeros((1, 24), device=device)}, runner['obs_groups'], num_actions=6, **pc).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)['model_state_dict'])
    net.eval()
    crawl = torch.jit.load(args.crawl_policy, map_location=device).eval()
    report = {'seed': seed, 'num_envs': N, 'gamma': gamma, 'dt': dt,
              'physics_validated': False, 'training_performed': False,
              'numerical_settings': {'physics_dt': env.physics_dt, 'position_iterations': cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count, 'velocity_iterations': cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count},
              'configured_effort_limit_nm': env.physics['effort_limit_nm'],
              'configured_velocity_limit_rad_s': env.physics['velocity_limit_rad_s'], 'script_continuation_policy': args.crawl_policy,
              'provenance': {str(f): file_hash(f) for f in (args.asset, args.contract, args.physics, args.confirmation, args.checkpoint, args.crawl_policy, __file__, 'src/le_snail/sim/env.py')},
              'method': 'Original settled resets, two-second excluded home settling; same30s horizon; environment termination becomes absorbing zero reward, never includes auto-reset reward. Acceptance success does not stop reward collection. Scripted arm uses confirmed paths then the recorded continuation policy. Critic is shaped policy value; script/hold returns are off-policy diagnostics, not critic calibration targets.',
              'groups': []}
    def save():
        report['speed_diagnostics'] = env.speed_diagnostics()
        assert report['speed_diagnostics']['physics_steps'] == env._sim_step_counter
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with torch.no_grad():
        for side, variants in {'left': [9, 34], 'right': [48, 62]}.items():
            selected = {}
            for variant in variants:
                matches = [r for r in source['results'] if r['stage'] == 'recover' and r['side'] == side and r['variant'] == f'perturbed{variant}' and r['effort_nm'] == .8 and r['vel_limit'] == 2. and r['success']]
                assert matches
                selected[matches[0]['env']] = variant
            # Repeated variants across64 origins; exact confirmed source indices preserved.
            variant_ids = [variants[i % len(variants)] for i in range(N)]
            for i, variant in selected.items():
                variant_ids[i] = variant
            keys = env.tensor(np.array([np.vstack((np.array(source['keys'][side]) + perturb[v], env.contract.home_rad)) for v in variant_ids]))
            for mode in args.modes:
                env.cfg.initial_side = side
                env.capture_enabled = False
                obs, _ = env.reset(seed=seed)
                for _ in range(round(2 / dt)):
                    obs, *_ = env.step(torch.zeros((N, 6), device=device))
                start = env.robot.data.root_pos_w.clone()
                env.start_pos.copy_(start)
                env.episode_length_buf.zero_()
                env.fallen_time.zero_()
                # Preserve tiny spent shaping budget from excluded settling, as in ordinary evaluation.
                initial_value = net.evaluate(obs).flatten().cpu().numpy()
                initial_budget = env.best_upright_progress.cpu().numpy().copy()
                active = torch.ones(N, device=device, dtype=torch.bool)
                key_i = torch.zeros(N, device=device, dtype=torch.long)
                held = torch.zeros_like(key_i)
                trials = [Trial(kind=side) for _ in range(N)]
                rows, terms, actions_raw, deltas, values, masks = [], [], [], [], [], []
                max_speed = torch.zeros(N, device=device)
                max_effort = torch.zeros(N, device=device)
                end_time = np.full(N, 30.)
                script_end = np.full(N, np.nan)
                ar = torch.arange(N, device=device)
                env.audit_rows = []
                for step in range(1, round(30 / dt) + 1):
                    before = env.target.clone()
                    value = net.evaluate(obs).flatten()
                    if mode == 'hold':
                        actions = torch.zeros((N, 6), device=device)
                    elif mode == 'deterministic':
                        actions = net.act_inference(obs)
                    elif mode == 'stochastic':
                        actions = net.act(obs)
                    else:
                        goal = keys[ar, key_i.clamp(max=3)]
                        script_actions = ((goal - env.home) / env.scale).clamp(-1, 1)
                        actions = torch.where((key_i <= 3)[:, None], script_actions, crawl(obs['policy']))
                    desired = torch.clamp(env.home + actions.clamp(-1, 1) * env.scale, env.lower, env.upper)
                    intended_delta = (desired - before).clamp(-env.slew, env.slew)
                    live = active.clone()
                    env.audit_active = live
                    env.audit_script = (key_i <= 3) if mode == 'script' else torch.zeros_like(live)
                    env.audit_accepted = torch.tensor([t.success for t in trials], device=device)
                    env.capture_enabled = args.capture_physics_steps
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    comps = torch.stack(list(env.reward_components.values()), dim=1)
                    # Sampled after each20Hz control step; solver caps are separately
                    # verified by the physics audit, not inferred from these peaks.
                    max_speed = torch.maximum(max_speed, env.robot.data.joint_vel.abs().amax(dim=1) * live)
                    max_effort = torch.maximum(max_effort, env.robot.data.applied_torque.abs().amax(dim=1) * live)
                    torch.testing.assert_close(comps.sum(dim=1), reward)
                    assert torch.isfinite(comps).all() and torch.isfinite(value).all()
                    if mode == 'script':
                        clipped = torch.clamp(env.home + script_actions * env.scale, env.lower, env.upper)
                        reached = (env.target - clipped).abs().amax(dim=1) < 1e-3
                        held = torch.where(reached & (key_i <= 3), held + 1, 0)
                        advance = (key_i <= 3) & (held >= torch.where(key_i == 3, 20, 2))
                        key_i += advance.long()
                        held = torch.where(advance, 0, held)
                        finished_script = advance & (key_i > 3) & live
                        script_end[finished_script.cpu().numpy()] = step * dt
                    pos, gravity, _ = env.last_transition
                    disp = pos - start
                    up = -gravity[:, 2]
                    term = terminated | truncated
                    lc = live.cpu().numpy()
                    pcpu, uc, tc = disp.cpu().numpy(), up.cpu().numpy(), term.cpu().numpy()
                    for i, trial in enumerate(trials):
                        if lc[i]:
                            trial.update(step * dt, float(pcpu[i, 0]), float(pcpu[i, 1]), float(uc[i]), fault=bool(tc[i]))
                    newly = live & term
                    end_time[newly.cpu().numpy()] = step * dt
                    active &= ~term
                    rows.append(torch.cat((disp, up[:, None], term[:, None]), dim=1).cpu().numpy())
                    terms.append((comps * live[:, None]).cpu().numpy())
                    actions_raw.append(actions.cpu().numpy())
                    deltas.append(intended_delta.cpu().numpy())
                    values.append(value.cpu().numpy())
                    masks.append(lc)
                term_names = list(env.reward_components)
                rewards = np.asarray(terms)
                raw, delta, mask = np.asarray(actions_raw), np.asarray(deltas), np.asarray(masks)
                trace = np.asarray(rows)
                value_trace = np.asarray(values)
                weights = gamma ** np.arange(len(rewards))
                summaries = []
                for i, trial in enumerate(trials):
                    prefix = {}
                    for horizon in (3.2, 4., 10., 30.):
                        count = round(horizon / dt)
                        sums = rewards[:count, i].sum(axis=0)
                        discounted = (rewards[:count, i] * weights[:count, None]).sum(axis=0)
                        prefix[str(horizon)] = {'undiscounted': dict(zip(term_names, sums.tolist())), 'discounted': dict(zip(term_names, discounted.tolist())), 'total': float(sums.sum()), 'discounted_total': float(discounted.sum())}
                    live = mask[:, i]
                    summaries.append({'env': i, 'variant': variant_ids[i] if mode == 'script' else None,
                                      'confirmed_source_index': i in selected if mode == 'script' else False,
                                      'success': trial.success, 'reason': trial.reason,
                                      'recovery_time_s': trial.recovery_time_s, 'acceptance_end_s': trial.last_time_s,
                                      'environment_end_s': float(end_time[i]),
                                      'script_end_s': float(script_end[i]) if np.isfinite(script_end[i]) else None,
                                      'max_sampled_joint_speed_rad_s': float(max_speed[i]),
                                      'max_sampled_estimated_effort_nm': float(max_effort[i]),
                                      'initial_critic': float(initial_value[i]), 'initial_spent_shaping': float(initial_budget[i]),
                                      'max_upright': float(trace[live, i, 3].max()),
                                      'action_outside_unit_fraction': float((np.abs(raw[live, i]) > 1).mean()),
                                      'slew_saturated_fraction': float((np.abs(delta[live, i]) >= .05 - 1e-6).mean()),
                                      'mean_abs_target_speed_rad_s': float(np.abs(delta[live, i]).mean() / dt),
                                      'prefix_returns': prefix})
                group = {'side': side, 'mode': mode, 'term_names': term_names, 'trials': summaries,
                         'successes': sum(t.success for t in trials),
                         'recovered': sum(t.recovery_time_s is not None for t in trials)}
                report['groups'].append(group)
                env.capture_enabled = False
                if args.capture_physics_steps:
                    group['physics_trace_columns'] = {'joint_position': [0, 6], 'joint_velocity': [6, 12],
                                                      'applied_target': [12, 18], 'unclipped_pd_estimate': [18, 24],
                                                      'body_contact_force_norm': [24, 24+env.robot.num_bodies],
                                                      'upright_active_script_accepted': [24+env.robot.num_bodies, 28+env.robot.num_bodies]}
                    group['physics_dt'] = env.physics_dt
                    group['contact_body_names'] = env.audit_contacts.body_names
                    np.savez_compressed(args.output / f'{side}-{mode}-physics.npz', samples=np.asarray(env.audit_rows))
                np.savez_compressed(args.output / f'{side}-{mode}.npz', components=rewards, state=trace, raw_action=raw, intended_target_delta=delta, critic=value_trace, active=mask)
                save()
                print(json.dumps({'side': side, 'mode': mode, 'successes': group['successes'], 'recovered': group['recovered'], 'mean_discounted30': float(np.mean([s['prefix_returns']['30.0']['discounted_total'] for s in summaries])), 'mean_critic': float(initial_value.mean())}), flush=True)
    print('Reward audit complete', flush=True)
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
