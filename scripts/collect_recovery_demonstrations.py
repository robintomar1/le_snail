#!/usr/bin/env python3
"""Collect original-start actor24 demonstrations; no learning or hidden actor inputs."""
import argparse
from pathlib import Path
import time
import traceback
import sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--physics', default='configs/experiments/physics-v2-002.json')
p.add_argument('--contract', default='configs/policy.json')
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--confirmation', default='runs/feasibility-side-recovery-20261014T111801Z/results.json')
p.add_argument('--policy', default='runs/recover-shaped-051-100/exported/policy.pt')
p.add_argument('--max-process-seconds', type=float, default=1100.)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new')
started = time.monotonic()
args.output.mkdir(parents=True)
app = AppLauncher(args).app
env = None
try:
    import json
    import numpy as np
    import torch
    from le_snail.bundle import file_hash
    from le_snail.evaluation import Trial
    from le_snail.sim.env import SnailEnv, make_config

    class CaptureEnv(SnailEnv):
        def _get_dones(self):
            result = super()._get_dones()
            self.recorded_target = self.target.clone()
            self.recorded_applied = self.applied_target.clone()
            return result

    N, seed = 64, 4242
    cfg = make_config(args.asset, args.contract, args.physics, num_envs=N, stage='recover',
                      evaluation=True, initial_side='left', device=args.device or 'cuda:0', seed=seed)
    env = CaptureEnv(cfg)
    device, dt = env.device, env.step_dt
    policy = torch.jit.load(args.policy, map_location=device).eval()
    source = json.loads(Path(args.confirmation).read_text())
    perturb = np.random.default_rng(seed).uniform(-.1, .1, (64, 3, 6))
    report = {'schema_version': 1, 'actor_observation_size': 24, 'policy_dt': dt, 'num_envs': N,
              'training_performed': False, 'groups': [], 'configuration': env.physics,
              'provenance': {str(f): file_hash(f) for f in (args.asset, args.contract, args.physics, args.confirmation, args.policy, __file__, 'src/le_snail/sim/env.py')},
              'selection_preregistered': 'Keep raw complete30s traces and outcomes. Training side pool uses successful teacher prefix through recovery+5cm acceptance, balanced across4variants. Upright preservation pool uses frozen policy prefix through crawl acceptance. Failures remain recorded and are excluded from supervised positive demonstrations. Phase/variant/rootstate are metadata only, never actor input.',
              'ambiguity_preregistered': {'q_rad': .01, 'filtered_qd_rad_s': .05, 'gravity_component': .01, 'gyro_rad_s': .05, 'previous_applied_target_rad': .01, 'material_next_applied_target_difference_rad': .025,
                                         'rule': 'Compare all dimensions within their stated bounds; stop for unexplained conflicting next applied commands. Absence of observed collisions does not prove representability.'}}
    def save():
        report['process_elapsed_seconds'] = time.monotonic() - started
        report['speed_diagnostics'] = env.speed_diagnostics()
        report['action_diagnostics'] = env.action_diagnostics() if hasattr(env, 'action_diagnostics') else None
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    schedule = [(side, repeat) for repeat in range(2) for side in ('left', 'right')]
    schedule += [('upright', repeat) for repeat in range(4)]
    with torch.no_grad():
        for side, repeat in schedule:
            env.cfg.initial_side = side
            obs, _ = env.reset(seed=seed+repeat)
            reset_pos = env.robot.data.root_pos_w.clone()
            reset_budget = env.best_upright_progress.clone()
            settle_return = torch.zeros(N, device=device)
            for _ in range(round(2/dt)):
                obs, reward, term, trunc, _ = env.step(torch.zeros((N, 6), device=device))
                assert not (term | trunc).any(), 'Settling fault'
                settle_return += reward
            drift = (env.robot.data.root_pos_w[:, :2]-reset_pos[:, :2]).norm(dim=1).max().item()
            passive_credit = (env.best_upright_progress-reset_budget).max().item()
            assert drift <= 1e-4 and passive_credit <= 1e-4 and settle_return.max() <= 0, (drift, passive_credit, settle_return.max())
            start = env.robot.data.root_pos_w.clone()
            env.start_pos.copy_(start)
            env.episode_length_buf.zero_()
            env.fallen_time.zero_()
            selected, variant_ids = {}, [-1]*N
            if side != 'upright':
                variants = {'left': [9, 34], 'right': [48, 62]}[side]
                for variant in variants:
                    matches = [r for r in source['results'] if r['stage']=='recover' and r['side']==side and r['variant']==f'perturbed{variant}' and r['effort_nm']==.8 and r['vel_limit']==2. and r['success']]
                    assert matches
                    selected[matches[0]['env']] = variant
                variant_ids = [variants[i % 2] for i in range(N)]
                for index, variant in selected.items(): variant_ids[index] = variant
                keys = env.tensor(np.array([np.vstack((np.array(source['keys'][side])+perturb[v], env.contract.home_rad)) for v in variant_ids]))
            key_i = torch.zeros(N, dtype=torch.long, device=device)
            held = torch.zeros_like(key_i)
            active = torch.ones(N, dtype=torch.bool, device=device)
            ar = torch.arange(N, device=device)
            trials = [Trial(kind='crawl' if side=='upright' else side) for _ in range(N)]
            names = ('observations','desired_action','pre_target','pre_applied','pre_queue','post_target','post_applied','joint_q','joint_qd','filtered_velocity','previous_q','root_state','phase','active','reward','terminal')
            rows = {key: [] for key in names}
            max_target_error = max_applied_error = 0.
            for step in range(1, round(30/dt)+1):
                assert time.monotonic()-started < args.max_process_seconds, 'Dataset process budget exhausted'
                observation = obs['policy'].clone()
                assert observation.shape == (N,24)
                before = env.target.clone()
                applied = env.applied_target.clone()
                queue = env.command_queue.clone()
                q = env.robot.data.joint_pos[:, env.joint_ids].clone()
                qd = env.robot.data.joint_vel[:, env.joint_ids].clone()
                filtered = env.filtered_velocity.clone()
                previous_q = env.previous_q.clone()
                root = env.robot.data.root_state_w.clone()
                torch.testing.assert_close(observation[:, :6], q-env.home, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(observation[:, 6:12], filtered, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(observation[:, 18:], (applied-env.home)/env.scale, rtol=1e-5, atol=1e-5)
                phase = key_i.clone() if side!='upright' else torch.full_like(key_i, -1)
                if side == 'upright':
                    actions = policy(observation)
                else:
                    goal = keys[ar, key_i.clamp(max=3)]
                    script_actions = ((goal-env.home)/env.scale).clamp(-1,1)
                    actions = torch.where((key_i<=3)[:,None], script_actions, policy(observation))
                expected_desired = (env.home+actions.clamp(-1,1)*env.scale).clamp(env.lower,env.upper)
                expected = before+(expected_desired-before).clamp(-env.slew,env.slew)
                live = active.clone()
                obs, reward, term, trunc, _ = env.step(actions)
                torch.testing.assert_close(env.recorded_target, expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(env.recorded_applied, expected, rtol=1e-5, atol=1e-6)
                max_target_error = max(max_target_error, float((env.recorded_target-expected).abs().max()))
                max_applied_error = max(max_applied_error, float((env.recorded_applied-expected).abs().max()))
                assert torch.isfinite(observation).all() and torch.isfinite(actions).all() and torch.isfinite(reward).all()
                if side != 'upright':
                    clipped = (env.home+script_actions*env.scale).clamp(env.lower,env.upper)
                    reached = (env.recorded_target-clipped).abs().amax(dim=1)<1e-3
                    held = torch.where(reached & (key_i<=3), held+1, 0)
                    advance = (key_i<=3) & (held>=torch.where(key_i==3,20,2))
                    key_i += advance.long()
                    held = torch.where(advance,0,held)
                pos, gravity, _ = env.last_transition
                disp = (pos-start).cpu().numpy()
                upright = (-gravity[:,2]).cpu().numpy()
                terminal = term | trunc
                for index, trial in enumerate(trials):
                    if bool(live[index]): trial.update(step*dt,float(disp[index,0]),float(disp[index,1]),float(upright[index]),fault=bool(terminal[index]))
                active &= ~terminal
                values = (observation,actions,before,applied,queue,env.recorded_target,env.recorded_applied,q,qd,filtered,previous_q,root,phase,live,reward,terminal)
                for name,value in zip(names,values): rows[name].append(value.cpu().numpy())
            stem = f'{side}-repeat{repeat}'
            np.savez_compressed(args.output/f'{stem}.npz', **{key:np.asarray(value) for key,value in rows.items()})
            group={'stem':stem,'side':side,'repeat':repeat,'variants':variant_ids,'confirmed_indices':selected,
                   'settle_drift_m':drift,'passive_shaping_credit':passive_credit,'max_target_error':max_target_error,'max_applied_error':max_applied_error,
                   'successes':sum(t.success for t in trials),'trials':[{'env':i,'success':t.success,'reason':t.reason,'recovery_time_s':t.recovery_time_s,'acceptance_end_s':t.last_time_s}for i,t in enumerate(trials)]}
            report['groups'].append(group)
            save()
            print(json.dumps({'group':stem,'successes':group['successes'],'elapsed_s':report['process_elapsed_seconds']}),flush=True)
    report['collection_complete']=True
    report['positive_control_passed']=all(all(group['trials'][int(index)]['success']for index in group['confirmed_indices'])for group in report['groups']if group['side']!='upright')
    save()
    assert report['positive_control_passed'], 'Repeated exact source-path replay failed'
    print('Demonstration collection complete; independent CPU ambiguity gate still required',flush=True)
except BaseException:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    if env is not None: env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
