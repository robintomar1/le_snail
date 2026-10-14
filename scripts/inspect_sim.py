#!/usr/bin/env python3
"""Record simulation inspection evidence without changing measured-validation flags.

Run inside Isaac with --headless --device cuda:0 --output NEW_DIRECTORY.
Use --checks-only for the deterministic control-contract regression checks.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset', default='assets/so101/snail.urdf')
parser.add_argument('--contract', default='configs/policy.json')
parser.add_argument('--physics', default='configs/sim.json')
parser.add_argument('--stage', choices=['crawl', 'recover'], default='crawl')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--checks-only', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error('--output must be new; inspection evidence is never overwritten')
args.enable_cameras = not args.checks_only
args.output.mkdir(parents=True)
app = AppLauncher(args).app

try:
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils.math import quat_apply, quat_mul
    from le_snail.contract import VelocityEstimator, rotation_matrix
    from le_snail.sim.env import SnailEnv, make_config

    class InspectionEnv(SnailEnv):
        def _setup_scene(self):
            super()._setup_scene()
            self.contacts = ContactSensor(ContactSensorCfg(
                prim_path='/World/envs/env_.*/Robot/.*', update_period=0.,
            ))
            self.scene.sensors['inspection_contacts'] = self.contacts

    cfg = make_config(args.asset, args.contract, args.physics, num_envs=2,
                      stage=args.stage, evaluation=True, device=args.device or 'cuda:0')
    cfg.robot_cfg.spawn.activate_contact_sensors = True
    cfg.viewer.resolution = (960, 720)
    env = InspectionEnv(cfg, render_mode=None if args.checks_only else 'rgb_array')
    zero = torch.zeros((env.num_envs, 6), device=env.device)
    report = {'checks': [], 'poses': {}, 'joint_probes': {}, 'tether_probes': [],
              'physics_validated': False, 'visual_review_required': True}

    def array(value):
        return value.detach().cpu().numpy()

    def check(name, passed, **evidence):
        report['checks'].append({'name': name, 'passed': bool(passed), **evidence})
        print(f'{"PASS" if passed else "FAIL"}: {name} {evidence}', flush=True)

    def state():
        return {'root_pos_m': array(env.robot.data.root_pos_w - env.scene.env_origins).tolist(),
                'body_link_pose_w': array(env.robot.data.body_link_pose_w).tolist(),
                'upright_cos': array(-env.robot.data.projected_gravity_b[:, 2]).tolist(),
                'joint_rad': array(env.robot.data.joint_pos[:, env.joint_ids]).tolist(),
                'joint_velocity_rad_s': array(env.robot.data.joint_vel[:, env.joint_ids]).tolist(),
                'root_velocity': array(env.robot.data.root_vel_w).tolist(),
                'contact_normal_forces_n': array(env.contacts.data.net_forces_w).tolist()}

    # Public step outputs are compared with the actual targets sent to the drives.
    for delay in (0, 1):
        obs, _ = env.reset(seed=7)
        env.delay_steps[:] = delay
        estimator = VelocityEstimator(env.contract.velocity_filter_alpha)
        initial = estimator.update(array(env.robot.data.joint_pos[0, env.joint_ids]), 0.)
        check(f'reset_velocity_delay_{delay}', np.allclose(array(obs['policy'][0, 6:12]), initial, atol=1e-6))
        action_error = observation_error = 0.
        for step, amplitude in enumerate((0.4, -0.4, 0.2), 1):
            actions = torch.full_like(zero, amplitude)
            obs, reward, term, trunc, _ = env.step(actions)
            applied = array(env.robot.data.joint_pos_target[0, env.joint_ids])
            expected_action = env.contract.applied_action(applied)
            if step == 1:
                check(f'first_applied_target_delay_{delay}',
                      np.allclose(applied, env.contract.home_rad if delay else
                                  env.contract.position_target([amplitude]*6, env.contract.home_rad), atol=1e-6),
                      applied_target_rad=applied.tolist(),
                      observation_previous_action=array(obs['policy'][0, 18:]).tolist())
            action_error = max(action_error, float(np.max(np.abs(array(obs['policy'][0, 18:]) - expected_action))))
            q = array(env.robot.data.joint_pos[0, env.joint_ids])
            velocity = estimator.update(q, step * env.step_dt)
            expected_obs = env.contract.observation(q, velocity,
                array(env.robot.data.projected_gravity_b[0]), array(env.robot.data.root_ang_vel_b[0]), applied)
            observation_error = max(observation_error, float(np.max(np.abs(array(obs['policy'][0]) - expected_obs))))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            assert not bool(term.any() or trunc.any()), 'Unexpected reset during contract probe'
        check(f'applied_action_delay_{delay}', action_error < 1e-5, max_error=action_error)
        check(f'portable_observation_delay_{delay}', observation_error < 1e-5, max_error=observation_error)

    env.reset(seed=7)
    env.delay_steps[:] = 1
    estimator = VelocityEstimator(env.contract.velocity_filter_alpha)
    estimator.update(array(env.robot.data.joint_pos[1, env.joint_ids]), 0.)
    obs, *_ = env.step(torch.full_like(zero, .3))
    estimator.update(array(env.robot.data.joint_pos[1, env.joint_ids]), env.step_dt)
    pose = env.robot.data.root_link_pose_w[:1].clone()
    pose[:, 1] += .3  # Only environment zero crosses the lane and auto-resets.
    env.robot.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.device))
    obs, _, term, _, _ = env.step(torch.full_like(zero, -.3))
    velocity = estimator.update(array(env.robot.data.joint_pos[1, env.joint_ids]), 2*env.step_dt)
    check('partial_reset_history', bool(term[0] and not term[1]) and
          np.allclose(array(obs['policy'][0, 18:]), 0., atol=1e-6) and
          np.allclose(array(obs['policy'][1, 18:]), env.contract.applied_action(
              array(env.robot.data.joint_pos_target[1, env.joint_ids])), atol=1e-6))
    check('partial_reset_velocity', np.allclose(array(obs['policy'][0, 6:12]), 0., atol=1e-6) and
          np.allclose(array(obs['policy'][1, 6:12]), velocity, atol=1e-5))

    check('terminal_snapshot_precedes_reset',
          float((env.last_transition[0][0] - env.scene.env_origins[0])[1]) > .15 and
          abs(float((env.robot.data.root_pos_w[0] - env.scene.env_origins[0])[1])) < .01)
    # Exercise public steps at the actual timeout boundary without simulating
    # two minutes of idle physics. The upstream step increments this counter.
    env.reset(seed=7)
    env.episode_length_buf[:] = env.max_episode_length - 2
    _, _, term, trunc, _ = env.step(zero)
    check('not_truncated_before_120s', not bool(trunc.any() or term.any()),
          scored_time_s=(env.max_episode_length - 1) * env.step_dt)
    env.episode_length_buf[:] = env.max_episode_length - 1
    _, _, term, trunc, _ = env.step(zero)
    check('truncated_exactly_at_120s', bool(trunc.all()) and not bool(term.any()),
          scored_time_s=env.max_episode_length * env.step_dt)

    if 'settled_starts' in env.physics:
        for side, rest in env.physics['settled_starts'].items():
            env.cfg.initial_side = side
            obs, _ = env.reset(seed=7)
            check(f'settled_reset_{side}', np.allclose(
                array(env.robot.data.root_link_pose_w[0])[:3] - array(env.scene.env_origins[0]),
                rest['root_pose'][:3], atol=1e-6) and np.allclose(
                array(env.robot.data.joint_pos[0, env.joint_ids]), rest['joint_rad'], atol=1e-6))
            upright = -env.robot.data.projected_gravity_b[:, 2]
            posture_ok = (upright > .8).all() if side == 'upright' else (upright < .2).all()
            check(f'settled_posture_{side}', bool(posture_ok), upright_cos=array(upright).tolist())
            start = env.robot.data.root_pos_w.clone()
            total = torch.zeros(env.num_envs, device=env.device)
            shaping_total = torch.zeros_like(total)
            fault = False
            for _ in range(40):
                _, reward, term, trunc, _ = env.step(zero)
                total += reward
                shaping_total += env.reward_components.get('upright_progress', 0.)
                fault |= bool(term.any() or trunc.any())
            drift = float((env.robot.data.root_pos_w[:, :2] - start[:, :2]).abs().max())
            check(f'no_passive_progress_{side}', not fault and drift < 1e-4 and bool((total <= 0).all()),
                  max_drift_m=drift, max_reward=float(total.max()))
            if env.physics.get('side_upright_progress_reward_cap', 0.) > 0:
                check(f'no_passive_upright_shaping_{side}', bool((shaping_total <= 1e-4).all()),
                      max_added_reward=float(shaping_total.max()))
        check('reward_components_logged', 'Reward/progress' in env.extras.get('log', {}) and
              'Reward/gated_out_progress' in env.extras.get('log', {}))
        if env.physics.get('side_upright_progress_reward_cap', 0.) > 0:
            cap = env.physics['side_upright_progress_reward_cap']

            def posture_credit(upright, sign=1):
                # Controlled state intervention tests the real reward path;
                # this is a reward invariant check, not physical recovery evidence.
                angle = sign * np.arccos(upright)
                pose = env.robot.data.root_link_pose_w.clone()
                pose[:, 3:] = env.tensor([np.cos(angle/2), np.sin(angle/2), 0., 0.])
                env.robot.write_root_pose_to_sim(pose)
                env._get_rewards()
                return env.reward_components['upright_progress'].clone()

            for side, sign in (('left', 1), ('right', -1)):
                env.cfg.initial_side = side
                env.reset(seed=7)
                env._get_rewards()
                check(f'upright_shaping_no_reset_credit_{side}',
                      bool((env.reward_components['upright_progress'].abs() < 1e-6).all()))
                total = torch.zeros(env.num_envs, device=env.device)
                credits = []
                for upright in (.1, .3, -.2, .1, .3, .6, .8, 1.):
                    credit = posture_credit(upright, sign)
                    credits.append(credit)
                    total += credit
                check(f'upright_shaping_monotone_budget_{side}',
                      torch.allclose(total, torch.full_like(total, cap), atol=1e-6) and
                      bool((torch.stack(credits) >= 0).all()) and
                      bool((credits[0] > 0).all() and (credits[1] > 0).all()),
                      total_reward=array(total).tolist())
                check(f'upright_shaping_repeated_partial_rise_{side}',
                      bool((torch.stack(credits[2:5]).abs() < 1e-6).all()))
                repeated = sum(posture_credit(u, sign) for _ in range(100) for u in (-.2, .8, 1.))
                check(f'upright_shaping_no_fall_recovery_refill_{side}',
                      bool((repeated.abs() < 1e-6).all()), repeated_reward=array(repeated).tolist())

            env.cfg.initial_side = 'upright'
            env.reset(seed=7)
            total = sum(posture_credit(u) for u in (-.2, .1, .5, .8, 1., -.2, 1.))
            check('upright_shaping_upright_start_ineligible_after_fall', bool((total == 0).all()))

            env.cfg.initial_side = 'left'
            env.reset(seed=7)
            earned = posture_credit(.4)
            env._reset_idx(torch.tensor([0], device=env.device))
            env._get_rewards()
            at_reset = env.reward_components['upright_progress'].clone()
            remaining = posture_credit(.8)
            check('upright_shaping_partial_reset_budget',
                  bool((at_reset.abs() < 1e-6).all()) and
                  abs(float(remaining[0]) - cap) < 1e-6 and
                  abs(float(earned[1] + remaining[1]) - cap) < 1e-6,
                  before_reset=array(earned).tolist(), after_reset=array(remaining).tolist())

            env.reset(seed=7)
            posture_credit(.3)
            observation = env._get_observations()['policy'].clone()
            applied = env.applied_target.clone()
            env.physics['side_upright_progress_reward_cap'] = 0.
            try:
                baseline = env._get_rewards().clone()
                components = {k: v.clone() for k, v in env.reward_components.items()}
                gated_out = env.extras['log']['Reward/gated_out_progress'].clone()
            finally:
                env.physics['side_upright_progress_reward_cap'] = cap
            shaped = env._get_rewards()
            check('upright_shaping_preserves_existing_rewards_and_actor',
                  'upright_progress' not in components and torch.equal(baseline, shaped) and
                  all(torch.equal(value, env.reward_components[key]) for key, value in components.items()) and
                  torch.equal(gated_out, env.extras['log']['Reward/gated_out_progress']) and
                  torch.equal(observation, env._get_observations()['policy']) and
                  torch.equal(applied, env.applied_target))
        if args.stage == 'recover':
            counts = {'upright': 0, 'left': 0, 'right': 0}
            matched = True
            env.cfg.evaluation = False
            try:
                for seed in range(7000, 7256):
                    env.reset(seed=seed)
                    poses = array(env.robot.data.root_link_pose_w).copy()
                    poses[:, :3] -= array(env.scene.env_origins)
                    joints = array(env.robot.data.joint_pos[:, env.joint_ids])
                    for pose, q in zip(poses, joints):
                        sides = [side for side, rest in env.physics['settled_starts'].items()
                                 if np.allclose(pose, rest['root_pose'], atol=1e-6) and
                                 np.allclose(q, rest['joint_rad'], atol=1e-6)]
                        matched &= len(sides) == 1
                        if len(sides) == 1:
                            counts[sides[0]] += 1
                fraction = (counts['left'] + counts['right']) / (256 * env.num_envs)
                expected = env.physics['recovery_start_fraction']
                check('recovery_reset_distribution', matched and all(counts.values()) and
                      abs(fraction - expected) < .1, counts=counts,
                      fallen_fraction=fraction, configured_fraction=expected)
            finally:
                env.cfg.evaluation = True
            # Exercise episode accounting across real steps and automatic resets.
            env.cfg.initial_side = 'left'
            env.reset(seed=7)
            before = env.side_episode_counts.clone()
            rest = env.physics['settled_starts']['upright']
            pose = env.tensor(rest['root_pose']).repeat(env.num_envs, 1)
            pose[:, :3] += env.scene.env_origins
            q = env.robot.data.joint_pos.clone()
            q[:, env.joint_ids] = env.tensor(rest['joint_rad'])
            env.robot.write_root_pose_to_sim(pose)
            env.robot.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
            env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            for _ in range(round(.5 / env.step_dt)):
                env.step(zero)
            check('recovery_diagnostics_no_transient_credit', not bool(env.reached_upright.any()))
            env.step(zero)
            check('recovery_diagnostics_stable_credit', bool(env.reached_upright.all()) and
                  int(env.side_episode_counts[0, 2] - before[0, 2]) == env.num_envs)
            env.episode_length_buf[:] = env.max_episode_length - 1
            env.step(zero)
            change = env.side_episode_counts - before
            check('recovery_diagnostics_pre_reset_completion',
                  int(change[0, 1]) == env.num_envs and int(change[0, 3]) == env.num_envs and
                  int(change[0, 4]) == 0 and not bool(env.reached_upright.any()),
                  counter_change=array(change).tolist())
            env.cfg.initial_side = 'right'
            env.reset(seed=7)
            before = env.side_episode_counts.clone()
            env.fallen_time[0] = env.physics['max_fallen_time_s']
            _, _, term, _, _ = env.step(zero)
            change = env.side_episode_counts - before
            check('recovery_diagnostics_partial_fallen_limit', bool(term[0] and not term[1]) and
                  int(change[1, 1]) == 1 and int(change[1, 4]) == 1 and int(change[1, 3]) == 0 and
                  bool((change[0] == 0).all()), counter_change=array(change).tolist())
        for side in ('upright', 'left'):
            env.cfg.initial_side = side
            for direction in (-1, 1):
                env.reset(seed=7)
                env.robot.write_root_velocity_to_sim(env.tensor([direction*.2, 0., 0., 0., 0., 0.]).repeat(env.num_envs, 1))
                _, reward, term, trunc, _ = env.step(zero)
                dx = env.last_transition[0][:, 0] - env.previous_x
                credited = env.reward_components['progress']
                posture = -env.last_transition[1][:, 2]
                sign_ok = bool((dx * direction > 0).all())
                if direction < 0:
                    credit_ok = bool((credited < 0).all()) and torch.allclose(credited, 30*dx)
                elif side == 'left':
                    credit_ok = bool((posture <= .5).all() and (credited == 0).all())
                else:
                    credit_ok = bool((posture >= .8).all() and (credited > 0).all())
                check(f'progress_sign_{side}_{direction}', sign_ok and credit_ok and
                      not bool(term.any() or trunc.any()) and torch.allclose(reward, sum(env.reward_components.values())),
                      displacement_m=array(dx).tolist(), progress_reward=array(credited).tolist(),
                      upright_cos=array(posture).tolist())

    if 'yaw_cost_weight' in env.physics:
        env.cfg.initial_side = 'upright'
        original_pose = list(env.physics['settled_starts']['upright']['root_pose'])
        quarter_turn = env.tensor([np.cos(np.pi/4), 0., 0., np.sin(np.pi/4)])
        rotated = quat_mul(quarter_turn, env.tensor(original_pose[3:]))
        env.physics['settled_starts']['upright']['root_pose'][3:] = array(rotated).tolist()
        try:
            for angle in (0., np.pi/2, np.pi, 2*np.pi):
                env.reset(seed=7)
                pose = env.robot.data.root_link_pose_w.clone()
                rotation = env.tensor([np.cos(angle/2), 0., 0., np.sin(angle/2)]).repeat(env.num_envs, 1)
                pose[:, 3:] = quat_mul(rotation, pose[:, 3:])
                env.robot.write_root_pose_to_sim(pose)
                env._get_rewards()
                yaw_cost = env.reward_components.get('yaw', torch.zeros(env.num_envs, device=env.device))
                expected = -env.step_dt * env.physics['yaw_cost_weight'] * (1-np.cos(angle))
                check(f'relative_yaw_cost_{angle:.3f}', 'yaw' in env.reward_components and
                      torch.allclose(yaw_cost, torch.full_like(yaw_cost, expected), atol=1e-6),
                      reward=array(yaw_cost).tolist(), expected=expected)
            env.reset(seed=7)
            pose = env.robot.data.root_link_pose_w.clone()
            pose[:, 3:] = quat_mul(quarter_turn.repeat(env.num_envs, 1), pose[:, 3:])
            env.robot.write_root_pose_to_sim(pose)
            env._reset_idx(torch.tensor([0], device=env.device))
            env._get_rewards()
            yaw_cost = env.reward_components.get('yaw', torch.zeros(env.num_envs, device=env.device))
            check('relative_yaw_partial_reset', 'yaw' in env.reward_components and
                  abs(float(yaw_cost[0])) < 1e-6 and
                  abs(float(yaw_cost[1]) + env.step_dt * env.physics['yaw_cost_weight']) < 1e-6)
        finally:
            env.physics['settled_starts']['upright']['root_pose'] = original_pose

    # Read the composed wrench before write_data_to_sim consumes it. This exercises
    # the real environment force path, including frame rotation and COM offset.
    for side in ('upright', 'left'):
        env.cfg.initial_side = side
        for name, velocity in (('rest', [0., 0., 0., 0., 0., 0.]),
                               ('translate', [.2, -.1, 0., 0., 0., 0.]),
                               ('rotate', [0., 0., 0., 0., 0., 2.])):
            env.reset(seed=7)
            env.drag[:] = .3
            env.robot.write_root_velocity_to_sim(env.tensor(velocity).repeat(env.num_envs, 1))
            env._apply_action()
            body = env.base_ids[0]
            q = env.robot.data.body_link_quat_w[:, body]
            force = quat_apply(q, env.robot.instantaneous_wrench_composer.composed_force_as_torch[:, body])
            torque = quat_apply(q, env.robot.instantaneous_wrench_composer.composed_torque_as_torch[:, body])
            lever = quat_apply(q, env.tensor(env.physics['tether_attachment_base_m']).repeat(env.num_envs, 1))
            lever -= env.robot.data.body_com_pos_w[:, body] - env.robot.data.body_link_pos_w[:, body]
            com_velocity = env.robot.data.body_com_vel_w[:, body]
            power = (force * com_velocity[:, :3]).sum(-1) + (torque * com_velocity[:, 3:]).sum(-1)
            moment_error = (torque - torch.cross(lever, force, dim=-1)).abs().max()
            magnitude = torch.linalg.vector_norm(force, dim=-1)
            passed = bool((power <= 1e-7).all() and (magnitude <= .3 + 1e-6).all() and moment_error < 1e-6)
            if name == 'rest':
                passed = passed and bool(magnitude.max() < 1e-7)
            check(f'tether_{side}_{name}', passed, max_power_w=float(power.max()), max_force_n=float(magnitude.max()))
            report['tether_probes'].append({'pose': side, 'motion': name,
                'force_world_n': array(force).tolist(), 'torque_world_nm': array(torque).tolist(),
                'power_w': array(power).tolist(), 'moment_error_nm': float(moment_error)})

    check('solver_effort_limit_matches_config', np.allclose(
        array(env.robot.root_physx_view.get_dof_max_forces()), env.physics['effort_limit_nm']))
    check('solver_velocity_limit_matches_config', np.allclose(
        array(env.robot.root_physx_view.get_dof_max_velocities()), env.physics['velocity_limit_rad_s']))
    report['mass_model'] = {'imported_masses_kg': array(env.imported_masses[0]).tolist(),
                            'nominal_masses_kg': array(env.nominal_masses[0]).tolist(),
                            'nominal_coms_xyzw': array(env.nominal_coms[0]).tolist(),
                            'base_payload': env.physics.get('base_payload')}
    report['robot'] = {'fixed_base': env.robot.is_fixed_base, 'joint_names': env.robot.joint_names,
        'body_names': env.robot.body_names, 'contact_body_names': env.contacts.body_names,
        'masses_kg': array(env.robot.root_physx_view.get_masses()).tolist(),
        'inertias_kg_m2': array(env.robot.root_physx_view.get_inertias()).tolist(),
        'joint_limits_rad': array(env.robot.data.joint_pos_limits).tolist(),
        'effort_limits_nm': array(env.robot.data.joint_effort_limits).tolist(),
        'velocity_limits_rad_s': array(env.robot.data.joint_vel_limits).tolist()}

    if not args.checks_only:
        origin = array(env.scene.env_origins[0])
        env.sim.set_camera_view(eye=origin + np.array([.65, .65, .45]),
                                target=origin + np.array([0., 0., .16]))
        # Inventory authored collision approximations, without claiming they are
        # identical to the cooked PhysX hulls.
        prims = [p for p in Usd.PrimRange(env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot'), Usd.TraverseInstanceProxies())]
        report['colliders'] = [{'path': str(p.GetPath()), 'type': p.GetTypeName(),
            'approximation': str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get())}
            for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
        env.sim.stage.Flatten().Export(str(args.output / 'stage.usda'))
        # Transform authored mesh vertices using live body poses. USD/Fabric's
        # presentation transforms need not be updated by each physics step.
        collision_vertices = []
        transforms = UsdGeom.XformCache()
        for prim in prims:
            if not prim.IsA(UsdGeom.Mesh):
                continue
            collider = prim
            while collider and not collider.HasAPI(UsdPhysics.CollisionAPI) and not collider.HasAPI(UsdPhysics.RigidBodyAPI):
                collider = collider.GetParent()
            if not collider or not collider.HasAPI(UsdPhysics.CollisionAPI):
                continue
            body = prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body:
                raise RuntimeError(f'Collider has no rigid body: {prim.GetPath()}')
            matrix, _ = transforms.ComputeRelativeTransform(prim, body)
            matrix = np.array(matrix)
            vertices = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            local = vertices @ matrix[:3, :3] + matrix[3, :3]
            mesh = UsdGeom.Mesh(prim)
            collision_vertices.append((str(prim.GetPath()), env.robot.body_names.index(body.GetName()), local,
                mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()))
        if not collision_vertices:
            raise RuntimeError('No authored collision mesh vertices found; clearance evidence would be empty')

        def mesh_clearance():
            poses = array(env.robot.data.body_link_pose_w[0])
            return [{'path': path, 'min_z_m': float((points @ rotation_matrix(poses[index, 3:]).T + poses[index, :3])[:, 2].min())}
                    for path, index, points, _, _ in collision_vertices]

        def collision_view(destination):
            # Render authored collision triangles as non-physical display meshes.
            # These show source surfaces, not the separately cooked convex hulls.
            poses = array(env.robot.data.body_link_pose_w[0])
            for number, (_, index, points, counts, indices) in enumerate(collision_vertices):
                mesh = UsdGeom.Mesh.Define(env.sim.stage, f'/InspectionGeometry/mesh_{number}')
                mesh.CreatePointsAttr(points @ rotation_matrix(poses[index, 3:]).T + poses[index, :3])
                mesh.CreateFaceVertexCountsAttr(counts)
                mesh.CreateFaceVertexIndicesAttr(indices)
                mesh.CreateDisplayColorAttr([(0.05, .6, .75)])
                mesh.CreateDoubleSidedAttr(True)
            robots = UsdGeom.Imageable(env.sim.stage.GetPrimAtPath('/World/envs'))
            robots.MakeInvisible()
            try:
                for _ in range(5):
                    picture = frame()
                imageio.imwrite(destination, picture)
            finally:
                robots.MakeVisible()
                env.sim.stage.RemovePrim('/InspectionGeometry')

        def frame():
            picture = env.render()
            if picture is None or not picture.size:
                raise RuntimeError('Renderer did not produce an image')
            return picture

        for side in ('upright', 'left', 'right'):
            env.cfg.initial_side = side
            env.reset(seed=7)
            samples = [{'time_s': 0., **state()}]
            clearances = {'initial': mesh_clearance()}
            # Warm the renderer without advancing physics.
            for _ in range(5):
                picture = frame()
            imageio.imwrite(args.output / f'{side}-initial.png', picture)
            faults = []
            with imageio.get_writer(args.output / f'{side}.mp4', fps=20) as writer:
                writer.append_data(picture)
                for step in range(1, 101):
                    obs, reward, term, trunc, _ = env.step(zero)
                    samples.append({'time_s': step * env.step_dt, **state()})
                    writer.append_data(frame())
                    if step in (40, 100):
                        clearances[f'{step * env.step_dt:.0f}s'] = mesh_clearance()
                        imageio.imwrite(args.output / f'{side}-{step * env.step_dt:.0f}s.png', frame())
                    if bool(term.any() or trunc.any()):
                        faults.append(step)
                        break  # Do not present an auto-reset as continued motion.
            (args.output / f'{side}.json').write_text(json.dumps(samples, indent=2) + '\n')
            report['poses'][side] = {'initial': samples[0], 'final': samples[-1], 'fault_steps': faults,
                                    'authored_collision_mesh_clearances': clearances}
            collision_view(args.output / f'{side}-authored-collisions.png')
            print(f'POSE {side}: final upright {samples[-1]["upright_cos"]}, faults {faults}', flush=True)

        env.cfg.initial_side = 'upright'
        for joint, name in enumerate(env.contract.joint_names):
            env.reset(seed=7)
            for _ in range(40):
                env.step(zero)
            start = array(env.robot.data.joint_pos[:, env.joint_ids]).copy()
            max_speed = max_effort = max_slew = 0.
            previous = env.robot.data.joint_pos_target[:, env.joint_ids].clone()
            records = []
            for step in range(60):
                actions = zero.clone()
                actions[:, joint] = .15 if step < 30 else -.15
                obs, reward, term, trunc, _ = env.step(actions)
                applied = env.robot.data.joint_pos_target[:, env.joint_ids]
                max_slew = max(max_slew, float(((applied - previous).abs() / env.step_dt).max()))
                previous = applied.clone()
                max_speed = max(max_speed, float(env.robot.data.joint_vel.abs().max()))
                max_effort = max(max_effort, float(env.robot.data.applied_torque.abs().max()))
                records.append({'time_s': (step+1)*env.step_dt, **state(), 'target_rad': array(applied).tolist()})
                if bool(term.any() or trunc.any()):
                    break
            report['joint_probes'][name] = {'start_joint_rad': start.tolist(),
                'final_joint_rad': records[-1]['joint_rad'], 'steps': len(records),
                'max_all_joint_speed_rad_s': max_speed, 'max_estimated_drive_effort_nm': max_effort,
                'max_target_slew_rad_s': max_slew}
            (args.output / f'joint-{name}.json').write_text(json.dumps(records, indent=2) + '\n')
            check(f'joint_speed_limit_{name}', max_speed <= env.physics['velocity_limit_rad_s'] + 1e-3,
                  max_joint_speed_rad_s=max_speed)
            check(f'estimated_drive_effort_limit_{name}', max_effort <= env.physics['effort_limit_nm'] + 1e-5,
                  max_estimated_drive_effort_nm=max_effort)
            check(f'target_slew_{name}', max_slew <= max(env.contract.max_target_speed_rad_s) + 1e-4,
                  max_target_speed_rad_s=max_slew)

    report['automated_checks_passed'] = all(c['passed'] for c in report['checks'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    env.close()
    if not report['automated_checks_passed']:
        raise AssertionError('Inspection checks failed; see report.json')
    print(f'Inspection checks passed; visual/physical validation still required: {args.output}', flush=True)
except BaseException:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    if 'env' in locals():
        env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
