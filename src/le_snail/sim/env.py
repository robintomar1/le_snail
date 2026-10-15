"""Isaac Lab 2.3.2 direct environment for a floating SO-101."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import warp as wp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from le_snail.contract import PolicyContract
from le_snail.bundle import file_hash
from le_snail.physics import combine_box_payload, load_physics


@configclass
class SnailEnvCfg(DirectRLEnvCfg):
    decimation = 10
    episode_length_s = 120.0
    action_space = 6
    observation_space = 24
    state_space = 0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.005, render_interval=10)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=2.0, replicate_physics=True)
    robot_cfg: ArticulationCfg | None = None
    contract_path: str = ''
    physics_path: str = ''
    stage: str = 'crawl'
    evaluation: bool = False
    initial_side: str = 'upright'


def make_config(asset, contract_path, physics_path, *, num_envs=256, stage='crawl', device='cuda:0', evaluation=False, initial_side='upright', seed=42):
    contract = PolicyContract.load(contract_path)
    p = load_physics(physics_path)
    if stage not in ('crawl', 'recover', 'robust') or initial_side not in ('upright', 'left', 'right'):
        raise ValueError('Unknown curriculum stage or initial side')
    if not Path(asset).is_file():
        raise FileNotFoundError(f'{asset}: run scripts/fetch_asset.py first')
    if 'settled_starts' in p:
        if file_hash(asset) != p['settled_start_asset_sha256'] or contract.fingerprint != p['settled_start_contract_fingerprint']:
            raise ValueError('Settled reset states belong to a different asset or policy contract')
        for state in p['settled_starts'].values():
            if any(not lo <= q <= hi for q, lo, hi in zip(state['joint_rad'], contract.lower_rad, contract.upper_rad)):
                raise ValueError('Settled reset joints exceed the policy limits')
    ratio = contract.dt / p['physics_dt']
    if not math.isclose(ratio, round(ratio)) or round(ratio) < 1:
        raise ValueError('Policy period must be an integer number of physics steps')
    if p['max_action_delay_steps'] not in (0, 1):
        raise ValueError('This implementation supports zero or one action-delay step')
    cfg = SnailEnvCfg()
    cfg.contract_path, cfg.physics_path = str(Path(contract_path).resolve()), str(Path(physics_path).resolve())
    cfg.stage, cfg.evaluation, cfg.initial_side, cfg.seed = stage, evaluation, initial_side, seed
    cfg.scene.num_envs = num_envs
    cfg.viewer.eye = (1.0, 1.0, 0.7)
    cfg.viewer.lookat = (0.2, 0.0, 0.1)
    cfg.decimation = round(ratio)
    cfg.sim.dt, cfg.sim.render_interval, cfg.sim.device = p['physics_dt'], cfg.decimation, device
    cfg.robot_cfg = ArticulationCfg(
        prim_path='/World/envs/env_.*/Robot',
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(Path(asset).resolve()), fix_base=False, root_link_name='base_link',
            merge_fixed_joints=True, self_collision=True, collider_type='convex_decomposition',
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=p['stiffness_nm_rad'], damping=p['damping_nm_s_rad']),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., p['initial_height_m']), joint_pos=dict(zip(contract.joint_names, contract.home_rad)), joint_vel={'.*': 0.}),
        actuators={'servos': ImplicitActuatorCfg(
            joint_names_expr=['.*'], stiffness=p['stiffness_nm_rad'], damping=p['damping_nm_s_rad'],
            effort_limit_sim=p['effort_limit_nm'], velocity_limit_sim=p['velocity_limit_rad_s'],
        )},
        soft_joint_pos_limit_factor=1.0,
    )
    return cfg


class SnailEnv(DirectRLEnv):
    cfg: SnailEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.contract = PolicyContract.load(cfg.contract_path)
        self.physics = load_physics(cfg.physics_path)
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        if self.robot.is_fixed_base:
            raise RuntimeError('The imported robot is fixed to the world; crawling is impossible')
        self.joint_ids, names = self.robot.find_joints(self.contract.joint_names, preserve_order=True)
        if names != self.contract.joint_names or self.robot.num_joints != 6:
            raise RuntimeError(f'Unexpected imported joint order: {names}')
        self.base_ids, _ = self.robot.find_bodies('base_link')
        if len(self.base_ids) != 1:
            raise RuntimeError('Expected one base_link')
        self.home = self.tensor(self.contract.home_rad)
        self.lower, self.upper = self.tensor(self.contract.lower_rad), self.tensor(self.contract.upper_rad)
        self.scale = self.tensor(self.contract.action_scale_rad)
        self.slew = self.tensor(self.contract.max_target_speed_rad_s) * self.step_dt
        self.target = self.home.repeat(self.num_envs, 1)
        self.applied_target = self.target.clone()
        self.old_target = self.target.clone()
        self.command_queue = self.target[:, None, :].repeat(1, 2, 1)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_q = self.target.clone()
        self.filtered_velocity = torch.zeros_like(self.target)
        self.observation_cache = torch.zeros((self.num_envs, 24), device=self.device)
        self.observation_step = -1
        self.start_pos = self.robot.data.root_pos_w.clone()
        self.start_yaw = torch.zeros(self.num_envs, device=self.device)
        self.previous_x = self.start_pos[:, 0].clone()
        self.fallen_time = torch.zeros(self.num_envs, device=self.device)
        # Episode start identity is simulator-only, shared by diagnostics and
        # optional side-start shaping. It never enters actor observations.
        self.initial_side = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_upright = torch.zeros(self.num_envs, device=self.device)
        self.best_upright_progress = torch.zeros(self.num_envs, device=self.device)
        self.side_shaping_reward = torch.zeros(2, device=self.device)
        self.stable_upright_steps = torch.zeros_like(self.initial_side)
        self.reached_upright = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Rows: left/right. Columns documented by recovery_diagnostics().
        self.side_episode_counts = torch.zeros((2, 6), dtype=torch.long, device=self.device)
        self.drag = torch.zeros(self.num_envs, device=self.device)
        self.gravity_w = self.tensor([0., 0., -1.]).repeat(self.num_envs, 1)
        self.base_indices_wp = wp.array(self.base_ids, dtype=wp.int32, device=self.device)
        self.nominal_masses = self.robot.root_physx_view.get_masses().clone()
        self.nominal_inertias = self.robot.root_physx_view.get_inertias().clone()
        self.imported_masses = self.nominal_masses.clone()
        self.imported_inertias = self.nominal_inertias.clone()
        self.imported_coms = self.robot.root_physx_view.get_coms().clone()
        if 'base_payload' in self.physics:
            body = self.base_ids[0]
            mass, center, inertia = combine_box_payload(
                float(self.nominal_masses[0, body]), self.imported_coms[0, body, :3].numpy(),
                self.nominal_inertias[0, body].numpy(), self.physics['base_payload'])
            self.nominal_masses[:, body] = mass
            self.nominal_inertias[:, body] = torch.as_tensor(inertia.reshape(9), dtype=self.nominal_inertias.dtype)
            coms = self.imported_coms.clone()
            coms[:, body, :3] = torch.as_tensor(center, dtype=coms.dtype)
            ids = self.robot._ALL_INDICES.cpu()
            # COM pose first; set_inertias then diagonalizes the full link-frame
            # COM tensor. Cache the combined nominal properties only once.
            self.robot.root_physx_view.set_masses(self.nominal_masses, ids)
            self.robot.root_physx_view.set_coms(coms, ids)
            self.robot.root_physx_view.set_inertias(self.nominal_inertias, ids)
        self.nominal_coms = self.robot.root_physx_view.get_coms().clone()
        from le_snail.speed import PhysicsSpeedDiagnostics
        self.speed_monitor = PhysicsSpeedDiagnostics(self.num_envs, 6, self.physics_dt, self.device)
        from le_snail.action_stats import ActionDiagnostics
        self.action_monitor = ActionDiagnostics(self.num_envs, self.contract.joint_names, self.device)
        self._speed_sample_counter = self._sim_step_counter
        self.probes = []
        for probe in self.physics['protected_points']:
            ids, _ = self.robot.find_bodies(probe['body'])
            if len(ids) != 1:
                raise ValueError(f'Protected probe body is ambiguous: {probe}')
            self.probes.append((ids[0], self.tensor(probe['position_m']).repeat(self.num_envs, 1), float(probe['radius_m'])))
        self._init_handover()

    def _init_handover(self):
        """Scripted-prefix handover curriculum (#6). Training resets only; never in evaluation.

        A teacher script drives the first k policy steps of some side-start episodes through the
        normal action path. Those steps earn zero reward and are reported through extras so the
        learner can exclude them; at handover the episode accounting restarts from that state.
        """
        n = self.num_envs
        self._env_index = torch.arange(n, device=self.device)
        self.teacher_step = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.teacher_left = torch.zeros(n, dtype=torch.long, device=self.device)
        self.teacher_keys = torch.zeros((n, 4, 6), device=self.device)
        self.teacher_key_i = torch.zeros_like(self.teacher_left)
        self.teacher_held = torch.zeros_like(self.teacher_left)
        self.handover_pending = torch.zeros_like(self.teacher_step)
        self.handover_variant = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.handover_prefix = torch.zeros_like(self.teacher_left)
        self.handover_cos = torch.full((n,), float('nan'), device=self.device)
        # Bin 0: official start without a prefix; 1-3: handover upright <0.2, 0.2-0.5, >=0.5.
        self.handover_bin = torch.zeros_like(self.teacher_left)
        self.handover_counts = torch.zeros((4, 2, 6), dtype=torch.long, device=self.device)
        self.prefix_terminations = torch.zeros(2, dtype=torch.long, device=self.device)
        # Unrecovered handover attempts may end early as time-outs (bootstrapped, never absorbing).
        self.attempt_steps = 0
        self.attempt_truncations = torch.zeros((4, 2), dtype=torch.long, device=self.device)
        self._logged_counts = torch.zeros((4, 2, 7), dtype=torch.long, device=self.device)
        # Side starts before their first stable upright: net |x|, |y| at recovery or episode end, lane exits.
        self.prerecovery_count = torch.zeros((4, 2), dtype=torch.long, device=self.device)
        self.prerecovery_abs_sum = torch.zeros((4, 2, 2), device=self.device)
        self.prerecovery_lane_exits = torch.zeros((4, 2), dtype=torch.long, device=self.device)
        # Upright-start episodes: falls (cos < 0.5 for >= 0.5 s) and later stable recoveries.
        self.upright_fell = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.upright_fall_recovered = torch.zeros_like(self.upright_fell)
        self.upright_fall_events = torch.zeros(2, dtype=torch.long, device=self.device)
        self.teacher = None
        if 'handover' not in self.physics:
            return
        config = self.physics['handover']
        if 'attempt_seconds' in config:
            self.attempt_steps = round(config['attempt_seconds'] / self.step_dt)
        names, sides = [], {}
        for code, side in ((1, 'left'), (2, 'right')):
            variants = config['scripts'][side]
            longest = max(len(v['prefix_steps']) for v in variants)
            points = torch.zeros((len(variants), longest), dtype=torch.long, device=self.device)
            for i, variant in enumerate(variants):
                points[i, :len(variant['prefix_steps'])] = torch.tensor(variant['prefix_steps'], device=self.device)
            sides[code] = {
                'keys': self.tensor([v['keyframes_rad'] + [self.contract.home_rad] for v in variants]),
                'points': points,
                'counts': torch.tensor([len(v['prefix_steps']) for v in variants], device=self.device),
                'ids': torch.arange(len(names), len(names) + len(variants), device=self.device)}
            names += [f"{side}:{v['id']}" for v in variants]
        self.teacher = {'fraction': float(config['fraction_of_side_starts']), 'sides': sides, 'names': names}

    def _assign_teachers(self, env_ids):
        side = self.initial_side[env_ids]
        chosen = (side > 0) & (torch.rand(len(env_ids), device=self.device) < self.teacher['fraction'])
        for code, table in self.teacher['sides'].items():
            selected = env_ids[chosen & (side == code)]
            if len(selected) == 0:
                continue
            variant = torch.randint(len(table['keys']), (len(selected),), device=self.device)
            point = (torch.rand(len(selected), device=self.device) * table['counts'][variant]).long()
            point = torch.minimum(point, table['counts'][variant] - 1)
            self.teacher_keys[selected] = table['keys'][variant]
            self.teacher_left[selected] = table['points'][variant, point]
            self.handover_prefix[selected] = table['points'][variant, point]
            self.handover_variant[selected] = table['ids'][variant]

    def _advance_teacher(self, teacher_actions):
        # Same keyframe sequencing as the feasibility probe: advance once the slewed target
        # reaches the clipped goal and holds (2 steps; 20 at the final home keyframe).
        active = self.teacher_step
        clipped = torch.clamp(self.home + teacher_actions * self.scale, self.lower, self.upper)
        reached = (self.target - clipped).abs().amax(dim=1) < 1e-3
        running = active & (self.teacher_key_i <= 3)
        self.teacher_held = torch.where(reached & running, self.teacher_held + 1, 0)
        advance = running & (self.teacher_held >= torch.where(self.teacher_key_i == 3, 20, 2))
        self.teacher_key_i += advance.long()
        self.teacher_held = torch.where(advance, 0, self.teacher_held)
        self.teacher_left -= active.long()
        finished = active & ((self.teacher_left <= 0) | (self.teacher_key_i > 3))
        self.teacher_left = torch.where(finished, 0, self.teacher_left)
        self.handover_pending |= finished

    def _handover(self, mask):
        """Restart episode accounting at the teacher's final state; the learner earns nothing for the prefix."""
        upright = -self.robot.data.projected_gravity_b[:, 2]
        self.start_pos[mask] = self.robot.data.root_pos_w[mask]
        self.start_yaw[mask] = self.yaw(self.robot.data.root_quat_w[mask])
        self.reset_upright[mask] = upright[mask]
        self.best_upright_progress[mask] = 0.
        self.fallen_time[mask] = 0.
        self.stable_upright_steps[mask] = 0
        self.reached_upright[mask] = False
        self.episode_length_buf[mask] = 0
        self.handover_cos[mask] = upright[mask]
        bins = 1 + (upright >= .2).long() + (upright >= .5).long()
        self.handover_bin[mask] = bins[mask]
        self.handover_pending[mask] = False
        side = torch.stack((self.initial_side == 1, self.initial_side == 2), dim=1) & mask[:, None]
        for b in (1, 2, 3):
            self.handover_counts[b, :, 0] += (side & (bins == b)[:, None]).sum(dim=0)

    def tensor(self, value):
        return torch.tensor(value, device=self.device, dtype=torch.float32)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # A local solid slab avoids a network dependency on NVIDIA's default ground USD.
        ground = sim_utils.CuboidCfg(size=(200., 200., 0.1), collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., friction_combine_mode='multiply', restitution=0.))
        ground.func('/World/Ground', ground, translation=(0., 0., -0.05))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=['/World/Ground'])
        self.scene.articulations['robot'] = self.robot
        light = sim_utils.DomeLightCfg(intensity=1800., color=(0.8, 0.8, 0.8))
        light.func('/World/Light', light)

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise RuntimeError('Policy produced non-finite actions')
        teacher_actions = learner = None
        if self.teacher is not None:
            self.teacher_step = self.teacher_left > 0
            learner = ~self.teacher_step
            if bool(self.teacher_step.any()):
                goal = self.teacher_keys[self._env_index, self.teacher_key_i.clamp(max=3)]
                teacher_actions = ((goal - self.home) / self.scale).clamp(-1., 1.)
                actions = torch.where(self.teacher_step[:, None], teacher_actions, actions)
        self.old_target.copy_(self.target)
        unclipped = self.home + torch.clamp(actions, -1., 1.) * self.scale
        desired = torch.clamp(unclipped, self.lower, self.upper)
        self.target += torch.clamp(desired - self.target, -self.slew, self.slew)
        self.action_monitor.update(actions, unclipped, desired, self.old_target, self.target, self.slew, self.lower, self.upper, mask=learner)
        self.command_queue[:, 1] = self.command_queue[:, 0]
        self.command_queue[:, 0] = self.target
        self.previous_x = self.robot.data.root_pos_w[:, 0].clone()
        if teacher_actions is not None:
            self._advance_teacher(teacher_actions)

    def _record_speed(self, completed_step):
        if completed_step <= self._speed_sample_counter:
            return
        q = self.robot.data.joint_pos[:, self.joint_ids]
        qd = self.robot.data.joint_vel[:, self.joint_ids]
        pd = self.physics['stiffness_nm_rad'] * (self.applied_target - q) - self.physics['damping_nm_s_rad'] * qd
        learner = ~self.teacher_step if self.teacher is not None else None
        context = torch.stack((self.handover_bin.float(), -self.robot.data.projected_gravity_b[:, 2],
                               self.teacher_step.float(), self.episode_length_buf.float()), dim=1)
        self.speed_monitor.update(qd, self.robot.data.joint_vel_limits[:, self.joint_ids], pd, mask=learner, context=context)
        self._speed_sample_counter = completed_step

    def speed_diagnostics(self):
        report = self.speed_monitor.report()
        location = report.get('longest_driven_proxy_location')
        if location is not None:
            location['joint'] = self.contract.joint_names[location['joint_index']]
            values = location.pop('context')
            if values is not None:
                location.update(handover_bin=int(values[0]), upright_cos=values[1], teacher_step=bool(values[2]),
                                policy_steps_since_episode_or_handover=int(values[3]))
        return report

    def action_diagnostics(self):
        return self.action_monitor.report()

    def _apply_action(self):
        # DirectRLEnv increments this counter before applying each substep.
        # Read the previous completed substep before replacing its target.
        self._record_speed(self._sim_step_counter - 1)
        delayed = self.command_queue[torch.arange(self.num_envs, device=self.device), self.delay_steps]
        self.applied_target.copy_(delayed)
        self.robot.set_joint_position_target(delayed, joint_ids=self.joint_ids)
        base_q = self.robot.data.body_link_quat_w[:, self.base_ids[0]]
        offset_w = quat_apply(base_q, self.tensor(self.physics['tether_attachment_base_m']).repeat(self.num_envs, 1))
        com_offset_w = self.robot.data.body_com_pos_w[:, self.base_ids[0]] - self.robot.data.body_link_pos_w[:, self.base_ids[0]]
        lever_w = offset_w - com_offset_w
        velocity = self.robot.data.body_com_vel_w[:, self.base_ids[0], :3]
        omega = self.robot.data.body_com_vel_w[:, self.base_ids[0], 3:]
        attachment_velocity = velocity + torch.cross(omega, lever_w, dim=-1)
        planar = attachment_velocity[:, :2]
        speed = torch.linalg.vector_norm(planar, dim=1, keepdim=True)
        force_w = torch.zeros((self.num_envs, 3), device=self.device)
        force_w[:, :2] = -self.drag[:, None] * planar / (speed + self.physics['drag_velocity_scale_m_s'])
        # Forces oppose attachment motion; torque is about the COM, so the tether cannot propel the robot.
        torque_w = torch.cross(lever_w, force_w, dim=-1)
        force_b = quat_apply_inverse(base_q, force_w)[:, None, :].contiguous()
        torque_b = quat_apply_inverse(base_q, torque_w)[:, None, :].contiguous()
        self.robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=wp.from_torch(force_b, dtype=wp.vec3f), torques=wp.from_torch(torque_b, dtype=wp.vec3f),
            body_ids=self.base_indices_wp,
        )

    def _get_observations(self):
        if self.teacher is not None:
            # After resets: rows the learner will act on next (teacher prefixes excluded).
            self.extras['next_observation_is_learner'] = (self.teacher_left == 0).float()
        if self.observation_step == self.common_step_counter:
            return {'policy': self.observation_cache}
        q = self.robot.data.joint_pos[:, self.joint_ids].clone()
        gravity = self.robot.data.projected_gravity_b.clone()
        gyro = self.robot.data.root_ang_vel_b.clone()
        if self.cfg.stage == 'robust':
            q += torch.randn_like(q) * self.physics['position_noise_std_rad']
            gyro += torch.randn_like(gyro) * self.physics['gyro_noise_std_rad_s']
            gravity += torch.randn_like(gravity) * self.physics['gravity_noise_std']
            gravity /= torch.linalg.vector_norm(gravity, dim=1, keepdim=True).clamp_min(1e-8)
        a = self.contract.velocity_filter_alpha
        self.filtered_velocity = a * (q - self.previous_q) / self.step_dt + (1-a) * self.filtered_velocity
        self.previous_q.copy_(q)
        self.observation_cache = torch.cat((q - self.home, self.filtered_velocity, gravity, gyro, (self.applied_target - self.home) / self.scale), dim=-1)
        self.observation_step = self.common_step_counter
        return {'policy': self.observation_cache}

    def protected_contact(self):
        bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for body, offset, radius in self.probes:
            position = self.robot.data.body_link_pos_w[:, body] + quat_apply(self.robot.data.body_link_quat_w[:, body], offset)
            bad |= position[:, 2] - radius <= 0.001
        return bad

    def _get_rewards(self):
        displacement = self.robot.data.root_pos_w - self.start_pos
        upright = -self.robot.data.projected_gravity_b[:, 2]
        progress = self.robot.data.root_pos_w[:, 0] - self.previous_x
        command_rate = (self.target - self.old_target) / self.step_dt
        gate = torch.ones_like(progress)
        if 'positive_progress_upright_range' in self.physics:
            low, high = self.physics['positive_progress_upright_range']
            gate = ((upright - low) / (high - low)).clamp(0., 1.)
        credited = progress.clamp(max=0.) + gate * progress.clamp(min=0.)
        if self.physics.get('side_start_progress_neutral_until_recovery', False):
            # Before its first stable upright a side-start episode neither earns nor pays for
            # displacement, like the acceptance test, which measures progress only after recovery.
            credited = torch.where((self.initial_side != 0) & ~self.reached_upright, 0., credited)
        self.reward_components = {
            'progress': 30. * credited,
            'tilt': -self.step_dt * 0.1 * (1-upright),
            'lateral': -self.step_dt * 0.3 * displacement[:, 1].square(),
            'command_rate': -self.step_dt * 0.02 * command_rate.square().sum(dim=1),
            'time': torch.full_like(progress, -self.step_dt * .02),
            'protected_contact': -self.protected_contact().float(),
        }
        if self.physics.get('yaw_cost_weight', 0.) > 0:
            heading_error = self.yaw(self.robot.data.root_quat_w) - self.start_yaw
            self.reward_components['yaw'] = -self.step_dt * self.physics['yaw_cost_weight'] * (1-torch.cos(heading_error))
        learner = None
        if self.teacher is not None:
            # Teacher prefix steps earn nothing; the learner gets no credit for the script's work.
            learner = (~self.teacher_step).float()
            self.reward_components = {name: value * learner for name, value in self.reward_components.items()}
        cap = self.physics.get('side_upright_progress_reward_cap', 0.)
        if cap > 0:
            posture_progress = ((upright - self.reset_upright) / (.8 - self.reset_upright).clamp_min(1e-6)).clamp(0., 1.)
            posture_progress = torch.where(self.initial_side != 0, posture_progress, 0.)
            best = torch.maximum(self.best_upright_progress, posture_progress)
            # Telescoping increments spend at most cap per episode. Falls and
            # recoveries never refill the budget; only an actual reset or handover does.
            credit = cap * (best - self.best_upright_progress)
            if learner is not None:
                credit = credit * learner
            self.best_upright_progress.copy_(best)
            self.reward_components['upright_progress'] = credit
            self.side_shaping_reward += torch.stack((credit[self.initial_side == 1].sum(),
                                                     credit[self.initial_side == 2].sum()))
        stats = {f'Reward/{name}': value.mean().detach() for name, value in self.reward_components.items()}
        gated_out = 30. * (progress - credited)
        stats['Reward/gated_out_progress'] = (gated_out if learner is None else gated_out * learner).mean().detach()
        if learner is not None:
            self.extras['learner_mask'] = learner
            # Per-step event counts by handover bin and side; the runner logs each iteration's
            # mean, so mean x steps-per-iteration recovers the events in that iteration.
            current = torch.cat((self.handover_counts, self.attempt_truncations[:, :, None]), dim=2)
            delta = (current - self._logged_counts).float()
            self._logged_counts.copy_(current)
            for b, bin_name in enumerate(('official', 'below_0.2', '0.2_to_0.5', 'at_least_0.5')):
                for s, side_name in enumerate(('left', 'right')):
                    for c, event in ((0, 'started'), (1, 'completed'), (2, 'stable_get_ups'), (6, 'attempt_truncations')):
                        stats[f'Handover/{bin_name}_{side_name}_{event}'] = delta[b, s, c]
        self.extras['log'] = {**self.extras.get('log', {}), **stats}
        return sum(self.reward_components.values())

    @staticmethod
    def yaw(quaternion):
        w, x, y, z = quaternion.unbind(-1)
        return torch.atan2(2*(w*z + x*y), 1-2*(y*y + z*z))

    def _get_dones(self):
        # Capture the final substep before any environment is autoreset.
        self._record_speed(self._sim_step_counter)
        if self.teacher is not None and bool(self.handover_pending.any()):
            self._handover(self.handover_pending.clone())
        self.last_transition =(self.robot.data.root_pos_w.clone(), self.robot.data.projected_gravity_b.clone(), self.protected_contact().clone())
        self.last_transition_quat = self.robot.data.root_quat_w.clone()
        displacement = self.robot.data.root_pos_w - self.start_pos
        upright = -self.robot.data.projected_gravity_b[:, 2]
        self.fallen_time = torch.where(upright < 0.5, self.fallen_time + self.step_dt, 0.)
        invalid = ~torch.isfinite(self.robot.data.root_state_w).all(dim=1)
        invalid |= ~torch.isfinite(self.robot.data.joint_pos).all(dim=1)
        terminated = invalid | self.protected_contact() | (displacement[:, 1].abs() > 0.15) | (self.robot.data.root_pos_w[:, 2] < -0.03)
        fallen_timeout = self.fallen_time > self.physics['max_fallen_time_s']
        terminated |= fallen_timeout
        # DirectRLEnv increments the counter before calling _get_dones.
        timeout = self.episode_length_buf >= self.max_episode_length
        self.stable_upright_steps = torch.where(upright >= .8, self.stable_upright_steps + 1, 0)
        # Eleven samples at20Hz span0.5s from the first qualifying sample.
        stable = self.stable_upright_steps >= round(.5 / self.step_dt) + 1
        first_upright = stable & ~self.reached_upright
        self.reached_upright |= stable
        if self.physics.get('side_start_heading_reset_at_recovery', False):
            # The heading cost targets turning while crawling, not the orientation a get-up leaves.
            recovered = first_upright & (self.initial_side != 0) & (self.teacher_left == 0)
            self.start_yaw[recovered] = self.yaw(self.robot.data.root_quat_w[recovered])
        side = torch.stack((self.initial_side == 1, self.initial_side == 2), dim=1)
        if self.teacher is not None and self.attempt_steps > 0:
            # End an unrecovered handover attempt as a time-out so its value is bootstrapped and
            # ending early cannot pay. Recovered episodes, or ones mid-way through a >=0.8 streak,
            # continue; official side starts (bin 0) are never truncated by this rule.
            attempt = ((self.handover_bin > 0) & ~self.reached_upright & (self.stable_upright_steps == 0)
                       & (self.episode_length_buf >= self.attempt_steps) & ~terminated)
            timeout = timeout | attempt
            for b in (1, 2, 3):
                self.attempt_truncations[b] += (side & (attempt & (self.handover_bin == b))[:, None]).sum(dim=0)
        ended = terminated | timeout
        # Diagnostics only: displacement of side starts before recovery, and upright-start falls.
        counted = (self.initial_side != 0) & (self.teacher_left == 0)
        closing = counted & (first_upright | (ended & ~self.reached_upright))
        lane_exit = counted & terminated & (displacement[:, 1].abs() > 0.15) & ~self.reached_upright
        for b in range(4):
            rows = side & (closing & (self.handover_bin == b))[:, None]
            self.prerecovery_count[b] += rows.sum(dim=0)
            self.prerecovery_abs_sum[b] += (rows[:, :, None] * displacement[:, None, :2].abs()).sum(dim=0)
            self.prerecovery_lane_exits[b] += (side & (lane_exit & (self.handover_bin == b))[:, None]).sum(dim=0)
        fell = (self.initial_side == 0) & ~self.upright_fell & (self.fallen_time >= .5)
        self.upright_fell |= fell
        recovered_after_fall = self.upright_fell & ~self.upright_fall_recovered & stable
        self.upright_fall_recovered |= recovered_after_fall
        self.upright_fall_events += torch.stack((fell.sum(), recovered_after_fall.sum()))
        events = torch.stack((ended, first_upright, ended & self.reached_upright,
                              ended & fallen_timeout, ended & fallen_timeout & self.reached_upright), dim=1)
        if self.teacher is None:
            self.side_episode_counts[:, 1:] += (side[:, :, None] & events[:, None, :]).sum(dim=0)
        else:
            # Official side starts keep the existing counters. Prefix steps are excluded and
            # handover episodes are counted by the upright cosine at handover.
            in_prefix = self.teacher_left > 0
            for b in range(4):
                rows = side & (~in_prefix & (self.handover_bin == b))[:, None]
                counts = (rows[:, :, None] & events[:, None, :]).sum(dim=0)
                self.handover_counts[b, :, 1:] += counts
                if b == 0:
                    self.side_episode_counts[:, 1:] += counts
            self.prefix_terminations += (side & (in_prefix & ended)[:, None]).sum(dim=0)
        if self.render_mode == 'rgb_array' and bool((terminated | timeout).any()):
            # Capture before DirectRLEnv auto-resets so evaluation videos retain
            # the same terminal transition used by the acceptance scorer.
            self.terminal_frame = self.render().copy()
        # Ordinary falls do not terminate episodes. Evaluation ends on its independent acceptance checks.
        return terminated, timeout

    def recovery_diagnostics(self):
        """Per-run side-start counters, including endings before automatic reset."""
        names = ('started', 'completed', 'reached_stable_upright', 'completed_after_stable_upright',
                 'completed_at_fallen_limit', 'completed_after_upright_and_at_fallen_limit')
        result = {'stable_definition': 'upright cosine >=0.8 continuously for0.5s',
                  'scope': 'side-start episodes in this environment lifetime; incomplete episodes excluded from fractions',
                  'sides': {}}
        for side, values in zip(('left', 'right'), self.side_episode_counts.cpu().tolist()):
            row = dict(zip(names, values))
            denominator = row['completed']
            row['completed_upright_fraction'] = row['completed_after_stable_upright'] / denominator if denominator else None
            row['completed_fallen_limit_fraction'] = row['completed_at_fallen_limit'] / denominator if denominator else None
            row['completed_other_endings'] = denominator - row['completed_at_fallen_limit']
            result['sides'][side] = row
        result['upright_progress_shaping'] = {
            'cap_per_side_start_episode': self.physics.get('side_upright_progress_reward_cap', 0.),
            'credited_reward_including_open_episodes': dict(zip(('left', 'right'), self.side_shaping_reward.cpu().tolist())),
        }
        if self.teacher is not None:
            result['scope'] += '; scripted-prefix handover episodes are excluded here (see handover_diagnostics), and shaping credit includes handover episodes'
        bins = ('official_start', 'handover_upright_below_0.2', 'handover_upright_0.2_to_0.5', 'handover_upright_at_least_0.5')
        counts, sums, lanes = (t.cpu().tolist() for t in (self.prerecovery_count, self.prerecovery_abs_sum, self.prerecovery_lane_exits))
        result['side_start_before_first_upright'] = {
            'definition': 'Net |x| and |y| from the episode or handover start at the first stable upright, or at episode end without '
                          'one; lane exits = terminations with |y| > 0.15 m before recovery. Teacher prefixes excluded.',
            'bins': {name: {side: {'closed_episodes': counts[b][s],
                                   'mean_abs_x_m': sums[b][s][0] / counts[b][s] if counts[b][s] else None,
                                   'mean_abs_y_m': sums[b][s][1] / counts[b][s] if counts[b][s] else None,
                                   'lane_exits_before_recovery': lanes[b][s]}
                            for s, side in enumerate(('left', 'right'))} for b, name in enumerate(bins)}}
        falls, recoveries = self.upright_fall_events.cpu().tolist()
        result['upright_start_falls'] = {'definition': 'upright-start episodes with cos < 0.5 for >= 0.5 s, and those reaching stable upright afterwards',
                                         'episodes_with_fall': falls, 'recovered_after_fall': recoveries}
        return result

    def handover_diagnostics(self):
        """Side-start outcomes by start type; counts for handover episodes begin at handover."""
        if self.teacher is None:
            return None
        names = ('started', 'completed', 'reached_stable_upright', 'completed_after_stable_upright',
                 'completed_at_fallen_limit', 'completed_after_upright_and_at_fallen_limit')
        bins = ('official_start', 'handover_upright_below_0.2', 'handover_upright_0.2_to_0.5', 'handover_upright_at_least_0.5')
        result = {'stable_definition': 'upright cosine >=0.8 continuously for0.5s, after handover for prefix episodes',
                  'fraction_of_side_starts': self.teacher['fraction'], 'teacher_scripts': self.teacher['names'],
                  'prefix_steps': {code: table['points'].cpu().tolist() for code, table in
                                   (('left', self.teacher['sides'][1]), ('right', self.teacher['sides'][2]))},
                  'bins': {}}
        for b, name in enumerate(bins):
            result['bins'][name] = {side: dict(zip(names, values))
                                    for side, values in zip(('left', 'right'), self.handover_counts[b].cpu().tolist())}
        result['prefix_terminations'] = dict(zip(('left', 'right'), self.prefix_terminations.cpu().tolist()))
        result['attempt_seconds'] = self.attempt_steps * self.step_dt if self.attempt_steps else None
        result['attempt_truncations'] = {name: dict(zip(('left', 'right'), values)) for name, values
                                         in zip(bins, self.attempt_truncations.cpu().tolist())}
        return result

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.speed_monitor.reset_envs(env_ids)
        n = len(env_ids)
        state = self.robot.data.default_root_state[env_ids].clone()
        state[:, :3] += self.scene.env_origins[env_ids]
        state[:, 7:] = 0.
        roll = torch.zeros(n, device=self.device)
        if self.cfg.evaluation:
            roll[:] = {'upright': 0., 'left': math.pi / 2, 'right': -math.pi / 2}[self.cfg.initial_side]
        elif self.cfg.stage != 'crawl':
            fallen = torch.rand(n, device=self.device) < self.physics['recovery_start_fraction']
            signs = torch.where(torch.rand(n, device=self.device) < 0.5, -1., 1.)
            roll = fallen * signs * math.pi / 2
        state[:, 3] = torch.cos(roll / 2)
        state[:, 4] = torch.sin(roll / 2)
        state[:, 5:7] = 0.
        state[:, 2] = self.scene.env_origins[env_ids, 2] + torch.where(roll.abs() > 0.1, self.physics['side_height_m'], self.physics['initial_height_m'])
        joints = self.robot.data.default_joint_pos[env_ids].clone()
        joints[:, self.joint_ids] = self.home
        if 'settled_starts' in self.physics:
            for side, rest in self.physics['settled_starts'].items():
                mask = roll.abs() < .1 if side == 'upright' else (roll > .1 if side == 'left' else roll < -.1)
                state[mask, :7] = self.tensor(rest['root_pose'])
                state[mask, :3] += self.scene.env_origins[env_ids[mask]]
                selected = joints[mask].clone()
                selected[:, self.joint_ids] = self.tensor(rest['joint_rad'])
                joints[mask] = selected
        self.robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
        commanded = joints.clone()
        commanded[:, self.joint_ids] = self.home
        self.robot.set_joint_position_target(commanded, env_ids=env_ids)
        self.target[env_ids] = self.home
        self.applied_target[env_ids] = self.home
        self.old_target[env_ids] = self.home
        self.command_queue[env_ids] = self.home
        self.previous_q[env_ids] = joints[:, self.joint_ids]
        self.filtered_velocity[env_ids] = 0.
        self.start_pos[env_ids] = state[:, :3]
        self.start_yaw[env_ids] = self.yaw(state[:, 3:7])
        self.previous_x[env_ids] = state[:, 0]
        self.fallen_time[env_ids] = 0.
        self.initial_side[env_ids] = torch.where(roll > .1, 1, torch.where(roll < -.1, 2, 0))
        self.reset_upright[env_ids] = -quat_apply_inverse(state[:, 3:7], self.gravity_w[env_ids])[:, 2]
        self.best_upright_progress[env_ids] = 0.
        self.stable_upright_steps[env_ids] = 0
        self.reached_upright[env_ids] = False
        self.upright_fell[env_ids] = False
        self.upright_fall_recovered[env_ids] = False
        if self.teacher is not None:
            self.teacher_left[env_ids] = 0
            self.teacher_key_i[env_ids] = 0
            self.teacher_held[env_ids] = 0
            self.handover_pending[env_ids] = False
            self.handover_variant[env_ids] = -1
            self.handover_prefix[env_ids] = 0
            self.handover_cos[env_ids] = float('nan')
            self.handover_bin[env_ids] = 0
            if not self.cfg.evaluation:
                self._assign_teachers(env_ids)
        official = self.teacher_left[env_ids] == 0
        starts = torch.stack((((self.initial_side[env_ids] == 1) & official).sum(),
                              ((self.initial_side[env_ids] == 2) & official).sum()))
        self.side_episode_counts[:, 0] += starts
        self.handover_counts[0, :, 0] += starts
        self.observation_step = -1
        self._randomize(env_ids)

    def _randomize(self, env_ids):
        robust = self.cfg.stage == 'robust'
        n = len(env_ids)
        ids_cpu = env_ids.cpu()
        def uniform(bounds):
            low, high = bounds
            return low + torch.rand(n, device=self.device) * (high-low)
        self.drag[env_ids] = uniform(self.physics['drag_force_range_n']) if robust else 0.
        self.delay_steps[env_ids] = torch.randint(self.physics['max_action_delay_steps'] + 1, (n,), device=self.device) if robust else 0
        mats = self.robot.root_physx_view.get_material_properties()
        friction = uniform(self.physics['static_friction_range']) if robust else torch.full((n,), sum(self.physics['static_friction_range']) / 2, device=self.device)
        mats[ids_cpu, :, 0] = friction.cpu()[:, None]
        mats[ids_cpu, :, 1] = friction.cpu()[:, None] * self.physics['dynamic_friction_ratio']
        mats[ids_cpu, :, 2] = 0.
        self.robot.root_physx_view.set_material_properties(mats, ids_cpu)
        scales = uniform(self.physics['mass_scale_range']).cpu() if robust else torch.ones(n)
        masses = self.nominal_masses.clone()
        inertias = self.nominal_inertias.clone()
        masses[ids_cpu] *= scales[:, None]
        inertias[ids_cpu] *= scales[:, None, None]
        self.robot.root_physx_view.set_masses(masses, ids_cpu)
        self.robot.root_physx_view.set_inertias(inertias, ids_cpu)
        for range_key, nominal_key, writer, actuator_field in (
                ('effort_limit_range_nm', 'effort_limit_nm', self.robot.write_joint_effort_limit_to_sim, 'effort_limit'),
                ('velocity_limit_range_rad_s', 'velocity_limit_rad_s', self.robot.write_joint_velocity_limit_to_sim, 'velocity_limit')):
            if range_key in self.physics:
                limits = uniform(self.physics[range_key]) if robust else torch.full((n,), self.physics[nominal_key], device=self.device)
                writer(limits[:, None].expand(n, 6), joint_ids=self.joint_ids, env_ids=env_ids)
                # Implicit drive physics uses the solver limit; keep its torque
                # estimate/metadata synchronized with the same randomized cap.
                getattr(self.robot.actuators['servos'], actuator_field)[env_ids] = limits[:, None]
        gains = uniform(self.physics['gain_scale_range']) if robust else torch.ones(n, device=self.device)
        self.robot.write_joint_stiffness_to_sim((gains[:, None] * self.physics['stiffness_nm_rad']).expand(n, 6), joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.write_joint_damping_to_sim((gains[:, None] * self.physics['damping_nm_s_rad']).expand(n, 6), joint_ids=self.joint_ids, env_ids=env_ids)
