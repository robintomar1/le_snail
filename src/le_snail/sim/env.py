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
from le_snail.physics import load_physics


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
        self.previous_x = self.start_pos[:, 0].clone()
        self.fallen_time = torch.zeros(self.num_envs, device=self.device)
        self.drag = torch.zeros(self.num_envs, device=self.device)
        self.gravity_w = self.tensor([0., 0., -1.]).repeat(self.num_envs, 1)
        self.base_indices_wp = wp.array(self.base_ids, dtype=wp.int32, device=self.device)
        self.nominal_masses = self.robot.root_physx_view.get_masses().clone()
        self.nominal_inertias = self.robot.root_physx_view.get_inertias().clone()
        self.probes = []
        for probe in self.physics['protected_points']:
            ids, _ = self.robot.find_bodies(probe['body'])
            if len(ids) != 1:
                raise ValueError(f'Protected probe body is ambiguous: {probe}')
            self.probes.append((ids[0], self.tensor(probe['position_m']).repeat(self.num_envs, 1), float(probe['radius_m'])))

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
        self.old_target.copy_(self.target)
        desired = torch.clamp(self.home + torch.clamp(actions, -1., 1.) * self.scale, self.lower, self.upper)
        self.target += torch.clamp(desired - self.target, -self.slew, self.slew)
        self.command_queue[:, 1] = self.command_queue[:, 0]
        self.command_queue[:, 0] = self.target
        self.previous_x = self.robot.data.root_pos_w[:, 0].clone()

    def _apply_action(self):
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
        costs = 0.3 * displacement[:, 1].square() + 0.1 * (1-upright) + 0.02 * command_rate.square().sum(dim=1) + 0.02
        return 30. * progress - self.step_dt * costs - self.protected_contact().float()

    def _get_dones(self):
        self.last_transition = (self.robot.data.root_pos_w.clone(), self.robot.data.projected_gravity_b.clone(), self.protected_contact().clone())
        displacement = self.robot.data.root_pos_w - self.start_pos
        upright = -self.robot.data.projected_gravity_b[:, 2]
        self.fallen_time = torch.where(upright < 0.5, self.fallen_time + self.step_dt, 0.)
        invalid = ~torch.isfinite(self.robot.data.root_state_w).all(dim=1)
        invalid |= ~torch.isfinite(self.robot.data.joint_pos).all(dim=1)
        terminated = invalid | self.protected_contact() | (displacement[:, 1].abs() > 0.15) | (self.robot.data.root_pos_w[:, 2] < -0.03)
        terminated |= self.fallen_time > self.physics['max_fallen_time_s']
        # DirectRLEnv increments the counter before calling _get_dones.
        timeout = self.episode_length_buf >= self.max_episode_length
        if self.render_mode == 'rgb_array' and bool((terminated | timeout).any()):
            # Capture before DirectRLEnv auto-resets so evaluation videos retain
            # the same terminal transition used by the acceptance scorer.
            self.terminal_frame = self.render().copy()
        # Ordinary falls do not terminate episodes. Evaluation ends on its independent acceptance checks.
        return terminated, timeout

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
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
        self.robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
        self.robot.set_joint_position_target(joints, env_ids=env_ids)
        self.target[env_ids] = self.home
        self.applied_target[env_ids] = self.home
        self.old_target[env_ids] = self.home
        self.command_queue[env_ids] = self.home
        self.previous_q[env_ids] = self.home
        self.filtered_velocity[env_ids] = 0.
        self.start_pos[env_ids] = state[:, :3]
        self.previous_x[env_ids] = state[:, 0]
        self.fallen_time[env_ids] = 0.
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
        gains = uniform(self.physics['gain_scale_range']) if robust else torch.ones(n, device=self.device)
        self.robot.write_joint_stiffness_to_sim((gains[:, None] * self.physics['stiffness_nm_rad']).expand(n, 6), joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.write_joint_damping_to_sim((gains[:, None] * self.physics['damping_nm_s_rad']).expand(n, 6), joint_ids=self.joint_ids, env_ids=env_ids)
