"""Hybrid scripted controller (#14): crawl gait, roll-based fall detection, scripted side get-up, resume.

A stateful torch.nn.Module that TorchScript can compile; it needs only torch (no Isaac, no scipy), so the
laptop deploy runtime can run it like a learned policy. Input: the 24-value policy observation
[q - home, filtered joint velocity, gravity in the base frame, gyro, (applied target - home) / scale].
Output: normalized actions in [-1, 1]; the environment or hardware applies the same clamp, joint limits
and slew as for a learned policy. Call reset() at the start of every episode.

Modes:
  GAIT     open-loop periodic gait a(t) = clip(c + r(t) A w_k(m 2 pi t / T + psi)), r(t) = min(1, t / ramp)
  SETTLE   fallen: command home until still, then pick the side from gravity (g_y < 0: lying on the left)
  GETUP    the confirmed keyframe plan for that side; a key advances once the applied target has reached it
           for advance_hold_s (final home key: final_hold_s), exactly as in the validated scripted get-ups
  CONFIRM  command home; stable upright (cos >= 0.8 for hold_s) resumes the gait from phase 0; still fallen
           after timeout_s retries with the other variant, up to max_attempts, then ABORT (hold home)
A fall is |roll| > roll_deg for roll_hold_s, or upright cosine < cos for cos_hold_s.
"""
import json
import math
from pathlib import Path

import torch

GAIT, SETTLE, GETUP, CONFIRM, ABORT, INIT = 0, 1, 2, 3, 4, 5
MODE_NAMES = ('gait', 'settle', 'getup', 'confirm', 'abort', 'init')


class HybridController(torch.nn.Module):
    def __init__(self, config: dict, contract: dict):
        super().__init__()
        dt = 1. / float(contract['policy_hz'])
        steps = lambda seconds: max(1, int(round(seconds / dt)))
        gait, getup, fall, settle, confirm = (config[k] for k in ('gait', 'getup', 'fall', 'settle', 'confirm'))
        self.dt = dt
        self.GAIT, self.SETTLE, self.GETUP, self.CONFIRM, self.ABORT, self.INIT = GAIT, SETTLE, GETUP, CONFIRM, ABORT, INIT
        self.register_buffer('home', torch.tensor(contract['home_rad'], dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(contract['action_scale_rad'], dtype=torch.float32))
        self.register_buffer('lower', torch.tensor(contract['lower_rad'], dtype=torch.float32))
        self.register_buffer('upper', torch.tensor(contract['upper_rad'], dtype=torch.float32))
        self.register_buffer('gait_c', torch.tensor(gait['c'], dtype=torch.float32))
        self.register_buffer('gait_a', torch.tensor(gait['A'], dtype=torch.float32))
        self.register_buffer('gait_psi', torch.tensor(gait['psi'], dtype=torch.float32))
        self.register_buffer('gait_m', torch.tensor(gait['m'], dtype=torch.float32))
        self.gait_period = float(gait['T'])
        self.gait_k = float(gait['k'])
        self.gait_ramp = float(gait['ramp_s'])
        # keys[side, variant, key, joint] in radians; side 0 = left, 1 = right; the last key is home.
        keys = [[variant for variant in getup['keys_rad'][side]] for side in ('left', 'right')]
        self.register_buffer('keys', torch.tensor(keys, dtype=torch.float32))
        self.num_variants = int(self.keys.shape[1])
        self.last_key = int(self.keys.shape[2]) - 1
        self.advance_steps = steps(getup['advance_hold_s'])
        self.final_steps = steps(getup['final_hold_s'])
        self.reached_tol = float(getup['reached_tol_rad'])
        self.roll_limit = math.radians(float(fall['roll_deg']))
        self.roll_steps = steps(fall['roll_hold_s'])
        self.fall_cos = float(fall['cos'])
        self.cos_steps = steps(fall['cos_hold_s'])
        self.settle_min = steps(settle['min_s'])
        self.settle_max = steps(settle['max_s'])
        self.still_gyro = float(settle['gyro_rad_s'])
        self.still_steps = steps(settle['still_s'])
        self.upright_cos = float(confirm['cos'])
        self.upright_steps = steps(confirm['hold_s'])
        self.confirm_steps = steps(confirm['timeout_s'])
        self.fallen_cos = float(confirm['fallen_cos'])
        self.max_attempts = int(config['max_attempts'])
        z = torch.zeros(1, dtype=torch.long)
        self.register_buffer('mode', z.clone())
        self.register_buffer('gait_time', torch.zeros(1))
        self.register_buffer('timer', z.clone())
        self.register_buffer('count', z.clone())
        self.register_buffer('roll_count', z.clone())
        self.register_buffer('cos_count', z.clone())
        self.register_buffer('side', z.clone())
        self.register_buffer('attempts', z.clone())
        self.register_buffer('key', z.clone())
        self.register_buffer('held', z.clone())
        self.register_buffer('getups', z.clone())

    @torch.jit.export
    def reset(self):
        for buffer in (self.timer, self.count, self.roll_count, self.cos_count, self.side, self.attempts,
                       self.key, self.held, self.getups):
            buffer.zero_()
        self.gait_time.zero_()
        self.mode.fill_(self.INIT)

    def reset_envs(self, env_ids: torch.Tensor):
        """Start a new episode for some rows of a batched controller (training-time teacher prefixes)."""
        for buffer in (self.timer, self.count, self.roll_count, self.cos_count, self.side, self.attempts,
                       self.key, self.held, self.getups):
            buffer[env_ids] = 0
        self.gait_time[env_ids] = 0.
        self.mode[env_ids] = self.INIT

    def _ensure(self, n: int, device: torch.device):
        if self.mode.shape[0] != n or self.mode.device != device:
            z = torch.zeros(n, dtype=torch.long, device=device)
            self.mode = torch.full((n,), self.INIT, dtype=torch.long, device=device)
            self.gait_time = torch.zeros(n, device=device)
            self.timer, self.count, self.roll_count, self.cos_count = z.clone(), z.clone(), z.clone(), z.clone()
            self.side, self.attempts, self.key, self.held, self.getups = z.clone(), z.clone(), z.clone(), z.clone(), z.clone()

    def gait_action(self, t: torch.Tensor) -> torch.Tensor:
        x = 2 * math.pi * t[:, None] / self.gait_period * self.gait_m + self.gait_psi
        s = torch.sin(x)
        if self.gait_k > 0.05:
            w = torch.tanh(self.gait_k * s) / math.tanh(self.gait_k)
        else:
            w = s
        ramp = torch.clamp(t / self.gait_ramp, max=1.)[:, None]
        return torch.clamp(self.gait_c + ramp * self.gait_a * w, -1., 1.)

    def key_action(self) -> torch.Tensor:
        variant = self.attempts % self.num_variants
        goal = self.keys[self.side, variant, torch.clamp(self.key, max=self.last_key)]
        return torch.clamp((goal - self.home) / self.scale, -1., 1.)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n = obs.shape[0]
        self._ensure(n, obs.device)
        gravity, gyro = obs[:, 12:15], obs[:, 15:18]
        applied = obs[:, 18:24] * self.scale + self.home
        cos = -gravity[:, 2]
        roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
        one = torch.ones_like(self.timer)
        self.roll_count = torch.where(roll.abs() > self.roll_limit, self.roll_count + 1, 0 * one)
        self.cos_count = torch.where(cos < self.fall_cos, self.cos_count + 1, 0 * one)
        fall = (self.roll_count >= self.roll_steps) | (self.cos_count >= self.cos_steps)

        # INIT: upright enough starts the gait; otherwise settle first.
        init = self.mode == self.INIT
        self.mode = torch.where(init & (cos >= self.fallen_cos), self.GAIT * one, torch.where(init, self.SETTLE * one, self.mode))
        self.timer = torch.where(init, 0 * one, self.timer)
        self.count = torch.where(init, 0 * one, self.count)

        # GAIT -> SETTLE on a fall.
        to_settle = (self.mode == self.GAIT) & fall
        self.mode = torch.where(to_settle, self.SETTLE * one, self.mode)
        self.timer = torch.where(to_settle, 0 * one, self.timer)
        self.count = torch.where(to_settle, 0 * one, self.count)

        # SETTLE: hold home until still (gyro and applied target), then choose the side and start the get-up.
        settling = self.mode == self.SETTLE
        at_home = (applied - self.home).abs().amax(dim=1) < self.reached_tol
        still = (torch.linalg.vector_norm(gyro, dim=1) < self.still_gyro) & at_home
        self.timer = torch.where(settling, self.timer + 1, self.timer)
        self.count = torch.where(settling & still, self.count + 1, torch.where(settling, 0 * one, self.count))
        settled = settling & (((self.timer >= self.settle_min) & (self.count >= self.still_steps)) | (self.timer >= self.settle_max))
        upright_now = settled & (cos >= self.fallen_cos)
        start_getup = settled & ~upright_now
        self.mode = torch.where(upright_now, self.CONFIRM * one, torch.where(start_getup, self.GETUP * one, self.mode))
        self.side = torch.where(start_getup, (gravity[:, 1] >= 0).long(), self.side)
        self.key = torch.where(start_getup, 0 * one, self.key)
        self.held = torch.where(start_getup, 0 * one, self.held)
        self.getups = torch.where(start_getup, self.getups + 1, self.getups)
        self.timer = torch.where(upright_now, 0 * one, self.timer)
        self.count = torch.where(upright_now, 0 * one, self.count)

        # GETUP: advance a key once the applied target has reached it for long enough (checked for the key
        # commanded on the previous step; a freshly started plan has not been commanded yet).
        getting_up = (self.mode == self.GETUP) & ~start_getup
        desired = torch.clamp(self.home + self.key_action() * self.scale, self.lower, self.upper)
        reached = (applied - desired).abs().amax(dim=1) < self.reached_tol
        self.held = torch.where(getting_up & reached, self.held + 1, torch.where(getting_up, 0 * one, self.held))
        need = torch.where(self.key == self.last_key, self.final_steps * one, self.advance_steps * one)
        advance = getting_up & (self.held >= need)
        self.key = torch.where(advance, self.key + 1, self.key)
        self.held = torch.where(advance, 0 * one, self.held)
        done = (self.mode == self.GETUP) & (self.key > self.last_key)
        self.mode = torch.where(done, self.CONFIRM * one, self.mode)
        self.timer = torch.where(done, 0 * one, self.timer)
        self.count = torch.where(done, 0 * one, self.count)

        # CONFIRM: stable upright resumes the gait; still fallen at the timeout retries, else resume anyway.
        confirming = (self.mode == self.CONFIRM) & ~done & ~upright_now
        self.timer = torch.where(confirming, self.timer + 1, self.timer)
        self.count = torch.where(confirming & (cos >= self.upright_cos), self.count + 1, torch.where(confirming, 0 * one, self.count))
        stable = confirming & (self.count >= self.upright_steps)
        timeout = confirming & ~stable & (self.timer >= self.confirm_steps)
        retry = timeout & (cos < self.fallen_cos)
        self.attempts = torch.where(retry, self.attempts + 1, self.attempts)
        give_up = retry & (self.attempts >= self.max_attempts)
        resume = stable | (timeout & ~retry)
        self.mode = torch.where(resume, self.GAIT * one, self.mode)
        self.mode = torch.where(retry & ~give_up, self.SETTLE * one, self.mode)
        self.mode = torch.where(give_up, self.ABORT * one, self.mode)
        self.gait_time = torch.where(resume, torch.zeros_like(self.gait_time), self.gait_time)
        self.timer = torch.where(retry, 0 * one, self.timer)
        self.count = torch.where(retry, 0 * one, self.count)

        # Actions.
        gait = self.mode == self.GAIT
        action = torch.zeros((n, 6), device=obs.device)
        action = torch.where(gait[:, None], self.gait_action(self.gait_time), action)   # first gait step at t = 0
        action = torch.where((self.mode == self.GETUP)[:, None], self.key_action(), action)
        self.gait_time = torch.where(gait, self.gait_time + self.dt, self.gait_time)
        return action

    @torch.jit.export
    def status(self) -> torch.Tensor:
        """Per environment: mode, side, attempts, key, get-ups started, gait time (s)."""
        return torch.stack((self.mode.float(), self.side.float(), self.attempts.float(), self.key.float(),
                            self.getups.float(), self.gait_time), dim=1)


def load_controller(config_path, contract_path) -> HybridController:
    config = json.loads(Path(config_path).read_text())
    contract = json.loads(Path(contract_path).read_text())
    return HybridController(config, contract)
