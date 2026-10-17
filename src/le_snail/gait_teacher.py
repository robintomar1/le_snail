"""hybrid-001's crawl gait, locked to the applied target instead of a clock (#15). Torch only; no simulator.

The hybrid controller plays its gait a(t) open loop from a clock. A learned policy that should crawl the same way
sees no clock, so to label any state it reaches (DAgger) the gait needs a phase read off the state itself. This
module keeps the gait's steady-state limit cycle of applied targets, sampled densely from the contract's exact
action path (clip, scale, home, joint limits, 1 rad/s slew), and each step:
  - finds the cycle sample nearest the applied target in the observation (searching only a short window ahead
    of the previous phase once it has one, so the near-crossings of the cycle do not make it jump), then
  - commands the gait action one policy step later, a(phase * T + dt).
Played closed loop, it reproduces the open-loop gait's targets once on the cycle. From any other pose (home after
a reset, or wherever a student policy drifted) it joins the cycle at the nearest point, at the slew limit.
Stateful like the hybrid controller: reset() each episode, or reset_envs(ids).
"""
import json
import math
from pathlib import Path

import torch


class PhaseGait(torch.nn.Module):
    def __init__(self, config: dict, contract: dict, periods: int = 60, window_back: float = 0.03, window_ahead: float = 0.2):
        super().__init__()
        gait = config['gait']
        self.dt = 1. / float(contract['policy_hz'])
        self.period = float(gait['T'])
        self.k = float(gait['k'])
        self.window_back, self.window_ahead = window_back, window_ahead
        self.register_buffer('home', torch.tensor(contract['home_rad'], dtype=torch.float64))
        self.register_buffer('scale', torch.tensor(contract['action_scale_rad'], dtype=torch.float64))
        self.register_buffer('c', torch.tensor(gait['c'], dtype=torch.float64))
        self.register_buffer('amplitude', torch.tensor(gait['A'], dtype=torch.float64))
        self.register_buffer('psi', torch.tensor(gait['psi'], dtype=torch.float64))
        self.register_buffer('m', torch.tensor(gait['m'], dtype=torch.float64))
        lower = torch.tensor(contract['lower_rad'], dtype=torch.float64)
        upper = torch.tensor(contract['upper_rad'], dtype=torch.float64)
        step = torch.tensor(contract['max_target_speed_rad_s'], dtype=torch.float64) * self.dt
        # Steady-state cycle: full amplitude from home, discard the first 5 periods, keep the rest. The policy period
        # is not a whole fraction of T, so the kept samples cover the phase densely.
        n = int(round((periods + 5) * self.period / self.dt))
        t = torch.arange(n, dtype=torch.float64) * self.dt
        actions = self._gait(t)
        target = self.home.clone()
        targets = torch.empty((n, 6), dtype=torch.float64)
        for i in range(n):
            goal = torch.clamp(self.home + actions[i] * self.scale, lower, upper)
            target = target + torch.clamp(goal - target, -step, step)
            targets[i] = target
        keep = t >= 5 * self.period
        self.register_buffer('cycle_phase', ((t[keep] / self.period) % 1.).float())
        self.register_buffer('cycle_target', ((targets[keep] - self.home) / self.scale).float())   # as in obs[18:24]
        self.register_buffer('phase', torch.zeros(0))
        self.register_buffer('locked', torch.zeros(0, dtype=torch.bool))

    def _gait(self, t: torch.Tensor) -> torch.Tensor:
        x = 2 * math.pi * t[:, None] / self.period * self.m + self.psi
        w = torch.tanh(self.k * torch.sin(x)) / math.tanh(self.k)
        return torch.clamp(self.c + self.amplitude * w, -1., 1.)

    @torch.jit.export
    def reset(self):
        self.phase = torch.zeros(0, device=self.phase.device)
        self.locked = torch.zeros(0, dtype=torch.bool, device=self.locked.device)

    @torch.jit.export
    def reset_envs(self, env_ids: torch.Tensor):
        if self.locked.numel() > 0:
            self.locked[env_ids] = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n = obs.shape[0]
        if self.locked.numel() != n:
            self.phase = torch.zeros(n, device=obs.device)
            self.locked = torch.zeros(n, dtype=torch.bool, device=obs.device)
        applied = obs[:, 18:24]
        distance = (applied[:, None, :] - self.cycle_target[None, :, :]).square().sum(dim=2)
        ahead = (self.cycle_phase[None, :] - self.phase[:, None]) % 1.
        inside = (ahead <= self.window_ahead) | (ahead >= 1. - self.window_back)
        allowed = inside | ~self.locked[:, None]
        distance = torch.where(allowed, distance, torch.full_like(distance, float('inf')))
        index = torch.argmin(distance, dim=1)
        self.phase = self.cycle_phase[index]
        self.locked = torch.ones_like(self.locked)
        return self._gait(self.phase.double() * self.period + self.dt).float()

    @torch.jit.export
    def current_phase(self) -> torch.Tensor:
        return self.phase


def load_phase_gait(config_path, contract_path) -> PhaseGait:
    return PhaseGait(json.loads(Path(config_path).read_text()), json.loads(Path(contract_path).read_text()))
