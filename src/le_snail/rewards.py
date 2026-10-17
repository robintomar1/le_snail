"""Two-portion reward: standing up and moving forward (#6). Pure torch; no simulator.

Per policy step, in physical units, identical in every episode and after every fall:
  standing up = w_phase * (gamma * Phi(s') - Phi(s)) + w_stand * dt * u^2
  forward     = w_fwd * gate * max(0, x - x_record)              new ground while upright (and crawling)
  effort      = -w_effort * dt * sum(command_rate^2) * gate       smoothness while upright only
Phi = max(progress, u^2). Progress is the kernel-regressed get-up phase over the k nearest states of a
bank of successful scripted get-ups (standardized base-frame gravity + joint angles), shrunk to 0 far
from the bank. u = ramp(cos, standing ramp), gate = ramp(cos, forward gate). The potential difference
uses the training discount, so any closed loop nets exactly zero discounted reward; a terminal
transition uses Phi(s') = 0 and the first step of an episode has no shaping.

Optional refinements (#15; absent keys keep the earlier behaviour exactly):
  uprightness 'roll_pitch': u and gate use e = cos(roll) * ramp(cos(pitch), pitch_bound) instead of the total
      tilt cosine, so front-to-back rocking within the pitch bound is not charged (every good scripted gait
      rocks 40-55 deg in pitch with roll ~0) while sideways tilt and large pitch are
  lane_per_s:     -w * dt * ramp(|y - y_start|, lane_band_m)                   always
  heading_per_s:  -w * dt * ramp(|yaw|, heading_band_deg) * gate               world heading against +x
      (heading_requires_crawl_mode: also * crawling, so get-ups and the turn they leave are not charged)
  contact_per_s:  -w * dt * [a non-support link touches the ground] * gate     support = base and gripper
"""
import math

import numpy as np
import torch


def ramp(value, low, high):
    return ((value - low) / (high - low)).clamp(0., 1.)


class ProgressBank:
    def __init__(self, path, device):
        data = np.load(path)
        # Float64 throughout: float32 on the GPU picked different neighbours near ties (~1e-4 in Phi,
        # verification 003), while float64 matches an independent CPU reference exactly.
        self.states = torch.as_tensor(data['bank_states'], dtype=torch.float64, device=device)
        self.phases = torch.as_tensor(data['phases'], dtype=torch.float64, device=device)
        self.mean = torch.as_tensor(data['mean'], dtype=torch.float64, device=device)
        self.std = torch.as_tensor(data['std'], dtype=torch.float64, device=device)
        self.sigma, self.k, self.shrink = float(data['sigma']), int(data['k']), float(data['shrink'])

    def progress(self, features, chunk=2048):
        """Get-up phase estimate in [0, 1) for rows of [gravity_b (3), joint angles (6)]."""
        z = (features.double() - self.mean) / self.std
        out = torch.empty(len(z), dtype=torch.float64, device=z.device)
        for start in range(0, len(z), chunk):
            q = z[start:start + chunk]
            # Direct differences: the |q|^2 + |b|^2 - 2 q.b expansion cancels badly exactly where the
            # nearest neighbours are, which made the potential jitter.
            d2 = torch.cdist(q, self.states, compute_mode='donot_use_mm_for_euclid_dist').square()
            near, index = torch.topk(d2, self.k, dim=1, largest=False)
            weight = torch.exp(-near / (2 * self.sigma ** 2))
            out[start:start + chunk] = (weight * self.phases[index]).sum(1) / (weight.sum(1) + self.shrink)
        return out.float()


def uprightness(config, gravity):
    """Uprightness used by the standing and forward portions: the tilt cosine, or roll-based with a pitch bound."""
    if config.get('uprightness', 'cos') == 'cos':
        return -gravity[:, 2]
    if config['uprightness'] != 'roll_pitch':
        raise ValueError(f"Unknown uprightness {config['uprightness']}")
    low, high = config['pitch_bound_deg']   # full credit up to low, none beyond high
    cos_pitch = gravity[:, 1:].norm(dim=1)   # unit gravity: cos(pitch) = |(g_y, g_z)|
    cos_roll = -gravity[:, 2] / cos_pitch.clamp_min(1e-6)
    return cos_roll * ramp(cos_pitch, math.cos(math.radians(high)), math.cos(math.radians(low)))


def two_portion_terms(config, bank, *, upright, features, x, x_record, previous_phi, terminated, command_rate, dt, gamma,
                      crawling=None, gravity=None, lateral=None, yaw=None, other_contact=None):
    """Reward components for one step plus the potential and running x record to carry forward.

    With forward_requires_crawl_mode, new ground is paid only while crawling: after a stable upright
    (cos >= 0.8 for 0.5 s) and until the next fall below cos 0.5. A lunge during a failed get-up or a
    fall is never paid, because x_record still advances.
    """
    level = upright if config.get('uprightness', 'cos') == 'cos' else uprightness(config, gravity)
    stand = ramp(level, *config['standing_ramp_cos']) ** 2
    gate = ramp(level, *config['forward_gate_cos'])
    forward_gate = gate * crawling.float() if config.get('forward_requires_crawl_mode', False) else gate
    phi = torch.maximum(bank.progress(features), stand)
    following = torch.where(terminated, torch.zeros_like(phi), phi)
    shaping = config['getup_progress_weight'] * (gamma * following - previous_phi)
    shaping = torch.where(torch.isfinite(previous_phi), shaping, torch.zeros_like(shaping))
    terms = {'getup_progress': shaping,
             'standing': config['standing_per_s'] * dt * stand,
             'forward': config['forward_per_m'] * forward_gate * (x - x_record).clamp_min(0.),
             'effort': -config['effort_weight'] * dt * command_rate.square().sum(dim=1) * gate}
    if config.get('lane_per_s'):
        terms['lane'] = -config['lane_per_s'] * dt * ramp(lateral.abs(), *config['lane_band_m'])
    if config.get('heading_per_s'):
        low, high = (math.radians(v) for v in config['heading_band_deg'])
        heading_gate = gate * crawling.float() if config.get('heading_requires_crawl_mode', False) else gate
        terms['heading'] = -config['heading_per_s'] * dt * ramp(yaw.abs(), low, high) * heading_gate
    if config.get('contact_per_s'):
        terms['contact'] = -config['contact_per_s'] * dt * other_contact.float() * gate
    return terms, phi, torch.maximum(x_record, x)
