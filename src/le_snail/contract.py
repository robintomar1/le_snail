"""Versioned observations and position targets shared by training and deployment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
OBSERVATION_ORDER = (
    "joint_position_minus_home_rad", "filtered_joint_velocity_rad_s",
    "gravity_base_unit", "gyro_base_rad_s", "previous_applied_action",
)


def finite_vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


@dataclass(frozen=True)
class PolicyContract:
    schema_version: int
    joint_names: list[str]
    home_rad: list[float]
    lower_rad: list[float]
    upper_rad: list[float]
    action_scale_rad: list[float]
    max_target_speed_rad_s: list[float]
    policy_hz: float
    velocity_filter_alpha: float
    observation_order: list[str]
    quaternion_order: str
    parameters_measured: bool = False

    def __post_init__(self):
        if type(self.parameters_measured) is not bool:
            raise ValueError('parameters_measured must be a boolean')
        if self.schema_version != 1 or tuple(self.joint_names) != JOINT_NAMES:
            raise ValueError("Unsupported policy version or joint ordering")
        if tuple(self.observation_order) != OBSERVATION_ORDER or self.quaternion_order != "wxyz":
            raise ValueError("Unsupported observation/quaternion convention")
        for name in ("home_rad", "lower_rad", "upper_rad", "action_scale_rad", "max_target_speed_rad_s"):
            finite_vector(getattr(self, name), 6, name)
        lo, hi, home = map(np.asarray, (self.lower_rad, self.upper_rad, self.home_rad))
        if np.any(lo >= hi) or np.any(home < lo) or np.any(home > hi):
            raise ValueError("Home must lie inside ordered joint limits")
        if np.any(np.asarray(self.action_scale_rad) <= 0) or np.any(np.asarray(self.max_target_speed_rad_s) <= 0):
            raise ValueError("Action scales and slew limits must be positive")
        if not np.isfinite(self.policy_hz) or self.policy_hz <= 0:
            raise ValueError("policy_hz must be finite and positive")
        if not 0 < self.velocity_filter_alpha <= 1:
            raise ValueError("velocity_filter_alpha must be in (0, 1]")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @property
    def dt(self):
        return 1.0 / self.policy_hz

    def applied_action(self, target):
        return (np.asarray(target) - self.home_rad) / self.action_scale_rad

    def position_target(self, action, previous_target, dt=None):
        action = finite_vector(action, 6, "action")
        previous = finite_vector(previous_target, 6, "previous target")
        dt = self.dt if dt is None else dt
        if not np.isfinite(dt) or dt <= 0 or dt > self.dt * 1.5:
            raise ValueError("Invalid or missed control interval")
        if np.any(previous < self.lower_rad) or np.any(previous > self.upper_rad):
            raise ValueError("Previous target is outside calibrated limits")
        desired = np.asarray(self.home_rad) + np.clip(action, -1, 1) * self.action_scale_rad
        desired = np.clip(desired, self.lower_rad, self.upper_rad)
        step = np.asarray(self.max_target_speed_rad_s) * dt
        return previous + np.clip(desired - previous, -step, step)

    def observation(self, position, velocity, gravity, gyro, previous_target):
        q = finite_vector(position, 6, "position")
        qd = finite_vector(velocity, 6, "velocity")
        g = finite_vector(gravity, 3, "gravity")
        w = finite_vector(gyro, 3, "gyro")
        target = finite_vector(previous_target, 6, "previous target")
        if not 0.98 <= np.linalg.norm(g) <= 1.02:
            raise ValueError("Gravity must be a unit vector")
        return np.concatenate((q - self.home_rad, qd, g, w, self.applied_action(target))).astype(np.float32)


def rotation_matrix(wxyz):
    """Rotation from sensor/body coordinates to world coordinates."""
    q = finite_vector(wxyz, 4, "quaternion")
    norm = np.linalg.norm(q)
    if not 0.9 <= norm <= 1.1:
        raise ValueError("Quaternion is not approximately unit length")
    w, x, y, z = q / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def imu_base_vectors(quaternion_world_from_sensor, gyro_sensor, rotation_base_from_sensor):
    r = np.asarray(rotation_base_from_sensor, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r @ r.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("IMU mounting must be a proper rotation matrix")
    gravity_sensor = rotation_matrix(quaternion_world_from_sensor).T @ np.array([0., 0., -1.])
    return r @ gravity_sensor, r @ finite_vector(gyro_sensor, 3, "gyro")


class VelocityEstimator:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.position = None
        self.timestamp = None
        self.velocity = np.zeros(6)

    def update(self, position, timestamp):
        q = finite_vector(position, 6, "position")
        if not np.isfinite(timestamp):
            raise ValueError("Invalid position timestamp")
        if self.timestamp is not None:
            dt = timestamp - self.timestamp
            if dt <= 0:
                raise ValueError("Joint timestamps must increase")
            self.velocity = self.alpha * (q - self.position) / dt + (1-self.alpha) * self.velocity
        self.position, self.timestamp = q.copy(), timestamp
        return self.velocity.copy()
