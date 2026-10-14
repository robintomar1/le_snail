"""Acceptance is based on displacement and recovery, never training reward."""
from dataclasses import dataclass
import math


@dataclass
class Trial:
    kind: str = "crawl"
    timeout_s: float = 120.0
    lane_half_width_m: float = 0.15
    started: bool = False
    finished: bool = False
    success: bool = False
    reason: str = "running"
    recovery_time_s: float | None = None
    recovery_x: float | None = None
    upright_since: float | None = None
    max_lateral_m: float = 0.0
    last_time_s: float = -1.0

    def __post_init__(self):
        if self.kind not in ("crawl", "left", "right"):
            raise ValueError("Trial kind must be crawl, left or right")
        if self.kind != "crawl":
            self.timeout_s = 30.0

    def update(self, elapsed_s, forward_m, lateral_m, upright_cos, *, fault=False):
        if self.finished:
            return self.success
        values = (elapsed_s, forward_m, lateral_m, upright_cos)
        if not all(math.isfinite(x) for x in values) or fault:
            self.finished, self.reason = True, "invalid_state_or_fault"
            return False
        if elapsed_s < 0 or elapsed_s <= self.last_time_s:
            raise ValueError("Trial time must strictly increase")
        self.last_time_s = elapsed_s
        if not self.started:
            self.started = True
            if self.kind != "crawl" and upright_cos > 0.5:
                self.finished, self.reason = True, "recovery_did_not_start_fallen"
                return False
        self.max_lateral_m = max(self.max_lateral_m, abs(lateral_m))
        if self.max_lateral_m > self.lane_half_width_m:
            self.finished, self.reason = True, "left_lane"
        elif elapsed_s > self.timeout_s:
            self.finished, self.reason = True, "timeout"
        else:
            # Require a half-second stable posture to avoid crediting a transient roll.
            if upright_cos >= 0.8:
                if self.upright_since is None:
                    self.upright_since = elapsed_s
                if self.recovery_time_s is None and elapsed_s - self.upright_since >= 0.5:
                    self.recovery_time_s, self.recovery_x = elapsed_s, forward_m
            else:
                self.upright_since = None
            stable = self.upright_since is not None and elapsed_s - self.upright_since >= 0.5
            if self.kind == "crawl":
                reached = forward_m >= 0.5
            else:
                reached = self.recovery_x is not None and forward_m - self.recovery_x >= 0.05
            if reached and stable:
                self.finished, self.success, self.reason = True, True, "passed"
            elif elapsed_s >= self.timeout_s:
                self.finished, self.reason = True, "timeout"
        return self.success


def summarize(trials):
    groups = {}
    for kind in ("crawl", "left", "right"):
        selected = [t for t in trials if t.kind == kind]
        groups[kind] = {"trials": len(selected), "successes": sum(t.success for t in selected)}
        groups[kind]["passed"] = len(selected) == 5 and all(t.finished for t in selected) and groups[kind]["successes"] >= 3
    return {"groups": groups, "passed": all(g["passed"] for g in groups.values())}
