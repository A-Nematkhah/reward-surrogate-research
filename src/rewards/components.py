"""Interpretable reward components for 2D navigation.

Each component maps a transition dict to a finite scalar. Components do not
apply weights — weighting is handled by ``WeightedRewardFunction``.

Components only use information available at (s, a, s'): no future access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np

from src.environment.dynamics import Circle, clearance_to_circle

# Linear baseline set + structurally distinct extensions for family diversity.
COMPONENT_NAMES = (
    "progress",
    "goal",
    "collision",
    "obstacle",
    "smoothness",
    "time",
    "energy",
    "risk",
    "potential",
    "nonlinear_progress",
    "distance",
)

LINEAR_COMPONENT_NAMES = (
    "progress",
    "goal",
    "collision",
    "obstacle",
    "smoothness",
    "time",
)


class RewardComponent(ABC):
    """Common interface for a single reward term."""

    name: str

    @abstractmethod
    def compute(self, transition: Mapping[str, Any]) -> float:
        """Return an unweighted component value (must be finite)."""

    def __call__(self, transition: Mapping[str, Any]) -> float:
        value = float(self.compute(transition))
        if not np.isfinite(value):
            raise RuntimeError(
                f"Reward component '{self.name}' produced non-finite value: {value}"
            )
        return value


class ProgressComponent(RewardComponent):
    """Improvement in distance to goal: d_prev - d_curr (positive when closer)."""

    name = "progress"

    def compute(self, transition: Mapping[str, Any]) -> float:
        prev_d = float(transition["prev_goal_distance"])
        curr_d = float(transition["goal_distance"])
        return prev_d - curr_d


class GoalComponent(RewardComponent):
    """Binary success indicator (1 if goal reached this step, else 0)."""

    name = "goal"

    def compute(self, transition: Mapping[str, Any]) -> float:
        return 1.0 if transition.get("success") else 0.0


class CollisionComponent(RewardComponent):
    """Binary failure indicator for collision or leaving the workspace."""

    name = "collision"

    def compute(self, transition: Mapping[str, Any]) -> float:
        if transition.get("collision") or transition.get("out_of_bounds"):
            return 1.0
        return 0.0


class ObstacleProximityComponent(RewardComponent):
    """
    Soft proximity penalty based on clearance to the nearest obstacle.

    value = max(0, 1 - clearance / safe_distance)

    At contact (clearance <= 0) the value is 1; beyond ``safe_distance`` it is 0.
    Scale is O(1) so weights remain interpretable.
    """

    name = "obstacle"

    def __init__(self, safe_distance: float = 1.0) -> None:
        if safe_distance <= 0:
            raise ValueError(f"safe_distance must be positive, got {safe_distance}")
        self.safe_distance = float(safe_distance)

    def compute(self, transition: Mapping[str, Any]) -> float:
        min_clearance = _min_obstacle_clearance(transition)
        if min_clearance is None:
            return 0.0
        return float(np.clip(1.0 - min_clearance / self.safe_distance, 0.0, 1.0))


class SmoothnessComponent(RewardComponent):
    """
    Action-change magnitude: ||a_t - a_{t-1}||_2 / sqrt(2).

    Normalized to [0, ~1] given actions in [-1, 1]^2.
    """

    name = "smoothness"

    def compute(self, transition: Mapping[str, Any]) -> float:
        action = np.asarray(transition["action"], dtype=np.float64).reshape(2)
        prev_action = np.asarray(transition["prev_action"], dtype=np.float64).reshape(2)
        delta = action - prev_action
        return float(np.linalg.norm(delta) / np.sqrt(2.0))


class TimeComponent(RewardComponent):
    """Constant per-step cost indicator (1.0 every step)."""

    name = "time"

    def compute(self, transition: Mapping[str, Any]) -> float:
        del transition
        return 1.0


class EnergyComponent(RewardComponent):
    """Action energy ||a||^2 / 2 in [0, 1] for actions in [-1, 1]^2."""

    name = "energy"

    def compute(self, transition: Mapping[str, Any]) -> float:
        action = np.asarray(transition["action"], dtype=np.float64).reshape(2)
        return float(np.sum(action * action) / 2.0)


class RiskComponent(RewardComponent):
    """
    Asymmetric near-obstacle risk: squared intrusion inside a risk radius.

    Structurally different from linear obstacle proximity (thresholded quadratic).
    """

    name = "risk"

    def __init__(self, risk_radius: float = 0.75) -> None:
        if risk_radius <= 0:
            raise ValueError(f"risk_radius must be positive, got {risk_radius}")
        self.risk_radius = float(risk_radius)

    def compute(self, transition: Mapping[str, Any]) -> float:
        min_clearance = _min_obstacle_clearance(transition)
        if min_clearance is None:
            return 0.0
        intrusion = max(0.0, self.risk_radius - float(min_clearance))
        return float((intrusion / self.risk_radius) ** 2)


class PotentialComponent(RewardComponent):
    """
    Potential-based shaping: Φ(d_prev) - Φ(d_curr) with Φ(d) = exp(-d).

    Encourages decrease of goal distance under a nonlinear potential.
    """

    name = "potential"

    def compute(self, transition: Mapping[str, Any]) -> float:
        prev_d = float(transition["prev_goal_distance"])
        curr_d = float(transition["goal_distance"])
        return float(np.exp(-prev_d) - np.exp(-curr_d))


class NonlinearProgressComponent(RewardComponent):
    """Signed squared progress: sign(Δd) * (Δd)^2 where Δd = d_prev - d_curr."""

    name = "nonlinear_progress"

    def compute(self, transition: Mapping[str, Any]) -> float:
        prev_d = float(transition["prev_goal_distance"])
        curr_d = float(transition["goal_distance"])
        delta = prev_d - curr_d
        return float(np.sign(delta) * (delta ** 2))


class DistanceComponent(RewardComponent):
    """Dense absolute distance cost: current goal distance (positive scalar)."""

    name = "distance"

    def compute(self, transition: Mapping[str, Any]) -> float:
        return float(transition["goal_distance"])


def _min_obstacle_clearance(transition: Mapping[str, Any]) -> float | None:
    position = np.asarray(transition["position"], dtype=np.float64)
    robot_radius = float(transition.get("robot_radius", 0.0))
    obstacles = transition.get("obstacles") or []
    if not obstacles:
        return None
    clearances: list[float] = []
    for obs in obstacles:
        if isinstance(obs, Circle):
            circle = obs
        else:
            circle = Circle(float(obs[0]), float(obs[1]), float(obs[2]))
        clearances.append(
            clearance_to_circle(position, circle, point_radius=robot_radius)
        )
    return float(min(clearances))


def build_default_components(
    *,
    obstacle_safe_distance: float = 1.0,
    risk_radius: float = 0.75,
) -> dict[str, RewardComponent]:
    """Create the standard component registry keyed by name."""
    components: list[RewardComponent] = [
        ProgressComponent(),
        GoalComponent(),
        CollisionComponent(),
        ObstacleProximityComponent(safe_distance=obstacle_safe_distance),
        SmoothnessComponent(),
        TimeComponent(),
        EnergyComponent(),
        RiskComponent(risk_radius=risk_radius),
        PotentialComponent(),
        NonlinearProgressComponent(),
        DistanceComponent(),
    ]
    registry = {c.name: c for c in components}
    missing = set(COMPONENT_NAMES) - set(registry)
    if missing:
        raise RuntimeError(f"Missing default components: {sorted(missing)}")
    return registry


def compute_all_components(
    transition: Mapping[str, Any],
    components: Mapping[str, RewardComponent] | None = None,
) -> dict[str, float]:
    """Evaluate every registered component on ``transition``."""
    registry = components or build_default_components()
    values = {name: registry[name](transition) for name in COMPONENT_NAMES}
    for name, value in values.items():
        if not np.isfinite(value):
            raise RuntimeError(f"Non-finite component '{name}': {value}")
    return values
