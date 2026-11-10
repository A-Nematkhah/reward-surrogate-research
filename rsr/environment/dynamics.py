"""Kinematic dynamics and geometric helpers for 2D navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Circle:
    """Axis-aligned world circle (obstacle, robot footprint, or target)."""

    x: float
    y: float
    radius: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    def contains_point(self, point: np.ndarray, *, margin: float = 0.0) -> bool:
        delta = point.astype(np.float64) - self.center
        return float(np.linalg.norm(delta)) <= self.radius + margin


@dataclass(frozen=True)
class Workspace:
    """Axis-aligned rectangular workspace, optionally centered at origin."""

    width: float
    height: float
    centered: bool = True

    @property
    def x_min(self) -> float:
        return -0.5 * self.width if self.centered else 0.0

    @property
    def x_max(self) -> float:
        return 0.5 * self.width if self.centered else self.width

    @property
    def y_min(self) -> float:
        return -0.5 * self.height if self.centered else 0.0

    @property
    def y_max(self) -> float:
        return 0.5 * self.height if self.centered else self.height

    @property
    def half_extents(self) -> np.ndarray:
        return np.array([0.5 * self.width, 0.5 * self.height], dtype=np.float64)

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    def contains(self, point: np.ndarray, *, margin: float = 0.0) -> bool:
        x, y = float(point[0]), float(point[1])
        return (
            self.x_min + margin <= x <= self.x_max - margin
            and self.y_min + margin <= y <= self.y_max - margin
        )

    def clip(self, point: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.clip(point[0], self.x_min, self.x_max),
                np.clip(point[1], self.y_min, self.y_max),
            ],
            dtype=np.float64,
        )


def normalize_action(action: np.ndarray) -> np.ndarray:
    """Clip a 2D action to [-1, 1]."""
    action = np.asarray(action, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(action)):
        raise ValueError(f"Action contains non-finite values: {action}")
    return np.clip(action, -1.0, 1.0)


def integrate_velocity(
    position: np.ndarray,
    action: np.ndarray,
    *,
    max_speed: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple kinematic step: velocity = action * max_speed, position += velocity * dt.

    Returns
    -------
    new_position, velocity
    """
    if max_speed <= 0:
        raise ValueError(f"max_speed must be positive, got {max_speed}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    unit_action = normalize_action(action)
    velocity = unit_action * float(max_speed)
    new_position = np.asarray(position, dtype=np.float64).reshape(2) + velocity * dt
    return new_position, velocity


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def circles_collide(a: Circle, b: Circle, *, margin: float = 0.0) -> bool:
    return distance(a.center, b.center) <= (a.radius + b.radius + margin)


def point_circle_collision(
    point: np.ndarray,
    circle: Circle,
    *,
    point_radius: float = 0.0,
    margin: float = 0.0,
) -> bool:
    return distance(point, circle.center) <= (circle.radius + point_radius + margin)


def clearance_to_circle(
    point: np.ndarray,
    circle: Circle,
    *,
    point_radius: float = 0.0,
) -> float:
    """Signed clearance: positive means separated, negative means penetration."""
    return distance(point, circle.center) - (circle.radius + point_radius)


def nearest_obstacles(
    point: np.ndarray,
    obstacles: Sequence[Circle],
    *,
    n: int,
    point_radius: float = 0.0,
) -> list[tuple[Circle, float]]:
    """
    Return up to ``n`` nearest obstacles as (obstacle, center-distance) pairs,
    sorted by increasing center distance.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    scored = [
        (obs, distance(point, obs.center))
        for obs in obstacles
    ]
    scored.sort(key=lambda item: item[1])
    return scored[:n]


def sample_point_in_workspace(
    rng: np.random.Generator,
    workspace: Workspace,
    *,
    margin: float,
) -> np.ndarray:
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if 2 * margin >= workspace.width or 2 * margin >= workspace.height:
        raise ValueError("margin too large for workspace")
    x = rng.uniform(workspace.x_min + margin, workspace.x_max - margin)
    y = rng.uniform(workspace.y_min + margin, workspace.y_max - margin)
    return np.array([x, y], dtype=np.float64)


def is_valid_placement(
    center: np.ndarray,
    radius: float,
    *,
    workspace: Workspace,
    others: Sequence[Circle],
    min_clearance: float,
) -> bool:
    """Check that a circle fits in the workspace with clearance from others."""
    if not workspace.contains(center, margin=radius):
        return False
    candidate = Circle(float(center[0]), float(center[1]), radius)
    for other in others:
        # Require geometric separation plus extra clearance
        if circles_collide(candidate, other, margin=min_clearance):
            return False
    return True
