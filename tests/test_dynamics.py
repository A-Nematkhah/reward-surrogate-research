"""Unit tests for kinematic / geometric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from rsr.environment.dynamics import (
    Circle,
    Workspace,
    circles_collide,
    clearance_to_circle,
    distance,
    integrate_velocity,
    is_valid_placement,
    nearest_obstacles,
    normalize_action,
    point_circle_collision,
    sample_point_in_workspace,
)


def test_normalize_action_clips() -> None:
    a = normalize_action(np.array([2.0, -3.0]))
    np.testing.assert_allclose(a, [1.0, -1.0])


def test_normalize_action_rejects_nan() -> None:
    with pytest.raises(ValueError):
        normalize_action(np.array([np.nan, 0.0]))


def test_integrate_velocity() -> None:
    pos = np.array([0.0, 0.0])
    new_pos, vel = integrate_velocity(pos, np.array([1.0, 0.0]), max_speed=2.0, dt=1.0)
    np.testing.assert_allclose(vel, [2.0, 0.0])
    np.testing.assert_allclose(new_pos, [2.0, 0.0])


def test_workspace_contains_and_bounds() -> None:
    ws = Workspace(width=10.0, height=10.0, centered=True)
    assert ws.contains(np.array([0.0, 0.0]))
    assert not ws.contains(np.array([6.0, 0.0]))
    assert ws.x_min == -5.0
    assert ws.x_max == 5.0


def test_collision_and_clearance() -> None:
    c = Circle(0.0, 0.0, 1.0)
    assert point_circle_collision(np.array([0.5, 0.0]), c, point_radius=0.1)
    assert not point_circle_collision(np.array([2.0, 0.0]), c, point_radius=0.1)
    assert clearance_to_circle(np.array([2.0, 0.0]), c, point_radius=0.1) == pytest.approx(
        0.9
    )
    assert circles_collide(Circle(0.0, 0.0, 1.0), Circle(1.5, 0.0, 1.0))


def test_nearest_obstacles_ordering() -> None:
    obstacles = [
        Circle(3.0, 0.0, 0.5),
        Circle(1.0, 0.0, 0.5),
        Circle(2.0, 0.0, 0.5),
    ]
    nearest = nearest_obstacles(np.array([0.0, 0.0]), obstacles, n=2)
    assert len(nearest) == 2
    assert nearest[0][0].x == 1.0
    assert nearest[1][0].x == 2.0


def test_sample_point_in_workspace_reproducible() -> None:
    ws = Workspace(10.0, 10.0, centered=True)
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    pa = sample_point_in_workspace(rng_a, ws, margin=0.5)
    pb = sample_point_in_workspace(rng_b, ws, margin=0.5)
    np.testing.assert_allclose(pa, pb)
    assert ws.contains(pa, margin=0.5)


def test_is_valid_placement() -> None:
    ws = Workspace(10.0, 10.0, centered=True)
    others = [Circle(0.0, 0.0, 1.0)]
    assert not is_valid_placement(
        np.array([0.5, 0.0]),
        0.5,
        workspace=ws,
        others=others,
        min_clearance=0.5,
    )
    assert is_valid_placement(
        np.array([3.0, 0.0]),
        0.5,
        workspace=ws,
        others=others,
        min_clearance=0.5,
    )


def test_distance() -> None:
    assert distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(5.0)
