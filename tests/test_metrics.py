"""Tests for evaluation metric aggregation."""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import aggregate_episode_metrics, metrics_to_serializable


def _ep(**kwargs):
    base = {
        "termination_reason": "timeout",
        "episode_length": 10,
        "path_length": 5.0,
        "episode_return": 1.0,
        "min_obstacle_clearance": 0.5,
    }
    base.update(kwargs)
    return base


def test_aggregate_rates() -> None:
    episodes = [
        _ep(termination_reason="success", episode_length=8),
        _ep(termination_reason="collision", episode_length=4),
        _ep(termination_reason="timeout", episode_length=20),
        _ep(termination_reason="out_of_bounds", episode_length=6),
    ]
    metrics = aggregate_episode_metrics(episodes)
    assert metrics["success_rate"] == pytest.approx(0.25)
    assert metrics["collision_rate"] == pytest.approx(0.25)
    assert metrics["timeout_rate"] == pytest.approx(0.25)
    assert metrics["out_of_bounds_rate"] == pytest.approx(0.25)
    assert metrics["mean_goal_time"] == pytest.approx(8.0)
    assert metrics["mean_episode_length"] == pytest.approx(9.5)


def test_mean_goal_time_nan_when_no_success() -> None:
    episodes = [_ep(termination_reason="timeout"), _ep(termination_reason="collision")]
    metrics = aggregate_episode_metrics(episodes)
    assert math.isnan(metrics["mean_goal_time"])
    serial = metrics_to_serializable(metrics)
    assert serial["mean_goal_time"] is None


def test_empty_episodes_raise() -> None:
    with pytest.raises(ValueError):
        aggregate_episode_metrics([])
