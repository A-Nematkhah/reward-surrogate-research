"""Task-performance metrics separate from training return."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.environment.navigation_2d import (
    REASON_COLLISION,
    REASON_OUT_OF_BOUNDS,
    REASON_SUCCESS,
    REASON_TIMEOUT,
)

METRIC_NAMES = (
    "success_rate",
    "collision_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "mean_episode_length",
    "mean_path_length",
    "mean_goal_time",
    "mean_min_obstacle_distance",
    "mean_evaluation_return",
)


def aggregate_episode_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """
    Aggregate per-episode evaluation records into summary metrics.

    ``mean_goal_time`` is the mean episode length over successful episodes only.
    If there are no successes, it is NaN (not zero), to avoid misleading values.
    """
    if not episodes:
        raise ValueError("Cannot aggregate metrics over an empty episode list")

    n = len(episodes)
    reasons = [str(ep.get("termination_reason", "")) for ep in episodes]

    success = sum(r == REASON_SUCCESS for r in reasons)
    collision = sum(r == REASON_COLLISION for r in reasons)
    oob = sum(r == REASON_OUT_OF_BOUNDS for r in reasons)
    timeout = sum(r == REASON_TIMEOUT for r in reasons)

    lengths = np.array([float(ep["episode_length"]) for ep in episodes], dtype=np.float64)
    paths = np.array([float(ep["path_length"]) for ep in episodes], dtype=np.float64)
    returns = np.array([float(ep["episode_return"]) for ep in episodes], dtype=np.float64)

    clearances = []
    for ep in episodes:
        val = ep.get("min_obstacle_clearance")
        if val is not None and np.isfinite(val):
            clearances.append(float(val))

    goal_times = [
        float(ep["episode_length"])
        for ep, reason in zip(episodes, reasons)
        if reason == REASON_SUCCESS
    ]

    metrics = {
        "success_rate": success / n,
        "collision_rate": collision / n,
        "out_of_bounds_rate": oob / n,
        "timeout_rate": timeout / n,
        "mean_episode_length": float(np.mean(lengths)),
        "mean_path_length": float(np.mean(paths)),
        "mean_goal_time": float(np.mean(goal_times)) if goal_times else float("nan"),
        "mean_min_obstacle_distance": (
            float(np.mean(clearances)) if clearances else float("nan")
        ),
        "mean_evaluation_return": float(np.mean(returns)),
        "n_episodes": float(n),
        "n_success": float(success),
        "n_collision": float(collision),
        "n_out_of_bounds": float(oob),
        "n_timeout": float(timeout),
    }
    return metrics


def episodes_to_dataframe(episodes: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(episodes))


def metrics_to_serializable(metrics: Mapping[str, float]) -> dict[str, Any]:
    """Convert metrics to JSON-friendly values (NaN -> None)."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            out[key] = None
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out
