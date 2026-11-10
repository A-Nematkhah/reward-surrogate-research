"""Policy evaluation and task metrics."""

from __future__ import annotations

from typing import Any

from rsr.evaluation.metrics import (
    METRIC_NAMES,
    aggregate_episode_metrics,
    metrics_to_serializable,
)

__all__ = [
    "METRIC_NAMES",
    "aggregate_episode_metrics",
    "evaluate_policy",
    "metrics_to_serializable",
    "resolve_eval_seeds",
]


def __getattr__(name: str) -> Any:
    if name in {"evaluate_policy", "resolve_eval_seeds"}:
        from rsr.evaluation.evaluator import evaluate_policy, resolve_eval_seeds

        return {
            "evaluate_policy": evaluate_policy,
            "resolve_eval_seeds": resolve_eval_seeds,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
