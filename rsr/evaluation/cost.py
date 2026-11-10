"""PPO step cost accounting for triage (features for all N + full runs for kept k)."""

from __future__ import annotations

import math
from typing import Any


def triage_cost(
    n: int,
    k: int,
    *,
    feature_steps_per_candidate: int,
    full_run_steps: int,
) -> dict[str, float]:
    """
    Cost model used by Phase 1 / Gate 1 reports.

    cost = feature_steps * N + full_run_steps * k
    baseline_all = full_run_steps * N
    savings_frac = 1 - cost / baseline_all
    """
    n = int(n)
    k = int(k)
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    if k > n:
        k = n
    feat = int(feature_steps_per_candidate)
    full = int(full_run_steps)
    if feat < 0 or full < 0:
        raise ValueError("step counts must be non-negative")

    cost = feat * n + full * k
    baseline = full * n
    savings = (baseline - cost) / baseline if baseline > 0 else float("nan")
    return {
        "n": float(n),
        "k": float(k),
        "feature_steps_per_candidate": float(feat),
        "full_run_steps": float(full),
        "cost_steps": float(cost),
        "baseline_steps_train_all": float(baseline),
        "savings_frac": float(savings),
        "meets_50pct_savings": (
            float(savings >= 0.50) if math.isfinite(savings) else float("nan")
        ),
    }


def compare_to_train_all(cost_info: dict[str, Any]) -> str:
    sav = float(cost_info["savings_frac"])
    if not math.isfinite(sav):
        return "undefined"
    return f"saves {100.0 * sav:.1f}% of full-run steps vs training all N"
