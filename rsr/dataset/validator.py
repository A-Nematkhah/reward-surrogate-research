"""Validate the Phase 1 reward–performance dataset."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rsr.dataset.builder import METRIC_COLUMNS, WEIGHT_COLUMNS
from rsr.utils.logging import get_logger

logger = get_logger("dataset.validator")

REQUIRED_COLUMNS = [
    "candidate_id",
    "family_id",
    "training_seed",
    "training_status",
    "total_timesteps",
    *WEIGHT_COLUMNS,
]


class DatasetValidationError(ValueError):
    """Raised when the dataset fails hard integrity checks."""


def validate_dataset(
    df: pd.DataFrame,
    *,
    min_completed: int = 1,
    min_families: int = 2,
    require_metric_diversity: bool = True,
    success_rate_min_span: float = 0.05,
) -> dict[str, Any]:
    """
    Validate dataset completeness and basic research usefulness.

    Returns a report dict. Raises ``DatasetValidationError`` for fatal issues
    (missing columns, zero completed rows when ``min_completed`` > 0).
    """
    report: dict[str, Any] = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    if df.empty:
        report["ok"] = False
        report["errors"].append("Dataset is empty")
        raise DatasetValidationError("Dataset is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report["ok"] = False
        report["errors"].append(f"Missing required columns: {missing}")
        raise DatasetValidationError(report["errors"][-1])

    n_total = len(df)
    n_completed = int((df["training_status"] == "completed").sum())
    n_failed = int((df["training_status"] == "failed").sum())
    n_other = n_total - n_completed - n_failed
    families = sorted(df["family_id"].dropna().unique().tolist())
    candidates = sorted(df["candidate_id"].dropna().unique().tolist())

    report["stats"] = {
        "n_rows": n_total,
        "n_completed": n_completed,
        "n_failed": n_failed,
        "n_other": n_other,
        "n_families": len(families),
        "n_candidates": len(candidates),
        "families": families,
    }

    if n_failed > 0:
        report["warnings"].append(
            f"{n_failed} failed training runs are present in the dataset "
            "(documented, not silently dropped)."
        )

    if n_completed < min_completed:
        report["ok"] = False
        msg = f"Only {n_completed} completed rows; need at least {min_completed}"
        report["errors"].append(msg)
        raise DatasetValidationError(msg)

    if len(families) < min_families:
        report["warnings"].append(
            f"Only {len(families)} families present; expected >= {min_families} "
            "for grouped generalization tests."
        )

    completed = df[df["training_status"] == "completed"].copy()

    # Completed rows must have metrics
    for col in METRIC_COLUMNS:
        if col not in completed.columns:
            report["ok"] = False
            report["errors"].append(f"Missing metric column: {col}")
            continue
        # mean_goal_time / clearance may be NaN; others should be finite
        if col in ("mean_goal_time", "mean_min_obstacle_distance"):
            continue
        bad = completed[col].isna() | ~np.isfinite(completed[col].astype(float))
        if bad.any():
            report["ok"] = False
            report["errors"].append(
                f"Non-finite values in completed rows for '{col}': {int(bad.sum())}"
            )

    # Duplicate (candidate, seed) pairs
    dup = completed.duplicated(subset=["candidate_id", "training_seed"], keep=False)
    if dup.any():
        report["warnings"].append(
            f"Duplicate completed (candidate_id, training_seed) rows: {int(dup.sum())}"
        )

    # Diversity checks
    if require_metric_diversity and len(completed) >= 2:
        sr = completed["success_rate"].astype(float)
        span = float(sr.max() - sr.min())
        report["stats"]["success_rate_span"] = span
        report["stats"]["success_rate_mean"] = float(sr.mean())
        report["stats"]["success_rate_std"] = float(sr.std(ddof=0))
        if span < success_rate_min_span:
            report["warnings"].append(
                f"Success-rate span across completed runs is only {span:.4f} "
                f"(<{success_rate_min_span}). Candidates may behave too similarly; "
                "investigate reward scales, difficulty, or training budget before Phase 2."
            )
        else:
            report["stats"]["has_performance_diversity"] = True

        # Seed stability (where multiple seeds exist)
        seed_counts = completed.groupby("candidate_id")["training_seed"].nunique()
        multi = seed_counts[seed_counts > 1]
        if len(multi) > 0:
            spreads = []
            for cid in multi.index:
                vals = completed.loc[
                    completed["candidate_id"] == cid, "success_rate"
                ].astype(float)
                spreads.append(float(vals.max() - vals.min()))
            report["stats"]["mean_seed_success_spread"] = float(np.mean(spreads))
            report["stats"]["max_seed_success_spread"] = float(np.max(spreads))

    if report["errors"]:
        report["ok"] = False
        raise DatasetValidationError("; ".join(report["errors"]))

    logger.info(
        "Dataset validation OK: completed=%d failed=%d families=%d candidates=%d",
        n_completed,
        n_failed,
        len(families),
        len(candidates),
    )
    return report
