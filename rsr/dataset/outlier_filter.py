"""Detect unstable PPO seed runs for filter / retrain workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def flag_seed_outliers(
    dataset_df: pd.DataFrame,
    *,
    metric: str = "success_rate",
    max_deviation_from_median: float = 0.30,
    min_seed_span: float = 0.35,
    completed_only: bool = True,
) -> pd.DataFrame:
    """
    Flag seed rows that look like broken / unstable PPO runs.

    A seed is flagged when:
    - the candidate's max−min ``metric`` span ≥ ``min_seed_span``, AND
    - that seed's |metric − candidate median| ≥ ``max_deviation_from_median``.

    Returns a dataframe of flagged rows with diagnostic columns.
    """
    required = {"candidate_id", "training_seed", metric}
    missing = required - set(dataset_df.columns)
    if missing:
        raise ValueError(f"dataset missing columns: {sorted(missing)}")

    df = dataset_df.copy()
    if completed_only and "training_status" in df.columns:
        df = df[df["training_status"] == "completed"].copy()

    rows: list[dict[str, Any]] = []
    for cand_id, grp in df.groupby("candidate_id", sort=True):
        values = grp[metric].astype(float)
        if values.size < 2 or values.isna().all():
            continue
        med = float(np.nanmedian(values.to_numpy()))
        span = float(np.nanmax(values.to_numpy()) - np.nanmin(values.to_numpy()))
        if span < float(min_seed_span):
            continue
        for _, row in grp.iterrows():
            val = float(row[metric])
            if not np.isfinite(val):
                continue
            dev = abs(val - med)
            if dev < float(max_deviation_from_median):
                continue
            rows.append(
                {
                    "candidate_id": str(cand_id),
                    "family_id": row.get("family_id"),
                    "training_seed": int(row["training_seed"]),
                    "metric": metric,
                    "metric_value": val,
                    "candidate_median": med,
                    "candidate_span": span,
                    "abs_dev_from_median": dev,
                    "reason": "seed_far_from_median_under_large_span",
                }
            )
    return pd.DataFrame(rows)


def drop_flagged_seeds(
    dataset_df: pd.DataFrame,
    flagged: pd.DataFrame,
) -> pd.DataFrame:
    """Return dataset without (candidate_id, training_seed) pairs listed in flagged."""
    if flagged is None or flagged.empty:
        return dataset_df.copy()
    keys = flagged[["candidate_id", "training_seed"]].astype(
        {"candidate_id": str, "training_seed": int}
    )
    out = dataset_df.copy()
    out["_cid"] = out["candidate_id"].astype(str)
    out["_seed"] = out["training_seed"].astype(int)
    ban = set(zip(keys["candidate_id"].astype(str), keys["training_seed"].astype(int)))
    mask = [ (c, s) not in ban for c, s in zip(out["_cid"], out["_seed"]) ]
    out = out.loc[mask].drop(columns=["_cid", "_seed"])
    return out.reset_index(drop=True)


def summarize_flags(flagged: pd.DataFrame) -> dict[str, Any]:
    if flagged is None or flagged.empty:
        return {"n_flagged_seeds": 0, "n_candidates": 0}
    return {
        "n_flagged_seeds": int(len(flagged)),
        "n_candidates": int(flagged["candidate_id"].nunique()),
        "mean_abs_dev": float(flagged["abs_dev_from_median"].mean()),
        "mean_span": float(flagged["candidate_span"].mean()),
    }
