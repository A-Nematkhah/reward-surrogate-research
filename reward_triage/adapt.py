"""Online surrogate adaptation after each generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reward_triage.feature_matrix import (
    FeatureSet,
    load_phase1_feature_table,
    records_to_xy,
)
from reward_triage.model import SurrogateModel
from reward_triage.store import StoreRecord

# Cached prior matrix for the locked 2D feature set (process-local).
_PRIOR_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_prior_xy(
    feature_cols: list[str],
    *,
    feature_set: FeatureSet = "early+math+behavioral",
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load phase1_108 X,y aligned to ``feature_cols`` (NaN if a col missing)."""
    key = f"{feature_set}:{','.join(feature_cols[:3])}:{len(feature_cols)}"
    if key in _PRIOR_CACHE:
        return _PRIOR_CACHE[key]
    feats, cols = load_phase1_feature_table(feature_set, root=root)
    # Align to surrogate schema.
    X = np.column_stack(
        [
            feats[c].to_numpy(dtype=float) if c in feats.columns else np.full(len(feats), np.nan)
            for c in feature_cols
        ]
    )
    y = feats["success_rate"].to_numpy(dtype=float)
    _PRIOR_CACHE[key] = (X, y)
    return X, y


def adapt_after_generation(
    surrogate: SurrogateModel,
    records: list[StoreRecord],
    *,
    out_dir: Path,
    generation: int,
    use_prior: bool = True,
    data_root: Path | None = None,
    min_new_labels: int = 1,
) -> dict[str, Any]:
    """Fit surrogate on prior (+ optional) plus all labeled records; save versioned artifact.

    Returns a small summary dict. If not enough new completed labels, skips fit.
    """
    X_new, y_new, ids = records_to_xy(records, surrogate.feature_cols)
    n_new = int(len(y_new))
    if n_new < min_new_labels:
        return {
            "adapted": False,
            "reason": "insufficient_new_labels",
            "n_new": n_new,
            "generation": generation,
        }

    X_prior = y_prior = None
    if use_prior:
        X_prior, y_prior = load_prior_xy(
            surrogate.feature_cols,
            feature_set=surrogate.feature_set,  # type: ignore[arg-type]
            root=data_root,
        )

    surrogate.adapt_with_prior(
        X_new,
        y_new,
        X_prior=X_prior,
        y_prior=y_prior,
        dataset_hash=f"gen{generation}_new={n_new}",
    )
    art = out_dir / f"gen_{generation}"
    surrogate.save(art)
    return {
        "adapted": True,
        "generation": generation,
        "n_new_labels": n_new,
        "n_prior": 0 if y_prior is None else int(len(y_prior)),
        "n_train_total": int(surrogate.meta.n_train) if surrogate.meta else None,
        "artifact": str(art),
        "candidate_ids": ids,
    }
