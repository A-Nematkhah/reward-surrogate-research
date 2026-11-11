"""Build feature matrices for early+math+behavioral (and ablations)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from reward_triage.store import StoreRecord

FeatureSet = Literal["math+behavioral", "early", "early+math+behavioral"]

DATA_108 = Path("data/navigation_2d/phase1_108")


def load_phase1_feature_table(
    feature_set: FeatureSet = "early+math+behavioral",
    *,
    root: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return dataframe with candidate_id, family_id, success_rate + feature cols."""
    base = root or DATA_108
    meta = pd.read_csv(base / "features" / "features_meta.csv")
    math_df = pd.read_csv(base / "features" / "features_math.csv")
    beh_df = pd.read_csv(base / "features" / "features_behavioral.csv")
    early_df = pd.read_csv(base / "features" / "features_early.csv")
    targets = pd.read_csv(base / "features" / "features_targets.csv")
    labels = pd.read_csv(base / "candidates_seed_agg.csv")[
        ["candidate_id", "success_rate"]
    ].rename(columns={"success_rate": "success_rate_agg"})

    parts = [meta]
    cols: list[str] = []
    if feature_set == "math+behavioral":
        parts.extend([math_df, beh_df])
        cols = list(math_df.columns) + list(beh_df.columns)
    elif feature_set == "early":
        parts.append(early_df)
        cols = list(early_df.columns)
    else:
        parts.extend([early_df, math_df, beh_df])
        cols = list(early_df.columns) + list(math_df.columns) + list(beh_df.columns)

    feats = pd.concat(parts + [targets[["success_rate"]]], axis=1)
    feats = feats.merge(labels, on="candidate_id", how="left")
    feats["success_rate"] = feats["success_rate_agg"].fillna(feats["success_rate"])
    feature_cols = [c for c in cols if pd.api.types.is_numeric_dtype(feats[c])]
    return feats, feature_cols


def strip_mb_prefix(feats: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in feats.items():
        name = k[3:] if k.startswith("mb_") else k
        out[name] = float(v)
    return out


def vectorize_record(
    record: StoreRecord,
    feature_cols: list[str],
) -> np.ndarray:
    """One row aligned to ``feature_cols`` (NaN if missing)."""
    flat: dict[str, float] = {}
    if record.features:
        flat.update(strip_mb_prefix(record.features))
    # Prefer averaged early from seed labels if not already in features.
    early_maps = [lab.early_features for lab in record.labels if lab.early_features]
    if early_maps:
        keys = set().union(*(m.keys() for m in early_maps))
        for key in keys:
            vals = [m[key] for m in early_maps if key in m]
            if vals:
                flat[key] = float(sum(vals) / len(vals))
    return np.array([flat.get(c, np.nan) for c in feature_cols], dtype=float)


def records_to_xy(
    records: list[StoreRecord],
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Labeled records → X, y (mean seed success), candidate_ids."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    ids: list[str] = []
    for rec in records:
        if not rec.labels:
            continue
        done = [lab for lab in rec.labels if lab.status == "completed"]
        if not done:
            continue
        xs.append(vectorize_record(rec, feature_cols))
        ys.append(float(sum(lab.success for lab in done) / len(done)))
        ids.append(rec.candidate.candidate_id)
    if not xs:
        return np.zeros((0, len(feature_cols))), np.zeros(0), []
    return np.vstack(xs), np.asarray(ys, dtype=float), ids


def pool_success_labels(records: list[StoreRecord]) -> dict[str, float]:
    out: dict[str, float] = {}
    for rec in records:
        done = [lab for lab in rec.labels if lab.status == "completed"]
        if done:
            out[rec.candidate.candidate_id] = float(
                sum(lab.success for lab in done) / len(done)
            )
    return out
