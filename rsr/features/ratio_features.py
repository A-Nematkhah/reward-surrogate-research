"""Scale-aware reward weight ratios for within-family discrimination.

Inductive features only (no labels, no other candidates). Complements the
smaller ratio set already embedded in ``static_features``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from rsr.rewards.components import COMPONENT_NAMES
from rsr.rewards.reward_function import RewardSpec


def _safe_ratio(num: float, den: float) -> float:
    d = float(den)
    if abs(d) < 1e-12:
        return 0.0
    v = float(num) / d
    return float(v) if np.isfinite(v) else 0.0


def _safe_log_abs_ratio(num: float, den: float) -> float:
    d = abs(float(den)) + 1e-8
    n = abs(float(num)) + 1e-8
    v = float(np.log(n / d))
    return v if np.isfinite(v) else 0.0


def extract_ratio_features_from_weights(weights: Mapping[str, float]) -> dict[str, float]:
    w = {name: float(weights.get(name, 0.0)) for name in COMPONENT_NAMES}
    aw = {k: abs(v) for k, v in w.items()}
    l1 = float(sum(aw.values()))
    feats: dict[str, float] = {}

    # Core tradeoffs
    pairs = [
        ("progress", "collision"),
        ("goal", "collision"),
        ("progress", "time"),
        ("goal", "time"),
        ("progress", "obstacle"),
        ("goal", "obstacle"),
        ("progress", "smoothness"),
        ("progress", "energy"),
        ("progress", "risk"),
        ("goal", "risk"),
        ("collision", "obstacle"),
        ("obstacle", "smoothness"),
        ("potential", "progress"),
        ("nonlinear_progress", "progress"),
        ("distance", "progress"),
        ("safety_mass", "progress"),  # synthetic
    ]
    safety = aw["collision"] + aw["obstacle"] + aw["risk"]
    for a, b in pairs:
        if a == "safety_mass":
            na, nb = safety, aw["progress"]
        else:
            na, nb = aw[a], aw[b]
        feats[f"ratio_{a}_over_{b}"] = _safe_ratio(na, nb)
        feats[f"log_ratio_{a}_over_{b}"] = _safe_log_abs_ratio(na, nb)

    # Signed progress vs penalties
    feats["signed_progress_minus_safety"] = float(w["progress"] + w["goal"] - safety)
    feats["signed_goal_minus_collision"] = float(w["goal"] - aw["collision"])

    # L1 shares (scale-invariant composition)
    if l1 > 0:
        for name in COMPONENT_NAMES:
            feats[f"share_{name}"] = float(aw[name] / l1)
        feats["share_safety"] = float(safety / l1)
        feats["share_task"] = float((aw["progress"] + aw["goal"]) / l1)
    else:
        for name in COMPONENT_NAMES:
            feats[f"share_{name}"] = 0.0
        feats["share_safety"] = 0.0
        feats["share_task"] = 0.0

    feats["ratio_task_over_safety"] = _safe_ratio(
        aw["progress"] + aw["goal"], safety + 1e-8
    )
    return feats


def extract_ratio_features_from_spec(spec: RewardSpec) -> dict[str, float]:
    return extract_ratio_features_from_weights(spec.weights)


def ratio_feature_matrix_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        weights = {
            name: float(row[f"weight_{name}"]) if f"weight_{name}" in row.index else 0.0
            for name in COMPONENT_NAMES
        }
        rows.append(extract_ratio_features_from_weights(weights))
    return pd.DataFrame(rows)
