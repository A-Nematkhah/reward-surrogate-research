"""Static features extracted directly from reward specifications (no training leakage)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.rewards.components import COMPONENT_NAMES
from src.rewards.reward_function import RewardSpec

# Family one-hot is optional static info; kept as features when family_id is known.
# This does not use evaluation outcomes.


def extract_static_features_from_spec(spec: RewardSpec) -> dict[str, float]:
    """Extract static features from a single reward specification."""
    weights = np.array(
        [float(spec.weights.get(name, 0.0)) for name in COMPONENT_NAMES],
        dtype=np.float64,
    )
    scales = np.array(
        [float(spec.scales.get(name, 1.0)) for name in COMPONENT_NAMES],
        dtype=np.float64,
    )
    effective = weights * scales
    abs_w = np.abs(weights)
    abs_e = np.abs(effective)

    features: dict[str, float] = {}
    for i, name in enumerate(COMPONENT_NAMES):
        features[f"w_{name}"] = float(weights[i])
        features[f"abs_w_{name}"] = float(abs_w[i])
        features[f"eff_{name}"] = float(effective[i])

    features["n_positive_weights"] = float(np.sum(weights > 0))
    features["n_negative_weights"] = float(np.sum(weights < 0))
    features["n_zero_weights"] = float(np.sum(np.isclose(weights, 0.0)))
    features["n_active_components"] = float(np.sum(~np.isclose(weights, 0.0)))

    features["weight_mean"] = float(np.mean(weights))
    features["weight_std"] = float(np.std(weights))
    features["weight_min"] = float(np.min(weights))
    features["weight_max"] = float(np.max(weights))
    features["weight_l1"] = float(np.sum(abs_w))
    features["weight_l2"] = float(np.linalg.norm(weights))
    features["effective_l1"] = float(np.sum(abs_e))

    # Ratios between important components (signed; 0-safe)
    features["ratio_progress_over_collision"] = _safe_ratio(
        weights[COMPONENT_NAMES.index("progress")],
        abs(weights[COMPONENT_NAMES.index("collision")]),
    )
    features["ratio_goal_over_collision"] = _safe_ratio(
        weights[COMPONENT_NAMES.index("goal")],
        abs(weights[COMPONENT_NAMES.index("collision")]),
    )
    features["ratio_progress_over_time"] = _safe_ratio(
        weights[COMPONENT_NAMES.index("progress")],
        abs(weights[COMPONENT_NAMES.index("time")]),
    )
    features["ratio_obstacle_over_progress"] = _safe_ratio(
        abs(weights[COMPONENT_NAMES.index("obstacle")]),
        abs(weights[COMPONENT_NAMES.index("progress")]) + 1e-8,
    )
    features["ratio_safety_over_progress"] = _safe_ratio(
        abs(weights[COMPONENT_NAMES.index("collision")])
        + abs(weights[COMPONENT_NAMES.index("obstacle")]),
        abs(weights[COMPONENT_NAMES.index("progress")]) + 1e-8,
    )
    features["ratio_goal_over_progress"] = _safe_ratio(
        abs(weights[COMPONENT_NAMES.index("goal")]),
        abs(weights[COMPONENT_NAMES.index("progress")]) + 1e-8,
    )
    features["ratio_task_over_safety"] = _safe_ratio(
        abs(weights[COMPONENT_NAMES.index("progress")])
        + abs(weights[COMPONENT_NAMES.index("goal")]),
        abs(weights[COMPONENT_NAMES.index("collision")])
        + abs(weights[COMPONENT_NAMES.index("obstacle")])
        + abs(weights[COMPONENT_NAMES.index("risk")])
        + 1e-8,
    )
    features["ratio_energy_over_progress"] = _safe_ratio(
        abs(weights[COMPONENT_NAMES.index("energy")]),
        abs(weights[COMPONENT_NAMES.index("progress")]) + 1e-8,
    )

    # Normalized weights (L1) for scale-invariant comparisons
    l1 = features["weight_l1"]
    if l1 > 0:
        for i, name in enumerate(COMPONENT_NAMES):
            features[f"w_norm_{name}"] = float(weights[i] / l1)
    else:
        for name in COMPONENT_NAMES:
            features[f"w_norm_{name}"] = 0.0

    return features


def extract_static_features_from_weights(
    weights: Mapping[str, float],
    *,
    scales: Mapping[str, float] | None = None,
) -> dict[str, float]:
    spec = RewardSpec(
        candidate_id="tmp",
        family_id="tmp",
        weights={name: float(weights.get(name, 0.0)) for name in COMPONENT_NAMES},
        scales={
            name: float((scales or {}).get(name, 1.0)) for name in COMPONENT_NAMES
        },
    )
    return extract_static_features_from_spec(spec)


def static_feature_matrix_from_dataframe(
    df: pd.DataFrame,
    *,
    include_family_one_hot: bool = True,
    family_levels: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a static feature table from a Phase 1 dataset (or candidate table).

    Expects ``weight_*`` columns. Does **not** use performance metrics.
    """
    rows: list[dict[str, float]] = []
    for _, row in df.iterrows():
        weights = {
            name: float(row[f"weight_{name}"]) if f"weight_{name}" in row.index else 0.0
            for name in COMPONENT_NAMES
        }
        feats = extract_static_features_from_weights(weights)
        rows.append(feats)

    feat_df = pd.DataFrame(rows, index=df.index)

    if include_family_one_hot and "family_id" in df.columns:
        levels = (
            list(family_levels)
            if family_levels is not None
            else sorted(df["family_id"].dropna().unique().tolist())
        )
        for family in levels:
            feat_df[f"family_{family}"] = (df["family_id"] == family).astype(float).to_numpy()

    # Guard: ensure no target leakage columns slipped in
    forbidden = {
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_goal_time",
        "mean_path_length",
        "mean_evaluation_return",
    }
    leak = forbidden.intersection(feat_df.columns)
    if leak:
        raise RuntimeError(f"Static features contain leaked metric columns: {leak}")

    return feat_df, list(feat_df.columns)


def _safe_ratio(numerator: float, denominator: float) -> float:
    denom = float(denominator)
    if abs(denom) < 1e-12:
        return 0.0
    value = float(numerator) / denom
    if not np.isfinite(value):
        return 0.0
    return value
