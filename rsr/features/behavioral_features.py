"""Behavioral probe features: how a reward scores fixed reference trajectories."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rsr.features.probe_trajectories import ProbeEpisode
from rsr.rewards.components import COMPONENT_NAMES
from rsr.rewards.reward_function import RewardSpec, WeightedRewardFunction

QUANTILES = (0.1, 0.5, 0.9)


def _score_transitions(
    reward_fn: WeightedRewardFunction,
    transitions: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    totals: list[float] = []
    comps: dict[str, list[float]] = {name: [] for name in COMPONENT_NAMES}
    for transition in transitions:
        total = float(reward_fn(None, transition))
        totals.append(total)
        for name in COMPONENT_NAMES:
            comps[name].append(float(reward_fn.last_components[name]))
    return (
        np.asarray(totals, dtype=np.float64),
        {k: np.asarray(v, dtype=np.float64) for k, v in comps.items()},
    )


def _summarize_series(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        out = {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_pct_zero": 0.0,
        }
        for q in QUANTILES:
            out[f"{prefix}_q{int(q * 100)}"] = 0.0
        return out

    out = {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_pct_zero": float(np.mean(np.isclose(values, 0.0))),
    }
    for q in QUANTILES:
        out[f"{prefix}_q{int(q * 100)}"] = float(np.quantile(values, q))
    return out


def extract_behavioral_features(
    spec: RewardSpec,
    episodes: Sequence[ProbeEpisode],
) -> dict[str, float]:
    """
    Score fixed probe trajectories with ``spec`` and summarize components.

    Does not use PPO training outcomes or evaluation metrics.
    """
    reward_fn = WeightedRewardFunction.from_spec(spec)
    features: dict[str, float] = {}

    all_totals: list[float] = []
    all_comps: dict[str, list[float]] = {name: [] for name in COMPONENT_NAMES}
    per_policy_totals: dict[str, list[float]] = {}

    for episode in episodes:
        totals, comps = _score_transitions(reward_fn, episode.transitions)
        all_totals.extend(totals.tolist())
        for name in COMPONENT_NAMES:
            all_comps[name].extend(comps[name].tolist())
        per_policy_totals.setdefault(episode.policy_name, []).extend(totals.tolist())

    totals_arr = np.asarray(all_totals, dtype=np.float64)
    features.update(_summarize_series(totals_arr, "total"))

    # Component summaries and contribution ratios
    means = {}
    for name in COMPONENT_NAMES:
        arr = np.asarray(all_comps[name], dtype=np.float64)
        features.update(_summarize_series(arr, f"comp_{name}"))
        means[name] = float(np.mean(arr)) if arr.size else 0.0

    abs_means = {k: abs(v) for k, v in means.items()}
    denom = sum(abs_means.values()) + 1e-12
    for name in COMPONENT_NAMES:
        features[f"contrib_ratio_{name}"] = abs_means[name] / denom
        # Weighted contribution under this candidate's weights
        features[f"weighted_mean_{name}"] = float(
            spec.weights.get(name, 0.0) * spec.scales.get(name, 1.0) * means[name]
        )

    # Pairwise correlations between component streams (upper triangle)
    if totals_arr.size >= 2:
        mat = np.column_stack([all_comps[name] for name in COMPONENT_NAMES])
        # Guard constant columns
        for i, name_i in enumerate(COMPONENT_NAMES):
            for j, name_j in enumerate(COMPONENT_NAMES):
                if j <= i:
                    continue
                col_i = mat[:, i]
                col_j = mat[:, j]
                if np.std(col_i) < 1e-12 or np.std(col_j) < 1e-12:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(col_i, col_j)[0, 1])
                    if not np.isfinite(corr):
                        corr = 0.0
                features[f"corr_{name_i}_{name_j}"] = corr

    for policy_name, vals in per_policy_totals.items():
        arr = np.asarray(vals, dtype=np.float64)
        features.update(_summarize_series(arr, f"policy_{policy_name}_total"))

    # Sanity: no metric leakage keys
    forbidden_substrings = (
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_goal_time",
        "ppo_",
    )
    for key in features:
        for bad in forbidden_substrings:
            if bad in key:
                raise RuntimeError(f"Behavioral feature leakage detected: {key}")

    return features


def behavioral_feature_matrix(
    specs: Sequence[RewardSpec],
    episodes: Sequence[ProbeEpisode],
) -> pd.DataFrame:
    rows = [extract_behavioral_features(spec, episodes) for spec in specs]
    return pd.DataFrame(rows)
