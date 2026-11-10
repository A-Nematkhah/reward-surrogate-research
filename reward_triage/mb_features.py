"""Gate-1 ``math+behavioral`` features from a sandbox reward scorer.

Port of ``rsr.features.math_features`` / ``behavioral_features`` without
importing torch. Scorer signature: ``transition -> (total, components|None)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

# Full registry (matches rsr.rewards.components.COMPONENT_NAMES).
COMPONENT_NAMES: tuple[str, ...] = (
    "progress",
    "goal",
    "collision",
    "obstacle",
    "smoothness",
    "time",
    "energy",
    "risk",
    "potential",
    "nonlinear_progress",
    "distance",
    "human_proximity",
    "human_intrusion",
    "crowded_progress",
)

# Gate-1 exported math+behavioral tables used this 11-component set
# (human_* / crowded_progress were added later and are absent from CSVs).
GATE1_BEHAVIORAL_COMPONENTS: tuple[str, ...] = (
    "progress",
    "goal",
    "collision",
    "obstacle",
    "smoothness",
    "time",
    "energy",
    "risk",
    "potential",
    "nonlinear_progress",
    "distance",
)

EPS = 1e-8
DIST_DELTA = 1e-2
ACTION_DELTA = 1e-2
QUANTILES = (0.1, 0.5, 0.9)

ScoreFn = Callable[[Mapping[str, Any]], tuple[float, Mapping[str, float] | None]]


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


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


def _score_total(score_fn: ScoreFn, transition: Mapping[str, Any]) -> float:
    total, _ = score_fn(transition)
    return float(total)


def extract_math_from_scorer(
    score_fn: ScoreFn,
    episodes: Sequence[Sequence[Mapping[str, Any]]],
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Math landscape features (same formulas as Gate 1)."""
    totals: list[float] = []
    step_deltas: list[float] = []
    grad_mags: list[float] = []
    dist_sens: list[float] = []

    for transitions in episodes:
        prev_total: float | None = None
        for transition in transitions:
            total = _score_total(score_fn, transition)
            totals.append(total)
            if prev_total is not None:
                step_deltas.append(abs(total - prev_total))
            prev_total = total

            t_plus = dict(transition)
            t_minus = dict(transition)
            t_plus["goal_distance"] = float(transition["goal_distance"]) + DIST_DELTA
            t_minus["goal_distance"] = float(transition["goal_distance"]) - DIST_DELTA
            r_plus = _score_total(score_fn, t_plus)
            r_minus = _score_total(score_fn, t_minus)
            dist_sens.append(abs(r_plus - r_minus) / (2.0 * DIST_DELTA))

            action = np.asarray(transition["action"], dtype=np.float64).reshape(2)
            g2 = 0.0
            for axis in range(2):
                a_plus = action.copy()
                a_minus = action.copy()
                a_plus[axis] += ACTION_DELTA
                a_minus[axis] -= ACTION_DELTA
                tp = dict(transition)
                tm = dict(transition)
                tp["action"] = a_plus
                tm["action"] = a_minus
                g = (_score_total(score_fn, tp) - _score_total(score_fn, tm)) / (
                    2.0 * ACTION_DELTA
                )
                g2 += float(g) ** 2
            grad_mags.append(float(np.sqrt(g2)))

    arr = np.asarray(totals, dtype=np.float64)
    wsrc = dict(weights or {})
    # Gate-1 math_n_terms_active counts non-zero weights over the exported set.
    n_active = float(
        sum(not np.isclose(float(wsrc.get(n, 0.0)), 0.0) for n in GATE1_BEHAVIORAL_COMPONENTS)
    )
    if arr.size == 0:
        return {
            "math_n_terms_active": n_active,
            "math_sparsity": 0.0,
            "math_variance": 0.0,
            "math_magnitude_mean": 0.0,
            "math_magnitude_l1": 0.0,
            "math_pos_neg_ratio": 0.0,
            "math_pos_mass": 0.0,
            "math_neg_mass": 0.0,
            "math_gradient_magnitude_mean": 0.0,
            "math_smoothness_mean_abs_delta": 0.0,
            "math_distance_sensitivity_mean": 0.0,
        }

    pos = arr[arr > 0]
    neg = arr[arr < 0]
    pos_mass = float(np.sum(pos)) if pos.size else 0.0
    neg_mass = float(np.sum(np.abs(neg))) if neg.size else 0.0
    return {
        "math_n_terms_active": n_active,
        "math_sparsity": float(np.mean(np.abs(arr) < 1e-8)),
        "math_variance": float(np.var(arr)),
        "math_magnitude_mean": float(np.mean(np.abs(arr))),
        "math_magnitude_l1": float(np.mean(np.abs(arr))),
        "math_magnitude_max": float(np.max(np.abs(arr))),
        "math_pos_mass": pos_mass,
        "math_neg_mass": neg_mass,
        "math_pos_neg_ratio": _finite_or_zero(pos_mass / (neg_mass + EPS)),
        "math_fraction_positive": float(np.mean(arr > 0)),
        "math_fraction_negative": float(np.mean(arr < 0)),
        "math_gradient_magnitude_mean": float(np.mean(grad_mags)) if grad_mags else 0.0,
        "math_gradient_magnitude_std": float(np.std(grad_mags)) if grad_mags else 0.0,
        "math_smoothness_mean_abs_delta": float(np.mean(step_deltas)) if step_deltas else 0.0,
        "math_smoothness_std_abs_delta": float(np.std(step_deltas)) if step_deltas else 0.0,
        "math_distance_sensitivity_mean": float(np.mean(dist_sens)) if dist_sens else 0.0,
        "math_distance_sensitivity_std": float(np.std(dist_sens)) if dist_sens else 0.0,
    }


def extract_behavioral_from_scorer(
    score_fn: ScoreFn,
    episodes: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    *,
    weights: Mapping[str, float] | None = None,
    scales: Mapping[str, float] | None = None,
    component_names: Sequence[str] = GATE1_BEHAVIORAL_COMPONENTS,
) -> dict[str, float]:
    """Behavioral probe features (Gate-1 column set by default)."""
    names = tuple(component_names)
    wsrc = {n: float((weights or {}).get(n, 0.0)) for n in names}
    ssrc = {n: float((scales or {}).get(n, 1.0)) for n in names}

    all_totals: list[float] = []
    all_comps: dict[str, list[float]] = {name: [] for name in names}
    per_policy_totals: dict[str, list[float]] = {}

    for policy_name, transitions in episodes:
        ep_totals: list[float] = []
        for transition in transitions:
            total, comps = score_fn(transition)
            ep_totals.append(float(total))
            all_totals.append(float(total))
            if comps is None:
                for name in names:
                    all_comps[name].append(0.0)
            else:
                for name in names:
                    all_comps[name].append(float(comps.get(name, 0.0)))
        per_policy_totals.setdefault(policy_name, []).extend(ep_totals)

    features: dict[str, float] = {}
    totals_arr = np.asarray(all_totals, dtype=np.float64)
    features.update(_summarize_series(totals_arr, "total"))

    means: dict[str, float] = {}
    for name in names:
        arr = np.asarray(all_comps[name], dtype=np.float64)
        features.update(_summarize_series(arr, f"comp_{name}"))
        means[name] = float(np.mean(arr)) if arr.size else 0.0

    abs_means = {k: abs(v) for k, v in means.items()}
    denom = sum(abs_means.values()) + 1e-12
    for name in names:
        features[f"contrib_ratio_{name}"] = abs_means[name] / denom
        features[f"weighted_mean_{name}"] = float(wsrc[name] * ssrc[name] * means[name])

    if totals_arr.size >= 2:
        mat = np.column_stack([all_comps[name] for name in names])
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
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

    return features


def extract_mb_features(
    score_fn: ScoreFn,
    episodes: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    *,
    weights: Mapping[str, float] | None = None,
    scales: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Combined Gate-1 math+behavioral feature dict."""
    ep_only = [trs for _, trs in episodes]
    feats = extract_math_from_scorer(score_fn, ep_only, weights=weights)
    feats.update(
        extract_behavioral_from_scorer(
            score_fn, episodes, weights=weights, scales=scales
        )
    )
    return feats


def group_transitions_into_episodes(
    transitions: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """Group flat probe transitions into (policy, transitions) episodes."""
    if not transitions:
        return []
    # Prefer explicit episode_index if present; else split on policy changes.
    if any("episode_index" in t for t in transitions):
        buckets: dict[tuple[Any, str], list[Mapping[str, Any]]] = {}
        order: list[tuple[Any, str]] = []
        for t in transitions:
            key = (t.get("episode_index"), str(t.get("policy", "unknown")))
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(t)
        return [(k[1], buckets[k]) for k in order]

    episodes: list[tuple[str, list[Mapping[str, Any]]]] = []
    cur_policy = str(transitions[0].get("policy", "unknown"))
    cur: list[Mapping[str, Any]] = []
    for t in transitions:
        p = str(t.get("policy", "unknown"))
        if p != cur_policy and cur:
            episodes.append((cur_policy, cur))
            cur = []
            cur_policy = p
        cur.append(t)
    if cur:
        episodes.append((cur_policy, cur))
    return episodes
