"""Mathematical reward-landscape features from scoring fixed probe trajectories."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from rsr.features.probe_trajectories import ProbeEpisode
from rsr.rewards.reward_function import RewardSpec, WeightedRewardFunction

EPS = 1e-8
DIST_DELTA = 1e-2
ACTION_DELTA = 1e-2


def _score_transition(reward_fn: WeightedRewardFunction, transition: Mapping[str, Any]) -> float:
    return float(reward_fn(None, transition))


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def extract_math_features(
    spec: RewardSpec,
    episodes: Sequence[ProbeEpisode],
) -> dict[str, float]:
    """
    Score fixed probes with ``spec`` and summarize landscape geometry.

    Does not use PPO outcomes. Features are prefixed with ``math_``.
    """
    reward_fn = WeightedRewardFunction.from_spec(spec)
    totals: list[float] = []
    step_deltas: list[float] = []
    grad_mags: list[float] = []
    dist_sens: list[float] = []

    for episode in episodes:
        prev_total: float | None = None
        for transition in episode.transitions:
            total = _score_transition(reward_fn, transition)
            totals.append(total)
            if prev_total is not None:
                step_deltas.append(abs(total - prev_total))
            prev_total = total

            # Distance sensitivity via finite difference on goal_distance
            t_plus = dict(transition)
            t_minus = dict(transition)
            t_plus["goal_distance"] = float(transition["goal_distance"]) + DIST_DELTA
            t_minus["goal_distance"] = float(transition["goal_distance"]) - DIST_DELTA
            r_plus = _score_transition(reward_fn, t_plus)
            r_minus = _score_transition(reward_fn, t_minus)
            dist_sens.append(abs(r_plus - r_minus) / (2.0 * DIST_DELTA))

            # Action gradient magnitude (2D action)
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
                g = (_score_transition(reward_fn, tp) - _score_transition(reward_fn, tm)) / (
                    2.0 * ACTION_DELTA
                )
                g2 += float(g) ** 2
            grad_mags.append(float(np.sqrt(g2)))

    arr = np.asarray(totals, dtype=np.float64)
    if arr.size == 0:
        return {
            "math_n_terms_active": float(
                sum(not np.isclose(spec.weights.get(n, 0.0), 0.0) for n in spec.weights)
            ),
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
    features = {
        "math_n_terms_active": float(
            sum(not np.isclose(float(w), 0.0) for w in spec.weights.values())
        ),
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
    return features
