"""Black-box reward features (no code weights, no early PPO)."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    if method == "pearson":
        r, _ = stats.pearsonr(x, y)
    else:
        r, _ = stats.spearmanr(x, y)
    if r is None or not np.isfinite(r):
        return 0.0
    return float(r)


def extract_bb_stats(rewards: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(rewards, dtype=np.float64)
    if arr.size == 0:
        return {
            "bb_stats_mean": 0.0,
            "bb_stats_std": 0.0,
            "bb_stats_q10": 0.0,
            "bb_stats_q50": 0.0,
            "bb_stats_q90": 0.0,
            "bb_stats_frac_zero": 0.0,
            "bb_stats_frac_saturated": 0.0,
            "bb_stats_scale": 0.0,
        }
    abs_max = float(np.max(np.abs(arr))) + 1e-12
    saturated = np.abs(arr) >= 0.99 * abs_max
    return {
        "bb_stats_mean": float(np.mean(arr)),
        "bb_stats_std": float(np.std(arr)),
        "bb_stats_q10": float(np.quantile(arr, 0.10)),
        "bb_stats_q50": float(np.quantile(arr, 0.50)),
        "bb_stats_q90": float(np.quantile(arr, 0.90)),
        "bb_stats_frac_zero": float(np.mean(np.abs(arr) < 1e-12)),
        "bb_stats_frac_saturated": float(np.mean(saturated)),
        "bb_stats_scale": float(abs_max),
    }


def extract_bb_corr(
    rewards: Sequence[float],
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    r = np.asarray(rewards, dtype=np.float64)
    progress = np.asarray([float(t.get("progress", 0.0)) for t in transitions], dtype=np.float64)
    collision = np.asarray(
        [float(bool(t.get("collision", False))) for t in transitions], dtype=np.float64
    )
    clearance = np.asarray([float(t.get("clearance", 0.0)) for t in transitions], dtype=np.float64)
    success = np.asarray(
        [float(bool(t.get("success", False))) for t in transitions], dtype=np.float64
    )
    out: dict[str, float] = {}
    for name, y in (
        ("progress", progress),
        ("collision", collision),
        ("clearance", clearance),
        ("success", success),
    ):
        out[f"bb_corr_pearson_{name}"] = _safe_corr(r, y, "pearson")
        out[f"bb_corr_spearman_{name}"] = _safe_corr(r, y, "spearman")
    return out


def extract_bb_policy(
    rewards: Sequence[float],
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    by_policy: dict[str, list[float]] = {}
    for r, t in zip(rewards, transitions, strict=True):
        p = str(t.get("policy", "unknown"))
        by_policy.setdefault(p, []).append(float(r))
    totals = {p: float(np.sum(v)) for p, v in by_policy.items()}
    # Fixed policy order for stable features.
    order = [
        "random",
        "goal_seeking",
        "obstacle_avoiding",
        "partial_expert",
        "edge_states",
    ]
    out: dict[str, float] = {}
    for p in order:
        out[f"bb_policy_return_{p}"] = float(totals.get(p, 0.0))
    ranked = sorted(order, key=lambda p: totals.get(p, 0.0), reverse=True)
    for i, p in enumerate(ranked):
        out[f"bb_policy_rank_{p}"] = float(i)
    return out


def extract_bb_shape(
    rewards: Sequence[float],
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    r = np.asarray(rewards, dtype=np.float64)
    done = np.asarray([bool(t.get("done", False)) for t in transitions])
    per_step = float(np.mean(np.abs(r))) if r.size else 0.0
    terminal = float(np.mean(np.abs(r[done]))) if np.any(done) else 0.0

    def _discounted(gamma: float) -> float:
        if r.size == 0:
            return 0.0
        # Approximate return sensitivity: sum gamma^t * r_t over the batch order.
        discs = np.array([gamma**t for t in range(r.size)], dtype=np.float64)
        return float(np.dot(discs, r))

    # Action sensitivity via finite differences on action_delta magnitude vs |r|.
    action_delta = np.asarray(
        [float(np.linalg.norm(np.asarray(t.get("action_delta", [0.0])))) for t in transitions],
        dtype=np.float64,
    )
    action_sens = _safe_corr(np.abs(r), action_delta, "pearson") if r.size else 0.0

    # Potential-shaping consistency: corr(r, progress) should dominate if shaped.
    progress = np.asarray([float(t.get("progress", 0.0)) for t in transitions], dtype=np.float64)
    shaping = _safe_corr(r, progress, "pearson")

    return {
        "bb_shape_terminal_mag": terminal,
        "bb_shape_per_step_mag": per_step,
        "bb_shape_return_g09": _discounted(0.9),
        "bb_shape_return_g099": _discounted(0.99),
        "bb_shape_action_sensitivity": action_sens,
        "bb_shape_potential_consistency": shaping,
    }


def extract_bb_comp(
    components: Sequence[Mapping[str, float] | None],
) -> dict[str, float]:
    keys: set[str] = set()
    for c in components:
        if c:
            keys.update(c.keys())
    out: dict[str, float] = {}
    if not keys:
        return {
            "bb_comp_n_keys": 0.0,
            "bb_comp_any": 0.0,
        }
    out["bb_comp_n_keys"] = float(len(keys))
    out["bb_comp_any"] = 1.0
    for key in sorted(keys):
        vals = np.asarray(
            [float((c or {}).get(key, 0.0)) for c in components],
            dtype=np.float64,
        )
        out[f"bb_comp_mean_{key}"] = float(np.mean(vals))
        out[f"bb_comp_std_{key}"] = float(np.std(vals))
    return out


def extract_code_features(code: str) -> dict[str, float]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "code_ast_nodes": 0.0,
            "code_n_terms": 0.0,
            "code_n_constants": 0.0,
        }
    n_nodes = sum(1 for _ in ast.walk(tree))
    n_binops = sum(1 for n in ast.walk(tree) if isinstance(n, ast.BinOp))
    n_const = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant))
    return {
        "code_ast_nodes": float(n_nodes),
        "code_n_terms": float(n_binops + 1),
        "code_n_constants": float(n_const),
    }


def extract_blackbox_features(
    rewards: Sequence[float],
    transitions: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, float] | None] | None = None,
    *,
    code: str | None = None,
    groups: Sequence[str] | None = None,
    score_fn: Any | None = None,
    weights: Mapping[str, float] | None = None,
    scales: Mapping[str, float] | None = None,
    episodes: Sequence[tuple[str, Sequence[Mapping[str, Any]]]] | None = None,
) -> dict[str, float]:
    """Extract selected feature groups.

    Default group is ``mb`` (Gate-1 math+behavioral). Optional: bb_*, code.
    """
    from reward_triage.mb_features import extract_mb_features, group_transitions_into_episodes

    wanted = set(groups or ("mb",))
    feats: dict[str, float] = {}
    if "mb" in wanted:
        if score_fn is None:
            raise ValueError("score_fn is required for feature group 'mb'")
        eps = list(episodes) if episodes is not None else group_transitions_into_episodes(transitions)
        feats.update(
            extract_mb_features(score_fn, eps, weights=weights, scales=scales)
        )
    if "bb_stats" in wanted:
        feats.update(extract_bb_stats(rewards))
    if "bb_corr" in wanted:
        feats.update(extract_bb_corr(rewards, transitions))
    if "bb_policy" in wanted:
        feats.update(extract_bb_policy(rewards, transitions))
    if "bb_shape" in wanted:
        feats.update(extract_bb_shape(rewards, transitions))
    if "bb_comp" in wanted:
        feats.update(extract_bb_comp(components or [None] * len(rewards)))
    if "code" in wanted and code is not None:
        feats.update(extract_code_features(code))
    return feats


def feature_hash(features: Mapping[str, float]) -> str:
    payload = ",".join(f"{k}={features[k]:.8g}" for k in sorted(features))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
