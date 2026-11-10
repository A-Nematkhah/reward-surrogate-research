"""Generation-level triage metrics (single source of truth for Phase 1+)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


def _as_float_array(x: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _is_constant(values: np.ndarray, *, atol: float = 1e-12) -> bool:
    if values.size == 0:
        return True
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return bool(np.nanmax(finite) - np.nanmin(finite) <= atol)


def break_ties(
    scores: np.ndarray,
    *,
    rng: np.random.Generator,
    higher_is_better: bool = True,
) -> np.ndarray:
    """Return argsort indices with seeded random tie-breaking."""
    scores = _as_float_array(scores)
    n = scores.size
    noise = rng.random(n) * 1e-12
    keyed = scores + noise if higher_is_better else -(scores + noise)
    # Stable argsort on (-score) for higher-is-better with tiny noise already applied.
    if higher_is_better:
        return np.argsort(-keyed, kind="mergesort")
    return np.argsort(keyed, kind="mergesort")


def select_topk_indices(
    scores: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    higher_is_better: bool = True,
) -> np.ndarray:
    """Keep top-k by score; if k >= n keep everyone."""
    scores = _as_float_array(scores)
    n = int(scores.size)
    if n == 0:
        return np.asarray([], dtype=int)
    kk = int(k)
    if kk >= n:
        return np.arange(n, dtype=int)
    if kk <= 0:
        return np.asarray([], dtype=int)
    order = break_ties(scores, rng=rng, higher_is_better=higher_is_better)
    return order[:kk]


def recall_at_k(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    k: int,
    *,
    rng: np.random.Generator | None = None,
    higher_is_better: bool = True,
) -> float:
    """1.0 if the true best pool member is among the kept top-k by score."""
    y_true = _as_float_array(y_true)
    y_score = _as_float_array(y_score)
    if y_true.size == 0:
        return float("nan")
    rng = rng or np.random.default_rng(0)
    kept = select_topk_indices(y_score, k, rng=rng, higher_is_better=higher_is_better)
    true_best = break_ties(y_true, rng=rng, higher_is_better=higher_is_better)[0]
    return float(true_best in set(kept.tolist()))


def regret_at_k(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    k: int,
    *,
    rng: np.random.Generator | None = None,
    higher_is_better: bool = True,
) -> float:
    """max(true) - max(true of kept). Lower is better."""
    y_true = _as_float_array(y_true)
    y_score = _as_float_array(y_score)
    if y_true.size == 0:
        return float("nan")
    rng = rng or np.random.default_rng(0)
    kept = select_topk_indices(y_score, k, rng=rng, higher_is_better=higher_is_better)
    if kept.size == 0:
        return float("nan")
    if higher_is_better:
        return float(np.max(y_true) - np.max(y_true[kept]))
    return float(np.min(y_true[kept]) - np.min(y_true))


def top1_hit(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    higher_is_better: bool = True,
) -> float:
    return recall_at_k(y_true, y_score, 1, rng=rng, higher_is_better=higher_is_better)


def pairwise_accuracy(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    se: Sequence[float] | np.ndarray | None = None,
    higher_is_better: bool = True,
) -> float:
    """
    Fraction of pairs where score order matches label order.

    Optional weights |delta|/SE (SE of the label difference); unweighted if se is None.
    Ties in labels are skipped.
    """
    y_true = _as_float_array(y_true)
    y_score = _as_float_array(y_score)
    n = y_true.size
    if n < 2:
        return float("nan")
    se_arr = None if se is None else _as_float_array(se)
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dy = y_true[i] - y_true[j]
            if abs(dy) <= 1e-15:
                continue
            ds = y_score[i] - y_score[j]
            agree = (dy > 0 and ds > 0) or (dy < 0 and ds < 0)
            if not higher_is_better:
                agree = (dy > 0 and ds < 0) or (dy < 0 and ds > 0)
            if se_arr is None:
                w = 1.0
            else:
                # Approximate SE of difference for independent seeds: hypot(se_i, se_j)
                sej = float(np.hypot(se_arr[i], se_arr[j]))
                w = abs(dy) / sej if sej > 1e-12 else abs(dy)
            den += w
            if agree:
                num += w
    if den <= 0:
        return float("nan")
    return float(num / den)


def spearman_safe(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    min_n: int = 4,
) -> float:
    """Spearman only if n >= min_n and neither side is constant; else NaN."""
    y_true = _as_float_array(y_true)
    y_score = _as_float_array(y_score)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    yt = y_true[mask]
    ys = y_score[mask]
    if yt.size < min_n or _is_constant(yt) or _is_constant(ys):
        return float("nan")
    corr = spearmanr(yt, ys).correlation
    return float(corr) if corr is not None and np.isfinite(corr) else float("nan")


def pool_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    k: int,
    *,
    rng: np.random.Generator | None = None,
    se: Sequence[float] | np.ndarray | None = None,
    higher_is_better: bool = True,
) -> dict[str, float]:
    """Bundle of triage metrics for one pool."""
    rng = rng or np.random.default_rng(0)
    return {
        "n": float(len(y_true)),
        "k": float(k),
        "recall_at_k": recall_at_k(
            y_true, y_score, k, rng=rng, higher_is_better=higher_is_better
        ),
        "regret_at_k": regret_at_k(
            y_true, y_score, k, rng=rng, higher_is_better=higher_is_better
        ),
        "top1_hit": top1_hit(
            y_true, y_score, rng=rng, higher_is_better=higher_is_better
        ),
        "pairwise_accuracy": pairwise_accuracy(
            y_true, y_score, se=se, higher_is_better=higher_is_better
        ),
        "spearman": spearman_safe(y_true, y_score),
    }


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    statistic: str = "mean",
) -> dict[str, float]:
    """Group/pool-level bootstrap CI over a 1-D array of per-pool metrics."""
    values = _as_float_array(values)
    values = values[np.isfinite(values)]
    rng = rng or np.random.default_rng(0)
    if values.size == 0:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0.0}
    if statistic == "median":
        est = float(np.median(values))
        boots = [
            float(np.median(rng.choice(values, size=values.size, replace=True)))
            for _ in range(n_boot)
        ]
    else:
        est = float(np.mean(values))
        boots = [
            float(np.mean(rng.choice(values, size=values.size, replace=True)))
            for _ in range(n_boot)
        ]
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimate": est,
        "lo": float(lo),
        "hi": float(hi),
        "n": float(values.size),
    }


def summarize_pools(
    per_pool: Sequence[Mapping[str, float]],
    *,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Aggregate per-pool metric dicts with mean/median bootstrap CIs."""
    rng = rng or np.random.default_rng(0)
    if not per_pool:
        return {}
    keys = [
        "recall_at_k",
        "regret_at_k",
        "top1_hit",
        "pairwise_accuracy",
        "spearman",
    ]
    out: dict[str, Any] = {"n_pools": float(len(per_pool))}
    for key in keys:
        arr = np.asarray([float(p[key]) for p in per_pool], dtype=np.float64)
        out[f"{key}_mean"] = bootstrap_ci(arr, n_boot=n_boot, rng=rng, statistic="mean")
        out[f"{key}_median"] = bootstrap_ci(
            arr, n_boot=n_boot, rng=rng, statistic="median"
        )
    return out
