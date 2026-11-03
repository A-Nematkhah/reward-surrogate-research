"""Surrogate evaluation metrics, ranking agreement, and uncertainty checks."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _is_constant(values: np.ndarray, *, atol: float = 1e-12) -> bool:
    if values.size == 0:
        return True
    return bool(np.nanmax(values) - np.nanmin(values) <= atol)


def association_status(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Return why rank correlations may be undefined.

    - ok
    - too_few
    - constant_prediction  (e.g. mean/median baselines within a fold)
    - constant_label
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return "too_few"
    yt = y_true[mask]
    yp = y_pred[mask]
    if _is_constant(yp):
        return "constant_prediction"
    if _is_constant(yt):
        return "constant_label"
    return "ok"


def _safe_rank_corr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, str]:
    status = association_status(y_true, y_pred)
    if status != "ok":
        return float("nan"), float("nan"), status
    sp = spearmanr(y_true, y_pred).correlation
    kd = kendalltau(y_true, y_pred).correlation
    sp_f = float(sp) if sp is not None and np.isfinite(sp) else float("nan")
    kd_f = float(kd) if kd is not None and np.isfinite(kd) else float("nan")
    if not np.isfinite(sp_f) or not np.isfinite(kd_f):
        return sp_f, kd_f, "undefined"
    return sp_f, kd_f, "ok"


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
            "kendall": float("nan"),
            "ranking_status": "too_few",
            "n": float(mask.sum()),
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    sp, kd, status = _safe_rank_corr(yt, yp)
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "spearman": sp,
        "kendall": kd,
        "ranking_status": status,
        "n": float(mask.sum()),
    }


def ranking_agreement(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> dict[str, float | str]:
    """
    Compare predicted vs actual rankings.

    For lower-is-better targets, both series are negated so that higher rank =
    better quality. Constant predictions yield NaN with ranking_status set.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if yt.size < 2:
        return {
            "rank_spearman": float("nan"),
            "rank_kendall": float("nan"),
            "rank_status": "too_few",
        }
    if not higher_is_better:
        yt = -yt
        yp = -yp
    sp, kd, status = _safe_rank_corr(yt, yp)
    return {
        "rank_spearman": sp,
        "rank_kendall": kd,
        "rank_status": status,
    }


def _orient_for_ranking(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    higher_is_better: bool,
) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if higher_is_better:
        return yt, yp
    return -yt, -yp


def top_k_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    k: int,
    higher_is_better: bool = True,
) -> float:
    """
    Fraction of true top-k items that appear in predicted top-k (set overlap / k).

    Both series are oriented so that larger values are better before ranking.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = _orient_for_ranking(y_true[mask], y_pred[mask], higher_is_better=higher_is_better)
    n = int(yt.size)
    if n == 0:
        return float("nan")
    kk = min(int(k), n)
    true_top = set(np.argsort(-yt)[:kk].tolist())
    pred_top = set(np.argsort(-yp)[:kk].tolist())
    return float(len(true_top & pred_top) / kk)


def pairwise_ranking_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> float:
    """
    Fraction of comparable pairs whose predicted order matches the true order.

    Ties in y_true are skipped. Ties in y_pred count as incorrect.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = _orient_for_ranking(y_true[mask], y_pred[mask], higher_is_better=higher_is_better)
    n = int(yt.size)
    if n < 2:
        return float("nan")
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if yt[i] == yt[j]:
                continue
            total += 1
            true_order = yt[i] > yt[j]
            pred_order = yp[i] > yp[j]
            if true_order == pred_order and yp[i] != yp[j]:
                correct += 1
    if total == 0:
        return float("nan")
    return float(correct / total)


def selection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    higher_is_better: bool = True,
    top_k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    out: dict[str, float] = {
        "pairwise_accuracy": pairwise_ranking_accuracy(
            y_true, y_pred, higher_is_better=higher_is_better
        )
    }
    for k in top_k_values:
        out[f"top_{k}_accuracy"] = top_k_accuracy(
            y_true, y_pred, k=k, higher_is_better=higher_is_better
        )
    return out


def uncertainty_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None,
) -> dict[str, float]:
    """
    Relate predictive uncertainty to absolute error.

    Does **not** claim calibration; reports association statistics only.
    """
    if y_std is None:
        return {
            "uncertainty_error_spearman": float("nan"),
            "mean_abs_error": float("nan"),
            "mean_pred_std": float("nan"),
        }
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_std)
    if mask.sum() < 2:
        return {
            "uncertainty_error_spearman": float("nan"),
            "mean_abs_error": float("nan"),
            "mean_pred_std": float("nan"),
        }
    abs_err = np.abs(y_true[mask] - y_pred[mask])
    std_m = y_std[mask]
    if _is_constant(std_m) or _is_constant(abs_err):
        sp_f = float("nan")
    else:
        sp = spearmanr(std_m, abs_err).correlation
        sp_f = float(sp) if sp is not None and np.isfinite(sp) else float("nan")
    return {
        "uncertainty_error_spearman": sp_f,
        "mean_abs_error": float(np.mean(abs_err)),
        "mean_pred_std": float(np.mean(std_m)),
    }


HIGHER_IS_BETTER = {
    "success_rate": True,
    "collision_rate": False,
    "timeout_rate": False,
    "mean_goal_time": False,
    "mean_path_length": False,
}

# Models that emit fold-constant predictions; ranking vs labels is N/A.
CONSTANT_RANKING_MODELS = frozenset({"mean", "median"})


DEFAULT_TOP_K = (1, 3, 5)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_name: str,
    y_std: np.ndarray | None = None,
    model_name: str | None = None,
    top_k_values: tuple[int, ...] = DEFAULT_TOP_K,
) -> dict[str, float | str]:
    hib = HIGHER_IS_BETTER.get(target_name, True)
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(ranking_agreement(y_true, y_pred, higher_is_better=hib))
    metrics.update(uncertainty_diagnostics(y_true, y_pred, y_std))
    metrics.update(
        selection_metrics(
            y_true, y_pred, higher_is_better=hib, top_k_values=top_k_values
        )
    )

    # Force ranking N/A for known constant predictors even if pooled OOF
    # accidentally varies across folds (artifact under grouped CV).
    if model_name in CONSTANT_RANKING_MODELS:
        metrics["spearman"] = float("nan")
        metrics["kendall"] = float("nan")
        metrics["rank_spearman"] = float("nan")
        metrics["rank_kendall"] = float("nan")
        metrics["ranking_status"] = "constant_predictor_model"
        metrics["rank_status"] = "constant_predictor_model"
        metrics["pairwise_accuracy"] = float("nan")
        for k in top_k_values:
            metrics[f"top_{k}_accuracy"] = float("nan")
    return metrics


def metrics_to_row(
    metrics: Mapping[str, Any],
    *,
    model: str,
    feature_set: str,
    target: str,
    split: str,
    fold: int | str | None = None,
) -> dict[str, Any]:
    row = {
        "model": model,
        "feature_set": feature_set,
        "target": target,
        "split": split,
        "fold": fold,
    }
    row.update(dict(metrics))
    return row


def _summarize_fold_metric(
    fold_metric_rows: list[Mapping[str, Any]],
    key: str,
    *,
    require_ranking_ok: bool = False,
) -> dict[str, float]:
    values = []
    for row in fold_metric_rows:
        if require_ranking_ok and str(row.get("ranking_status", "ok")) != "ok":
            continue
        val = row.get(key)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fval):
            values.append(fval)
    arr = np.asarray(values, dtype=np.float64)
    prefix = f"fold_{key}"
    if arr.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"n_folds_defined_{key}": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan"),
        f"n_folds_defined_{key}": float(arr.size),
    }


def summarize_fold_rankings(fold_metric_rows: list[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate per-fold ranking/selection metrics into mean/std."""
    out: dict[str, float] = {}
    out.update(
        _summarize_fold_metric(fold_metric_rows, "spearman", require_ranking_ok=True)
    )
    out.update(_summarize_fold_metric(fold_metric_rows, "pairwise_accuracy"))
    out.update(_summarize_fold_metric(fold_metric_rows, "top_1_accuracy"))
    out.update(_summarize_fold_metric(fold_metric_rows, "top_3_accuracy"))
    out.update(_summarize_fold_metric(fold_metric_rows, "top_5_accuracy"))
    return out
