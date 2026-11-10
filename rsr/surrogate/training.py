"""Train and cross-validate surrogate models with grouped family splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    ParameterGrid,
    train_test_split,
)

from src.surrogate.baselines import MeanPredictor, MedianPredictor, make_ridge_baseline
from src.surrogate.evaluation import (
    evaluate_predictions,
    metrics_to_row,
    summarize_fold_rankings,
)
from src.surrogate.models import (
    make_ensemble_trees,
    make_extra_trees,
    make_gaussian_process,
    make_hist_gradient_boosting,
    make_random_forest,
    make_rank_extra_trees,
    predict_with_uncertainty,
    select_feature_indices,
    tree_feature_importances,
)
from src.utils.config import relpath_to_project
from src.utils.logging import get_logger

logger = get_logger("surrogate.training")


def load_feature_bundle(features_dir: str | Path) -> dict[str, pd.DataFrame]:
    features_dir = Path(features_dir)
    required = ("meta", "static", "behavioral", "hybrid", "targets")
    bundle: dict[str, pd.DataFrame] = {
        name: pd.read_csv(features_dir / f"features_{name}.csv") for name in required
    }
    for optional in ("early", "math", "interaction", "probe", "behavioral_early"):
        path = features_dir / f"features_{optional}.csv"
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.shape[1] > 0:
            bundle[optional] = frame
    return bundle


def _finite_mask(y: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(y, dtype=np.float64))


def _resolve_feature_sets(config: Mapping[str, Any]) -> list[str]:
    primary = list(config.get("feature_sets", ["static", "behavioral", "hybrid"]))
    secondary: list[str] = []
    if bool(config.get("include_secondary_feature_sets", False)):
        secondary = list(config.get("secondary_feature_sets", []))
    seen: set[str] = set()
    out: list[str] = []
    for name in primary + secondary:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _feature_selection_cfg(config: Mapping[str, Any]) -> tuple[bool, int, int]:
    cfg = config.get("feature_selection", {}) or {}
    enabled = bool(cfg.get("enabled", False))
    max_features = int(cfg.get("max_features", 40))
    min_to_prune = int(cfg.get("min_features_to_prune", 25))
    return enabled, max_features, min_to_prune


def _maybe_select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: Mapping[str, Any],
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply train-fold-only pruning; returns X_tr, X_te, selected indices or None."""
    enabled, max_features, min_to_prune = _feature_selection_cfg(config)
    if (not enabled) or X_train.shape[1] < min_to_prune:
        return X_train, X_test, None
    idx = select_feature_indices(
        X_train,
        y_train,
        max_features=max_features,
        random_state=random_state,
    )
    return X_train[:, idx], X_test[:, idx], idx


def _build_model(
    name: str,
    config: Mapping[str, Any],
    *,
    random_state: int,
    params: Mapping[str, Any] | None = None,
) -> Any:
    params = dict(params or {})
    hp = config.get("hyperparameters", {})
    scale = bool(config.get("preprocessing", {}).get("scale_features", True))

    if name == "mean":
        return MeanPredictor()
    if name == "median":
        return MedianPredictor()
    if name == "ridge":
        alphas = hp.get("ridge", {}).get("alphas", [0.1, 1.0, 10.0, 100.0])
        return make_ridge_baseline(alphas=list(alphas), scale_features=scale)
    if name == "random_forest":
        cfg = hp.get("random_forest", {})
        return make_random_forest(
            n_estimators=int(params.get("n_estimators", cfg.get("n_estimators", 100))),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_state,
        )
    if name == "extra_trees":
        cfg = hp.get("extra_trees", {})
        return make_extra_trees(
            n_estimators=int(params.get("n_estimators", cfg.get("n_estimators", 100))),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_state,
        )
    if name in {"hist_gradient_boosting", "gradient_boosting"}:
        cfg = hp.get("hist_gradient_boosting", hp.get("gradient_boosting", {}))
        lr_default = cfg.get("learning_rate", 0.1)
        if isinstance(lr_default, list):
            lr_default = lr_default[0]
        leaf_default = cfg.get("min_samples_leaf", 5)
        if isinstance(leaf_default, list):
            leaf_default = leaf_default[0]
        return make_hist_gradient_boosting(
            max_depth=params.get("max_depth", None),
            learning_rate=float(params.get("learning_rate", lr_default)),
            max_iter=int(params.get("max_iter", cfg.get("max_iter", 100))),
            min_samples_leaf=int(params.get("min_samples_leaf", leaf_default)),
            random_state=random_state,
        )
    if name == "gaussian_process":
        cfg = hp.get("gaussian_process", {})
        return make_gaussian_process(
            normalize_y=bool(cfg.get("normalize_y", True)),
            alpha=float(cfg.get("alpha", 1e-6)),
            random_state=random_state,
            scale_features=scale,
        )
    if name == "rank_extra_trees":
        cfg = hp.get("rank_extra_trees", hp.get("extra_trees", {}))
        return make_rank_extra_trees(
            n_estimators=int(params.get("n_estimators", cfg.get("n_estimators", 400))),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=random_state,
        )
    if name == "ensemble_trees":
        rf_cfg = hp.get("random_forest", {})
        et_cfg = hp.get("extra_trees", {})
        hgb_cfg = hp.get("hist_gradient_boosting", {})
        hgb_lr = hgb_cfg.get("learning_rate", 0.1)
        if isinstance(hgb_lr, list):
            hgb_lr = hgb_lr[0]
        hgb_leaf = hgb_cfg.get("min_samples_leaf", 10)
        if isinstance(hgb_leaf, list):
            hgb_leaf = hgb_leaf[0]
        return make_ensemble_trees(
            random_state=random_state,
            rf_params={
                "n_estimators": int(rf_cfg.get("n_estimators", 200)),
                "max_depth": None,
                "min_samples_leaf": 2,
            },
            et_params={
                "n_estimators": int(et_cfg.get("n_estimators", 250)),
                "max_depth": None,
                "min_samples_leaf": 2,
            },
            hgb_params={
                "max_depth": 5,
                "learning_rate": float(hgb_lr),
                "max_iter": int(hgb_cfg.get("max_iter", 150)),
                "min_samples_leaf": int(hgb_leaf),
            },
        )
    raise ValueError(f"Unknown model '{name}'")


def _param_grid_for(name: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    hp = config.get("hyperparameters", {})
    if name == "random_forest":
        cfg = hp.get("random_forest", {})
        grid = {
            "max_depth": cfg.get("max_depth", [None]),
            "min_samples_leaf": cfg.get("min_samples_leaf", [1]),
        }
        return list(ParameterGrid(grid))
    if name in {"extra_trees", "rank_extra_trees"}:
        cfg = hp.get(name if name in hp else "extra_trees", hp.get("extra_trees", {}))
        grid = {
            "max_depth": cfg.get("max_depth", [None]),
            "min_samples_leaf": cfg.get("min_samples_leaf", [1]),
        }
        return list(ParameterGrid(grid))
    if name in {"hist_gradient_boosting", "gradient_boosting"}:
        cfg = hp.get("hist_gradient_boosting", hp.get("gradient_boosting", {}))
        grid = {
            "max_depth": cfg.get("max_depth", [None, 3]),
            "learning_rate": cfg.get("learning_rate", [0.05, 0.1]),
            "min_samples_leaf": cfg.get("min_samples_leaf", [5, 10]),
        }
        return list(ParameterGrid(grid))
    return [{}]


def _select_params_nested(
    model_name: str,
    config: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    *,
    random_state: int,
) -> dict[str, Any]:
    """
    Inner GroupKFold hyperparameter search on the outer training families only.

    Outer LOFO/GroupKFold test families never enter this selection.
    """
    grid = _param_grid_for(model_name, config)
    if len(grid) <= 1:
        return grid[0] if grid else {}

    unique_groups = np.unique(groups_train)
    max_inner = int(config.get("validation", {}).get("inner_n_splits", 3))
    n_splits = min(max_inner, len(unique_groups))
    if n_splits < 2:
        return grid[0]

    gkf = GroupKFold(n_splits=n_splits)
    best_params = grid[0]
    best_score = float("inf")
    rank_objective = model_name == "rank_extra_trees"
    for params in grid:
        fold_scores: list[float] = []
        for tr, va in gkf.split(X_train, y_train, groups_train):
            # Feature pruning already applied by the outer fold caller.
            model = _build_model(
                model_name, config, random_state=random_state, params=params
            )
            model.fit(X_train[tr], y_train[tr])
            pred, _ = predict_with_uncertainty(model, X_train[va])
            if rank_objective:
                # Higher Spearman is better → minimize negative Spearman
                if len(pred) >= 2 and np.std(pred) > 0 and np.std(y_train[va]) > 0:
                    corr = float(pd.Series(y_train[va]).corr(pd.Series(pred), method="spearman"))
                    fold_scores.append(-corr if np.isfinite(corr) else 0.0)
                else:
                    fold_scores.append(0.0)
            else:
                fold_scores.append(float(np.mean(np.abs(y_train[va] - pred))))
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_params = params
    return best_params


# Backward-compatible alias
_select_params_grouped = _select_params_nested


def _fit_predict_fold(
    model_name: str,
    config: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    groups_train: np.ndarray,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray | None, Any, dict[str, Any]]:
    X_tr, X_te, feat_idx = _maybe_select_features(
        X_train,
        y_train,
        X_test,
        config,
        random_state=random_state,
    )
    params = _select_params_nested(
        model_name,
        config,
        X_tr,
        y_train,
        groups_train,
        random_state=random_state,
    )
    model = _build_model(model_name, config, random_state=random_state, params=params)
    model.fit(X_tr, y_train)
    pred, std = predict_with_uncertainty(model, X_te)
    out_params = dict(params)
    if feat_idx is not None:
        out_params["n_selected_features"] = int(len(feat_idx))
    return pred, std, model, out_params


def run_grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    model_name: str,
    feature_set: str,
    target: str,
    config: Mapping[str, Any],
    meta: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    """Grouped K-fold CV by family. Returns metric rows, OOF predictions, importances."""
    y = np.asarray(y, dtype=np.float64)
    mask = _finite_mask(y)
    X = np.asarray(X, dtype=np.float64)[mask]
    y = y[mask]
    groups = np.asarray(groups)[mask]
    meta_m = meta.iloc[np.where(mask)[0]].reset_index(drop=True)

    n_families = len(np.unique(groups))
    if n_families < 2:
        raise ValueError("Need at least 2 families for grouped CV")

    val_cfg = config.get("validation", {})
    strategy = str(val_cfg.get("strategy", "grouped"))
    seed = int(config.get("random_seed", 123))

    if strategy in {"leave_one_family_out", "lofo"}:
        splitter = LeaveOneGroupOut()
        split_iter = list(splitter.split(X, y, groups))
        split_name = "lofo"
    else:
        n_splits = int(val_cfg.get("n_cv_folds", 4))
        n_splits = min(n_splits, n_families)
        if n_splits < 2:
            raise ValueError("Need at least 2 folds for grouped CV")
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = list(splitter.split(X, y, groups))
        split_name = "grouped_cv"

    metric_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []

    oof_pred = np.full(len(y), np.nan)
    oof_std = np.full(len(y), np.nan)

    for fold_i, (tr, te) in enumerate(split_iter):
        pred, std, model, params = _fit_predict_fold(
            model_name,
            config,
            X[tr],
            y[tr],
            X[te],
            groups[tr],
            random_state=seed + fold_i,
        )
        oof_pred[te] = pred
        if std is not None:
            oof_std[te] = std

        metrics = evaluate_predictions(
            y[te],
            pred,
            target_name=target,
            y_std=std,
            model_name=model_name,
        )
        row = metrics_to_row(
            metrics,
            model=model_name,
            feature_set=feature_set,
            target=target,
            split=split_name,
            fold=fold_i,
        )
        row["params"] = json.dumps(params)
        row["test_families"] = json.dumps(
            sorted({str(g) for g in groups[te].tolist()})
        )
        row["n_test"] = int(len(te))
        metric_rows.append(row)
        fold_metric_rows.append(row)

        for i_local, i_global in enumerate(te):
            pred_rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "target": target,
                    "split": split_name,
                    "fold": fold_i,
                    "candidate_id": meta_m.iloc[i_global]["candidate_id"],
                    "family_id": meta_m.iloc[i_global]["family_id"],
                    "y_true": float(y[i_global]),
                    "y_pred": float(pred[i_local]),
                    "y_std": None if std is None else float(std[i_local]),
                }
            )

        fi = tree_feature_importances(model)
        if fi is not None:
            importances.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "target": target,
                    "fold": fold_i,
                    "importances": fi.tolist(),
                }
            )

    # Aggregate OOF metrics + dual fold-summary statistics
    oof_std_arg = oof_std if np.any(np.isfinite(oof_std)) else None
    oof_metrics = evaluate_predictions(
        y,
        oof_pred,
        target_name=target,
        y_std=oof_std_arg,
        model_name=model_name,
    )
    oof_metrics.update(summarize_fold_rankings(fold_metric_rows))
    metric_rows.append(
        metrics_to_row(
            oof_metrics,
            model=model_name,
            feature_set=feature_set,
            target=target,
            split=f"{split_name}_oof",
            fold="all",
        )
    )
    return metric_rows, pd.DataFrame(pred_rows), importances


def run_random_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    model_name: str,
    feature_set: str,
    target: str,
    config: Mapping[str, Any],
    meta: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Secondary random holdout split (for comparison only)."""
    y = np.asarray(y, dtype=np.float64)
    mask = _finite_mask(y)
    X = np.asarray(X, dtype=np.float64)[mask]
    y = y[mask]
    groups = np.asarray(groups)[mask]
    meta_m = meta.iloc[np.where(mask)[0]].reset_index(drop=True)

    test_size = float(config.get("validation", {}).get("random_test_size", 0.25))
    seed = int(config.get("random_seed", 123))
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)

    pred, std, _, params = _fit_predict_fold(
        model_name,
        config,
        X[tr],
        y[tr],
        X[te],
        groups[tr],
        random_state=seed,
    )
    metrics = evaluate_predictions(
        y[te],
        pred,
        target_name=target,
        y_std=std,
        model_name=model_name,
    )
    row = metrics_to_row(
        metrics,
        model=model_name,
        feature_set=feature_set,
        target=target,
        split="random_holdout",
        fold=0,
    )
    row["params"] = json.dumps(params)

    pred_rows = []
    for i_local, i_global in enumerate(te):
        pred_rows.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "target": target,
                "split": "random_holdout",
                "fold": 0,
                "candidate_id": meta_m.iloc[i_global]["candidate_id"],
                "family_id": meta_m.iloc[i_global]["family_id"],
                "y_true": float(y[i_global]),
                "y_pred": float(pred[i_local]),
                "y_std": None if std is None else float(std[i_local]),
            }
        )
    return [row], pd.DataFrame(pred_rows)


def run_phase2_experiments(
    features_dir: str | Path,
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run all model × feature-set × target evaluations."""
    bundle = load_feature_bundle(features_dir)
    meta = bundle["meta"]
    targets_df = bundle["targets"]
    feature_sets = _resolve_feature_sets(config)
    targets = list(config.get("targets", list(targets_df.columns)))
    model_names = list(config.get("baselines", [])) + list(config.get("surrogates", []))

    output_dir = Path(
        output_dir or config.get("output", {}).get("results_dir", "experiments/phase2")
    )
    pred_dir = Path(
        config.get("output", {}).get("predictions_dir", output_dir / "predictions")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []
    all_preds: list[pd.DataFrame] = []
    all_importances: list[dict[str, Any]] = []

    groups = meta["family_id"].to_numpy()

    for feature_set in feature_sets:
        if feature_set not in bundle:
            logger.warning("Feature set '%s' missing from bundle; skipping", feature_set)
            continue
        X_df = bundle[feature_set]
        if X_df.shape[1] == 0:
            logger.warning("Feature set '%s' is empty; skipping", feature_set)
            continue
        X = X_df.to_numpy(dtype=np.float64)
        # Replace any residual non-finite features with 0 (should be rare)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        for target in targets:
            if target not in targets_df.columns:
                logger.warning("Target %s missing; skipping", target)
                continue
            y = targets_df[target].to_numpy(dtype=np.float64)
            logger.info("Running %s features × target=%s", feature_set, target)

            for model_name in model_names:
                try:
                    m_rows, pred_df, imps = run_grouped_cv(
                        X,
                        y,
                        groups,
                        model_name=model_name,
                        feature_set=feature_set,
                        target=target,
                        config=config,
                        meta=meta,
                    )
                    all_metrics.extend(m_rows)
                    all_preds.append(pred_df)
                    all_importances.extend(imps)

                    if config.get("validation", {}).get("also_random_split", True):
                        r_rows, r_pred = run_random_split(
                            X,
                            y,
                            groups,
                            model_name=model_name,
                            feature_set=feature_set,
                            target=target,
                            config=config,
                            meta=meta,
                        )
                        all_metrics.extend(r_rows)
                        all_preds.append(r_pred)
                except Exception as exc:  # noqa: BLE001 — record and continue
                    logger.exception(
                        "Failed %s / %s / %s: %s", model_name, feature_set, target, exc
                    )
                    all_metrics.append(
                        {
                            "model": model_name,
                            "feature_set": feature_set,
                            "target": target,
                            "split": "grouped_cv_oof",
                            "fold": "all",
                            "mae": None,
                            "rmse": None,
                            "r2": None,
                            "spearman": None,
                            "kendall": None,
                            "failure": f"{type(exc).__name__}: {exc}",
                        }
                    )

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    metrics_path = Path(config.get("output", {}).get("metrics_csv", output_dir / "surrogate_metrics.csv"))
    if not metrics_path.is_absolute():
        metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    preds_path = pred_dir / "all_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    imp_path = pred_dir / "tree_importances.json"
    with imp_path.open("w", encoding="utf-8") as handle:
        json.dump(all_importances, handle)

    summary = {
        "n_metric_rows": int(len(metrics_df)),
        "n_prediction_rows": int(len(preds_df)),
        "models": model_names,
        "feature_sets": feature_sets,
        "targets": targets,
        "metrics_path": relpath_to_project(metrics_path),
        "predictions_path": relpath_to_project(preds_path),
        "importances_path": relpath_to_project(imp_path),
    }
    with (output_dir / "phase2_run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "Phase 2 complete: %d metric rows, %d prediction rows",
        len(metrics_df),
        len(preds_df),
    )
    return {
        "metrics": metrics_df,
        "predictions": preds_df,
        "importances": all_importances,
        "summary": summary,
    }
