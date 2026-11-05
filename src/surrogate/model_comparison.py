"""LOFO model-comparison experiment (does not touch Phase 2 artifacts)."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.surrogate.evaluation import (
    HIGHER_IS_BETTER,
    association_status,
    pairwise_ranking_accuracy,
    regression_metrics,
)
from src.surrogate.model_zoo import (
    available_models,
    is_constant_ranker,
    is_ranking_model,
    make_model,
    needs_groups,
)
from src.utils.config import PROJECT_ROOT, relpath_to_project
from src.utils.logging import get_logger

logger = get_logger("model_comparison")

COMBO_FEATURE_SETS = {
    "early+math": ("early", "math"),
    "early+behavioral": ("early", "behavioral"),
    "early+static": ("early", "static"),
    "math+behavioral": ("math", "behavioral"),
    "early+math+behavioral": ("early", "math", "behavioral"),
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def resolve_path(path: str | Path, *, root: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or PROJECT_ROOT) / p


def load_feature_matrix(
    features_dir: str | Path,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (X_df, meta, targets). Builds combo sets in memory only."""
    features_dir = Path(features_dir)
    meta = pd.read_csv(features_dir / "features_meta.csv")
    targets = pd.read_csv(features_dir / "features_targets.csv")
    if feature_set in COMBO_FEATURE_SETS:
        parts = [
            pd.read_csv(features_dir / f"features_{name}.csv")
            for name in COMBO_FEATURE_SETS[feature_set]
        ]
        for part in parts:
            part.reset_index(drop=True, inplace=True)
        X = pd.concat(parts, axis=1)
        # De-duplicate column names if any overlap
        if X.columns.duplicated().any():
            cols = []
            seen: dict[str, int] = {}
            for c in X.columns:
                if c not in seen:
                    seen[c] = 0
                    cols.append(c)
                else:
                    seen[c] += 1
                    cols.append(f"{c}__{seen[c]}")
            X.columns = cols
    else:
        path = features_dir / f"features_{feature_set}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        X = pd.read_csv(path)
    if len(X) != len(meta) or len(targets) != len(meta):
        raise ValueError("Feature/meta/target row count mismatch")
    return X, meta, targets


@dataclass
class FoldPreprocessor:
    imputer: SimpleImputer
    keep_mask: np.ndarray
    selector_kind: str
    selected_idx: np.ndarray | None
    scaler: StandardScaler | None
    pca: PCA | None
    target_transform: str
    y_train_for_rank: np.ndarray | None

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        X = self.imputer.transform(X)
        X = X[:, self.keep_mask]
        if self.selected_idx is not None:
            X = X[:, self.selected_idx]
        if self.pca is not None:
            X = self.pca.transform(X)
        elif self.scaler is not None:
            X = self.scaler.transform(X)
        return np.asarray(X, dtype=np.float64)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.target_transform == "logit":
            eps = 1e-4
            y_c = np.clip(y, eps, 1.0 - eps)
            return np.log(y_c / (1.0 - y_c))
        if self.target_transform == "rank":
            # Map using training ranks scale: approximate via interpolation on sorted train
            assert self.y_train_for_rank is not None
            order = np.argsort(self.y_train_for_rank)
            xs = self.y_train_for_rank[order]
            ys = np.linspace(0.0, 1.0, len(xs))
            return np.interp(y, xs, ys)
        return y

    def inverse_y(self, y_t: np.ndarray) -> np.ndarray:
        y_t = np.asarray(y_t, dtype=np.float64)
        if self.target_transform == "logit":
            return 1.0 / (1.0 + np.exp(-y_t))
        # rank / none: leave as score (ranking metrics still valid)
        return y_t


def fit_preprocessor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    target_transform: str,
    feature_selection: str,
    k: int,
    pca_components: int | None,
    standardize: bool,
) -> FoldPreprocessor:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train)
    # Drop constant columns
    keep = np.std(X_imp, axis=0) > 1e-12
    if not np.any(keep):
        keep = np.ones(X_imp.shape[1], dtype=bool)
    X_k = X_imp[:, keep]
    selected_idx = None
    pca = None
    scaler = None

    if feature_selection in {"top_k_by_spearman", "top_k_by_mutual_info"}:
        kk = min(int(k), X_k.shape[1])
        scores = np.zeros(X_k.shape[1], dtype=np.float64)
        if feature_selection == "top_k_by_mutual_info":
            try:
                scores = mutual_info_regression(
                    X_k, y_train, random_state=0, n_neighbors=3
                )
            except Exception:  # noqa: BLE001
                feature_selection = "top_k_by_spearman"
        if feature_selection == "top_k_by_spearman":
            for j in range(X_k.shape[1]):
                col = X_k[:, j]
                if np.std(col) < 1e-12 or np.std(y_train) < 1e-12:
                    scores[j] = 0.0
                else:
                    sp = spearmanr(col, y_train).correlation
                    scores[j] = abs(float(sp)) if sp is not None and np.isfinite(sp) else 0.0
        selected_idx = np.argsort(-scores)[:kk]
        X_sel = X_k[:, selected_idx]
    elif feature_selection == "pca":
        n_comp = min(int(pca_components or 5), X_k.shape[0], X_k.shape[1])
        n_comp = max(1, n_comp)
        pca = PCA(n_components=n_comp, random_state=0)
        X_sel = pca.fit_transform(X_k)
        selected_idx = None
    else:
        X_sel = X_k

    if standardize and pca is None:
        scaler = StandardScaler()
        scaler.fit(X_sel)

    return FoldPreprocessor(
        imputer=imputer,
        keep_mask=keep,
        selector_kind=feature_selection,
        selected_idx=selected_idx,
        scaler=scaler,
        pca=pca,
        target_transform=target_transform,
        y_train_for_rank=np.asarray(y_train, dtype=np.float64),
    )


def _within_family_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    min_size: int = 4,
) -> dict[str, float]:
    vals: list[float] = []
    for g in np.unique(groups):
        m = groups == g
        if int(m.sum()) < min_size:
            continue
        status = association_status(y_true[m], y_pred[m])
        if status != "ok":
            continue
        sp = spearmanr(y_true[m], y_pred[m]).correlation
        if sp is not None and np.isfinite(sp):
            vals.append(float(sp))
    if not vals:
        return {
            "within_family_spearman_mean": float("nan"),
            "within_family_spearman_median": float("nan"),
            "within_family_spearman_std": float("nan"),
            "n_families_scored": 0.0,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "within_family_spearman_mean": float(np.mean(arr)),
        "within_family_spearman_median": float(np.median(arr)),
        "within_family_spearman_std": float(np.std(arr)),
        "n_families_scored": float(len(arr)),
    }


def _topk_selection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    k: int,
    higher_is_better: bool,
) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    n = len(yt)
    if n == 0:
        return {
            f"top{k}_selected_mean": float("nan"),
            f"top{k}_oracle_mean": float("nan"),
            f"top{k}_random_mean": float("nan"),
            f"top{k}_overlap": float("nan"),
            f"top{k}_regret": float("nan"),
        }
    kk = min(k, n)
    if higher_is_better:
        true_order = np.argsort(-yt)
        pred_order = np.argsort(-yp)
    else:
        true_order = np.argsort(yt)
        pred_order = np.argsort(yp)
    true_top = true_order[:kk]
    pred_top = pred_order[:kk]
    selected_mean = float(np.mean(yt[pred_top]))
    oracle_mean = float(np.mean(yt[true_top]))
    random_mean = float(np.mean(yt))
    overlap = float(len(set(true_top.tolist()) & set(pred_top.tolist())) / kk)
    regret = float(oracle_mean - selected_mean) if higher_is_better else float(
        selected_mean - oracle_mean
    )
    return {
        f"top{k}_selected_mean": selected_mean,
        f"top{k}_oracle_mean": oracle_mean,
        f"top{k}_random_mean": random_mean,
        f"top{k}_overlap": overlap,
        f"top{k}_regret": regret,
    }


def within_family_top_hit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    k: int = 3,
    higher_is_better: bool = True,
) -> float:
    hits = []
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        if len(m) < 1:
            continue
        yt, yp = y_true[m], y_pred[m]
        if higher_is_better:
            best = int(np.nanargmax(yt))
            top = np.argsort(-yp)[: min(k, len(m))]
        else:
            best = int(np.nanargmin(yt))
            top = np.argsort(yp)[: min(k, len(m))]
        hits.append(1.0 if best in set(top.tolist()) else 0.0)
    return float(np.mean(hits)) if hits else float("nan")


def bootstrap_ci_over_families(
    family_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    higher_is_better: bool,
    min_family_size: int = 4,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    families = np.unique(family_ids)
    pooled_sp: list[float] = []
    within_sp: list[float] = []
    top10_means: list[float] = []
    for _ in range(int(n_boot)):
        samp = rng.choice(families, size=len(families), replace=True)
        masks = [family_ids == f for f in samp]
        # Concatenate sampled families (with replacement => duplicate blocks)
        idx = np.concatenate([np.where(m)[0] for m in masks])
        yt, yp = y_true[idx], y_pred[idx]
        status = association_status(yt, yp)
        if status == "ok":
            sp = spearmanr(yt, yp).correlation
            if sp is not None and np.isfinite(sp):
                pooled_sp.append(float(sp))
        w = _within_family_spearman(
            yt, yp, family_ids[idx], min_size=min_family_size
        )["within_family_spearman_mean"]
        if np.isfinite(w):
            within_sp.append(float(w))
        tk = _topk_selection_metrics(
            yt, yp, k=10, higher_is_better=higher_is_better
        )["top10_selected_mean"]
        if np.isfinite(tk):
            top10_means.append(float(tk))

    def _ci(vals: list[float]) -> tuple[float, float, float]:
        if not vals:
            return float("nan"), float("nan"), float("nan")
        a = np.asarray(vals, dtype=np.float64)
        return float(np.mean(a)), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))

    pm, plo, phi = _ci(pooled_sp)
    wm, wlo, whi = _ci(within_sp)
    tm, tlo, thi = _ci(top10_means)
    return {
        "boot_pooled_spearman_mean": pm,
        "boot_pooled_spearman_lo": plo,
        "boot_pooled_spearman_hi": phi,
        "boot_within_spearman_mean": wm,
        "boot_within_spearman_lo": wlo,
        "boot_within_spearman_hi": whi,
        "boot_top10_mean": tm,
        "boot_top10_lo": tlo,
        "boot_top10_hi": thi,
    }


def _expand_grid(hp: Mapping[str, Any], *, max_combos: int | None) -> list[dict[str, Any]]:
    if not hp:
        return [{}]
    keys = list(hp.keys())
    values = []
    for k in keys:
        v = hp[k]
        if isinstance(v, list):
            values.append(v)
        else:
            values.append([v])
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if max_combos is not None and len(combos) > max_combos:
        # Deterministic truncate
        return combos[:max_combos]
    return combos


def _prep_options(config: Mapping[str, Any], *, quick: bool) -> list[dict[str, Any]]:
    prep = config.get("preprocessing", {})
    tt = list(prep.get("target_transform_options", ["none"]))
    fs = list(prep.get("feature_selection_options", ["none"]))
    ks = list(prep.get("top_k_values", [15]))
    pcs = list(prep.get("pca_components", [5]))
    opts = []
    for t in tt:
        for f in fs:
            if f.startswith("top_k"):
                for k in ks:
                    opts.append(
                        {
                            "target_transform": t,
                            "feature_selection": f,
                            "k": int(k),
                            "pca_components": None,
                        }
                    )
            elif f == "pca":
                for n in pcs:
                    opts.append(
                        {
                            "target_transform": t,
                            "feature_selection": f,
                            "k": int(ks[0]),
                            "pca_components": int(n),
                        }
                    )
            else:
                opts.append(
                    {
                        "target_transform": t,
                        "feature_selection": f,
                        "k": int(ks[0]),
                        "pca_components": None,
                    }
                )
    if quick:
        # Keep a tiny deterministic subset
        defaults = {
            "target_transform": prep.get("default_target_transform", "none"),
            "feature_selection": prep.get("default_feature_selection", "top_k_by_spearman"),
            "k": int(prep.get("default_k", 15)),
            "pca_components": 5,
        }
        return [defaults]
    return opts


def _fit_predict(
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    params: Mapping[str, Any],
    prep_cfg: Mapping[str, Any],
    standardize: bool,
    random_state: int,
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    t0 = time.perf_counter()
    pre = fit_preprocessor(
        X_tr,
        y_tr,
        target_transform=str(prep_cfg["target_transform"]),
        feature_selection=str(prep_cfg["feature_selection"]),
        k=int(prep_cfg.get("k", 15)),
        pca_components=prep_cfg.get("pca_components"),
        standardize=standardize,
    )
    Xtr = pre.transform_X(X_tr)
    Xte = pre.transform_X(X_te)
    ytr = pre.transform_y(y_tr)
    model_params = dict(params)
    if model_name == "gp_ard":
        model_params["n_features_hint"] = int(Xtr.shape[1])
        model_params["ard"] = Xtr.shape[1] <= 15
    model = make_model(model_name, random_state=random_state, params=model_params)
    fit_kwargs: dict[str, Any] = {}
    if needs_groups(model_name):
        fit_kwargs["groups"] = groups_tr
    try:
        model.fit(Xtr, ytr, **fit_kwargs)
    except TypeError:
        model.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred_t = np.asarray(model.predict(Xte), dtype=np.float64)
    pred_time = time.perf_counter() - t1
    # For ranking models / rank transform, keep score; else inverse logit
    if is_ranking_model(model_name) or prep_cfg["target_transform"] == "rank":
        pred = pred_t
    else:
        pred = pre.inverse_y(pred_t)
    meta = {
        "params": dict(params),
        "prep": dict(prep_cfg),
        "n_features_used": int(Xtr.shape[1]),
    }
    return pred, fit_time, pred_time, meta


def _inner_select(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    config: Mapping[str, Any],
    random_state: int,
    quick: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (best_params, best_prep) via inner LOFO on training families."""
    if is_constant_ranker(model_name):
        return {}, {
            "target_transform": "none",
            "feature_selection": "none",
            "k": 15,
            "pca_components": None,
        }

    hp = dict(config.get("hyperparameters", {}).get(model_name, {}))
    # Strip non-grid keys that are lists of lists awkwardly
    max_combos = None
    if quick:
        max_combos = int(config.get("quick_overrides", {}).get("hyperparameter_max_combos", 2))
        qhp = config.get("quick_overrides", {}).get(model_name)
        if isinstance(qhp, dict):
            hp.update(qhp)
    param_grid = _expand_grid(hp, max_combos=max_combos)
    prep_grid = _prep_options(config, quick=quick)
    # Limit joint search
    if not quick and len(param_grid) * len(prep_grid) > 24:
        prep_grid = prep_grid[:4]
        param_grid = param_grid[:6]

    standardize_for = set(config.get("preprocessing", {}).get("standardize_for", []))
    standardize = model_name in standardize_for or model_name.startswith("mlp")

    logo = LeaveOneGroupOut()
    unique = np.unique(groups)
    if len(unique) < 3:
        return param_grid[0], prep_grid[0]

    best_score = -np.inf
    best_params = param_grid[0]
    best_prep = prep_grid[0]
    hib = True  # inner score uses within-family spearman on raw y vs pred score

    for params in param_grid:
        for prep_cfg in prep_grid:
            fold_scores: list[float] = []
            for tr, te in logo.split(X, y, groups):
                try:
                    pred, _, _, _ = _fit_predict(
                        model_name,
                        X[tr],
                        y[tr],
                        groups[tr],
                        X[te],
                        params=params,
                        prep_cfg=prep_cfg,
                        standardize=standardize,
                        random_state=random_state,
                    )
                except Exception:  # noqa: BLE001
                    fold_scores.append(-1.0)
                    continue
                w = _within_family_spearman(y[te], pred, groups[te], min_size=2)[
                    "within_family_spearman_mean"
                ]
                if np.isfinite(w):
                    fold_scores.append(float(w))
                else:
                    # Fallback MAE (lower better -> negate)
                    mae = float(np.mean(np.abs(y[te] - pred)))
                    fold_scores.append(-mae)
            score = float(np.mean(fold_scores)) if fold_scores else -np.inf
            if score > best_score:
                best_score = score
                best_params = params
                best_prep = prep_cfg
    return best_params, best_prep


def _ppo_cost(feature_set: str, config: Mapping[str, Any]) -> int:
    costs = config.get("ppo_cost_steps", {})
    if feature_set in costs:
        return int(costs[feature_set])
    if "early" in feature_set:
        return 5000
    return 0


def evaluate_oof_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    target: str,
    model_name: str,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    hib = bool(config.get("higher_is_better", HIGHER_IS_BETTER).get(target, True))
    min_size = int(config.get("topk", {}).get("min_family_size_for_within", 4))
    metrics: dict[str, Any] = {}
    if is_constant_ranker(model_name):
        reg = regression_metrics(y_true, y_pred)
        metrics.update({k: reg[k] for k in ("mae", "rmse", "r2", "n")})
        metrics["spearman"] = float("nan")
        metrics["kendall"] = float("nan")
        metrics["ranking_status"] = "constant_predictor_model"
    else:
        metrics.update(regression_metrics(y_true, y_pred))
        if is_ranking_model(model_name):
            # MAE/R2 vs raw labels not meaningful for pure rankers
            metrics["mae"] = float("nan")
            metrics["rmse"] = float("nan")
            metrics["r2"] = float("nan")
    metrics.update(_within_family_spearman(y_true, y_pred, groups, min_size=min_size))
    metrics["pairwise_accuracy"] = pairwise_ranking_accuracy(
        y_true, y_pred, higher_is_better=hib
    )
    # Within-family pairwise
    wp = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2:
            continue
        wp.append(
            pairwise_ranking_accuracy(y_true[m], y_pred[m], higher_is_better=hib)
        )
    metrics["within_pairwise_accuracy"] = float(np.nanmean(wp)) if wp else float("nan")
    for k in config.get("topk", {}).get("pooled_k", [5, 10, 20]):
        metrics.update(
            _topk_selection_metrics(y_true, y_pred, k=int(k), higher_is_better=hib)
        )
    metrics["within_family_top3_hit"] = within_family_top_hit(
        y_true,
        y_pred,
        groups,
        k=int(config.get("topk", {}).get("within_family_k", 3)),
        higher_is_better=hib,
    )
    # Per-family spearman list for fold std
    fam_sp = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2:
            continue
        st = association_status(y_true[m], y_pred[m])
        if st != "ok":
            continue
        sp = spearmanr(y_true[m], y_pred[m]).correlation
        if sp is not None and np.isfinite(sp):
            fam_sp.append(float(sp))
    metrics["fold_spearman_mean"] = float(np.mean(fam_sp)) if fam_sp else float("nan")
    metrics["fold_spearman_std"] = float(np.std(fam_sp)) if fam_sp else float("nan")
    n_boot = int(config.get("n_bootstrap", 2000))
    metrics.update(
        bootstrap_ci_over_families(
            groups,
            y_true,
            y_pred,
            n_boot=n_boot,
            seed=seed,
            higher_is_better=hib,
            min_family_size=min_size,
        )
    )
    return metrics


STACK_BASES = [
    ("ridge", "early"),
    ("extra_trees", "behavioral"),
    ("lightgbm", "early+math"),
]


def _run_single_model_lofo(
    model_name: str,
    feature_set: str,
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    repeat: int,
    random_state: int,
    quick: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    logo = LeaveOneGroupOut()
    oof_pred = np.full(len(y), np.nan)
    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    fit_times: list[float] = []
    pred_times: list[float] = []

    standardize_for = set(config.get("preprocessing", {}).get("standardize_for", []))
    standardize = model_name in standardize_for or model_name.startswith("mlp")

    for fold_i, (tr, te) in enumerate(logo.split(X, y, groups)):
        params, prep_cfg = _inner_select(
            model_name,
            X[tr],
            y[tr],
            groups[tr],
            config=config,
            random_state=random_state + fold_i,
            quick=quick,
        )
        try:
            pred, ft, pt, meta_fit = _fit_predict(
                model_name,
                X[tr],
                y[tr],
                groups[tr],
                X[te],
                params=params,
                prep_cfg=prep_cfg,
                standardize=standardize,
                random_state=random_state + fold_i,
            )
        except ImportError as exc:
            logger.info("Skipping %s (missing dependency: %s)", model_name, exc)
            return [], [], {"skipped": True, "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Fold failure %s/%s/%s: %s", model_name, feature_set, target, exc)
            pred = np.full(len(te), np.nan)
            ft = pt = float("nan")
            meta_fit = {"error": str(exc)}
        oof_pred[te] = pred
        fit_times.append(ft)
        pred_times.append(pt)
        hib = bool(config.get("higher_is_better", HIGHER_IS_BETTER).get(target, True))
        fold_metrics = regression_metrics(y[te], pred)
        if is_constant_ranker(model_name) or is_ranking_model(model_name):
            if is_constant_ranker(model_name):
                fold_metrics["spearman"] = float("nan")
                fold_metrics["kendall"] = float("nan")
        w = _within_family_spearman(y[te], pred, groups[te], min_size=2)
        fold_rows.append(
            {
                "repeat": repeat,
                "fold": fold_i,
                "model": model_name,
                "feature_set": feature_set,
                "target": target,
                "test_family": str(groups[te][0]),
                "n_test": int(len(te)),
                "fit_time_sec": ft,
                "predict_time_sec": pt,
                "spearman": fold_metrics.get("spearman"),
                "mae": fold_metrics.get("mae"),
                **w,
                "params": json.dumps(meta_fit.get("params", params)),
                "prep": json.dumps(meta_fit.get("prep", prep_cfg)),
            }
        )
        for i_local, i_global in enumerate(te):
            pred_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold_i,
                    "family_id": meta.iloc[i_global]["family_id"],
                    "candidate_id": meta.iloc[i_global]["candidate_id"],
                    "model": model_name,
                    "feature_set": feature_set,
                    "target": target,
                    "y_true": float(y[i_global]),
                    "y_pred": float(pred[i_local]) if np.isfinite(pred[i_local]) else np.nan,
                    "score": float(pred[i_local]) if np.isfinite(pred[i_local]) else np.nan,
                }
            )

    summary = evaluate_oof_bundle(
        y,
        oof_pred,
        groups,
        target=target,
        model_name=model_name,
        config=config,
        seed=random_state,
    )
    summary.update(
        {
            "repeat": repeat,
            "model": model_name,
            "feature_set": feature_set,
            "target": target,
            "fit_time_sec_sum": float(np.nansum(fit_times)),
            "predict_time_sec_sum": float(np.nansum(pred_times)),
            "needs_ppo_steps": _ppo_cost(feature_set, config),
            "skipped": False,
        }
    )
    return fold_rows, pred_rows, summary


def run_model_comparison(
    config: Mapping[str, Any],
    *,
    targets: Sequence[str] | None = None,
    feature_sets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    cfg = dict(config)
    if quick:
        q = cfg.get("quick_overrides", {})
        cfg["n_repeats"] = int(q.get("n_repeats", 1))
        cfg["n_bootstrap"] = int(q.get("n_bootstrap", 200))
        cfg["quick"] = True

    features_dir = resolve_path(cfg.get("features_dir", "experiments/phase2/features"))
    output_dir = resolve_path(cfg.get("output_dir", "experiments/model_comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = list(targets or cfg.get("targets", ["success_rate"]))
    feature_sets = list(feature_sets or cfg.get("feature_sets", ["early+math"]))
    models = list(models or cfg.get("models", ["extra_trees"]))
    avail = available_models()
    runnable: list[str] = []
    skipped: list[dict[str, Any]] = []
    for m in models:
        if not avail.get(m, False):
            reason = f"optional dependency missing for '{m}'"
            logger.info("Skipping %s (%s)", m, reason)
            skipped.append({"model": m, "reason": reason})
            continue
        runnable.append(m)
    logger.info("Runnable models (%d): %s", len(runnable), runnable)

    n_repeats = int(cfg.get("n_repeats", 3))
    seed0 = int(cfg.get("random_seed", 123))

    all_fold: list[dict[str, Any]] = []
    all_pred: list[dict[str, Any]] = []
    all_sum: list[dict[str, Any]] = []

    # Cache feature matrices
    cache: dict[str, tuple[np.ndarray, pd.DataFrame, pd.DataFrame]] = {}

    def get_xy(fs: str, target: str):
        if fs not in cache:
            X_df, meta, targets_df = load_feature_matrix(features_dir, fs)
            cache[fs] = (
                np.nan_to_num(X_df.to_numpy(dtype=np.float64), nan=0.0),
                meta,
                targets_df,
            )
        X, meta, targets_df = cache[fs]
        if target not in targets_df.columns:
            raise KeyError(target)
        y = targets_df[target].to_numpy(dtype=np.float64)
        groups = meta["family_id"].to_numpy()
        return X, y, groups, meta

    for repeat in range(n_repeats):
        rs = seed0 + 1000 * repeat
        for fs in feature_sets:
            for target in targets:
                X, y, groups, meta = get_xy(fs, target)
                for model_name in runnable:
                    if model_name in {"stack_ridge", "rank_average"}:
                        # Build stacked features from base OOF on training via nested LOFO is expensive;
                        # approximate: fit bases on full LOFO OOF predictions collected first if present,
                        # else skip with message in quick mode unless bases available.
                        logger.info(
                            "Ensemble %s uses simplified stacking from per-fold base fits",
                            model_name,
                        )
                        fold_rows, pred_rows, summary = _run_stack_model(
                            model_name,
                            fs,
                            target,
                            X,
                            y,
                            groups,
                            meta,
                            config=cfg,
                            repeat=repeat,
                            random_state=rs,
                            quick=quick,
                        )
                    else:
                        logger.info(
                            "Running repeat=%s model=%s features=%s target=%s",
                            repeat,
                            model_name,
                            fs,
                            target,
                        )
                        fold_rows, pred_rows, summary = _run_single_model_lofo(
                            model_name,
                            fs,
                            target,
                            X,
                            y,
                            groups,
                            meta,
                            config=cfg,
                            repeat=repeat,
                            random_state=rs,
                            quick=quick,
                        )
                    if summary.get("skipped"):
                        skipped.append(
                            {
                                "model": model_name,
                                "feature_set": fs,
                                "target": target,
                                "reason": summary.get("reason"),
                            }
                        )
                        continue
                    all_fold.extend(fold_rows)
                    all_pred.extend(pred_rows)
                    all_sum.append(summary)

    fold_df = pd.DataFrame(all_fold)
    pred_df = pd.DataFrame(all_pred)
    sum_df = pd.DataFrame(all_sum)

    # Paired vs baseline
    paired_rows = []
    baseline = str(cfg.get("baseline_model", "extra_trees"))
    if not sum_df.empty and not pred_df.empty:
        for fs in feature_sets:
            for target in targets:
                for repeat in sorted(pred_df["repeat"].unique()):
                    base = pred_df[
                        (pred_df.model == baseline)
                        & (pred_df.feature_set == fs)
                        & (pred_df.target == target)
                        & (pred_df.repeat == repeat)
                    ]
                    if base.empty:
                        continue
                    for model_name in pred_df.model.unique():
                        if model_name == baseline:
                            continue
                        other = pred_df[
                            (pred_df.model == model_name)
                            & (pred_df.feature_set == fs)
                            & (pred_df.target == target)
                            & (pred_df.repeat == repeat)
                        ]
                        if other.empty:
                            continue
                        merged = base.merge(
                            other,
                            on=["candidate_id", "family_id"],
                            suffixes=("_base", "_other"),
                        )
                        diffs = []
                        for fam, g in merged.groupby("family_id"):
                            if len(g) < 2:
                                continue
                            sb = association_status(g["y_true_base"], g["y_pred_base"])
                            so = association_status(g["y_true_other"], g["y_pred_other"])
                            if sb != "ok" or so != "ok":
                                continue
                            sp_b = spearmanr(g["y_true_base"], g["y_pred_base"]).correlation
                            sp_o = spearmanr(g["y_true_other"], g["y_pred_other"]).correlation
                            if sp_b is None or sp_o is None:
                                continue
                            diffs.append(float(sp_o) - float(sp_b))
                        if not diffs:
                            continue
                        arr = np.asarray(diffs, dtype=np.float64)
                        rng = np.random.default_rng(seed0)
                        boots = [
                            float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
                            for _ in range(min(int(cfg.get("n_bootstrap", 2000)), 500))
                        ]
                        paired_rows.append(
                            {
                                "repeat": repeat,
                                "feature_set": fs,
                                "target": target,
                                "model": model_name,
                                "baseline": baseline,
                                "mean_spearman_diff": float(np.mean(arr)),
                                "diff_ci_lo": float(np.quantile(boots, 0.025)),
                                "diff_ci_hi": float(np.quantile(boots, 0.975)),
                                "n_families": int(len(arr)),
                                "note": "n_families=12 has low power; CI descriptive only",
                            }
                        )

    paired_df = pd.DataFrame(paired_rows)

    # Selection-bias guard for exploratory grid
    bias_rows = []
    if not sum_df.empty:
        for model_name, g in sum_df.groupby("model"):
            sp = g["spearman"].astype(float)
            bias_rows.append(
                {
                    "model": model_name,
                    "median_spearman_across_feature_sets": (
                        float(np.nanmedian(sp.to_numpy(dtype=float)))
                        if np.isfinite(sp.astype(float)).any()
                        else float("nan")
                    ),
                    "frac_spearman_gt_0_5": float(
                        np.nanmean((sp.astype(float) > 0.5).astype(float))
                    ),
                    "n_configs": int(len(g)),
                }
            )
    bias_df = pd.DataFrame(bias_rows)

    # Persist
    pred_path = output_dir / "oof_predictions.csv"
    fold_path = output_dir / "metrics_by_fold.csv"
    sum_path = output_dir / "metrics_summary.csv"
    topk_path = output_dir / "topk_metrics.csv"
    paired_path = output_dir / "paired_vs_baseline.csv"
    bias_path = output_dir / "selection_bias_guard.csv"
    pred_df.to_csv(pred_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    sum_df.to_csv(sum_path, index=False)
    # topk columns already in summary; also write focused extract
    topk_cols = [c for c in sum_df.columns if c.startswith("top") or c in {
        "model", "feature_set", "target", "repeat", "within_family_top3_hit"
    }]
    sum_df[topk_cols].to_csv(topk_path, index=False)
    paired_df.to_csv(paired_path, index=False)
    bias_df.to_csv(bias_path, index=False)

    run_config = {
        "config": cfg,
        "targets": targets,
        "feature_sets": feature_sets,
        "models_requested": models,
        "skipped": skipped,
        "seed": seed0,
        "quick": quick,
        "package_versions": _package_versions(),
        "git_commit": _git_commit(),
        "paths": {
            "predictions": relpath_to_project(pred_path),
            "metrics_summary": relpath_to_project(sum_path),
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, default=str)

    logger.info("Wrote model comparison artifacts under %s", output_dir)
    return {
        "predictions": pred_df,
        "metrics_by_fold": fold_df,
        "metrics_summary": sum_df,
        "paired": paired_df,
        "selection_bias": bias_df,
        "skipped": skipped,
        "output_dir": output_dir,
    }


def _run_stack_model(
    model_name: str,
    feature_set: str,
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    repeat: int,
    random_state: int,
    quick: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Stack/rank-average using LOFO predictions of fixed base models on same X."""
    avail = available_models()
    # Prefer lightgbm when present; otherwise HGB so stacking still runs offline.
    third = "lightgbm" if avail.get("lightgbm") else "hist_gradient_boosting"
    base_names = ["ridge", "extra_trees", third]
    base_preds = []
    for b in base_names:
        logger.info("Stack base fit: %s (always quick)", b)
        try:
            fr, pr, sm = _run_single_model_lofo(
                b,
                feature_set,
                target,
                X,
                y,
                groups,
                meta,
                config=config,
                repeat=repeat,
                random_state=random_state,
                # Bases must stay cheap; stacking already multiplies cost.
                quick=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Stack base %s failed: %s", b, exc)
            return [], [], {"skipped": True, "reason": f"base {b}: {exc}"}
        if sm.get("skipped"):
            return [], [], {"skipped": True, "reason": sm.get("reason")}
        # Reconstruct OOF vector
        oof = np.full(len(y), np.nan)
        pdf = pd.DataFrame(pr)
        id_to_idx = {cid: i for i, cid in enumerate(meta["candidate_id"].tolist())}
        for _, row in pdf.iterrows():
            oof[id_to_idx[row["candidate_id"]]] = row["y_pred"]
        base_preds.append(oof)
    Z = np.column_stack(base_preds)
    # Second-level LOFO on stacked features
    return _run_single_model_lofo(
        model_name,
        feature_set,
        target,
        Z,
        y,
        groups,
        meta,
        config=config,
        repeat=repeat,
        random_state=random_state,
        quick=quick,
    )


def _package_versions() -> dict[str, str]:
    out = {}
    for name in [
        "numpy",
        "pandas",
        "sklearn",
        "torch",
        "lightgbm",
        "catboost",
        "xgboost",
        "tabpfn",
    ]:
        mod = None
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            out[name] = "not_installed"
    return out


def _git_commit() -> str | None:
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return None
