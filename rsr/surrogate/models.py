"""Classical surrogate regressors (RF, ExtraTrees, GP) plus rank/ensemble wrappers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def make_random_forest(
    *,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        # Keep modest parallelism; nested CV already multiplies work.
        n_jobs=2,
    )


def make_extra_trees(
    *,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=2,
    )


def make_hist_gradient_boosting(
    *,
    max_depth: Optional[int] = None,
    learning_rate: float = 0.1,
    max_iter: int = 100,
    min_samples_leaf: int = 5,
    random_state: int = 0,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )


def make_gaussian_process(
    *,
    normalize_y: bool = True,
    alpha: float = 1e-6,
    random_state: int = 0,
    scale_features: bool = True,
) -> Pipeline | GaussianProcessRegressor:
    # Moderately flexible kernel; WhiteKernel absorbs observation noise.
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=normalize_y,
        n_restarts_optimizer=2,
        random_state=random_state,
    )
    if scale_features:
        return Pipeline([("scaler", StandardScaler()), ("model", gpr)])
    return gpr


class RankExtraTreesRegressor(BaseEstimator, RegressorMixin):
    """ExtraTrees fit on midrank(y); predictions are rank scores (higher ⇒ better rank)."""

    def __init__(
        self,
        *,
        n_estimators: int = 400,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=False)
        ranks = _midranks(np.asarray(y, dtype=np.float64))
        self.estimator_ = make_extra_trees(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, ranks)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        X = check_array(X, dtype=np.float64)
        return self.estimator_.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.feature_importances_


class EnsembleTreesRegressor(BaseEstimator, RegressorMixin):
    """Average RF + ExtraTrees + HistGB predictions."""

    def __init__(
        self,
        *,
        random_state: int = 0,
        rf_params: dict[str, Any] | None = None,
        et_params: dict[str, Any] | None = None,
        hgb_params: dict[str, Any] | None = None,
    ) -> None:
        self.random_state = random_state
        self.rf_params = dict(rf_params or {})
        self.et_params = dict(et_params or {})
        self.hgb_params = dict(hgb_params or {})

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=False)
        rf = make_random_forest(
            n_estimators=int(self.rf_params.get("n_estimators", 300)),
            max_depth=self.rf_params.get("max_depth"),
            min_samples_leaf=int(self.rf_params.get("min_samples_leaf", 1)),
            random_state=self.random_state,
        )
        et = make_extra_trees(
            n_estimators=int(self.et_params.get("n_estimators", 400)),
            max_depth=self.et_params.get("max_depth"),
            min_samples_leaf=int(self.et_params.get("min_samples_leaf", 1)),
            random_state=self.random_state + 1,
        )
        hgb = make_hist_gradient_boosting(
            max_depth=self.hgb_params.get("max_depth"),
            learning_rate=float(self.hgb_params.get("learning_rate", 0.1)),
            max_iter=int(self.hgb_params.get("max_iter", 200)),
            min_samples_leaf=int(self.hgb_params.get("min_samples_leaf", 10)),
            random_state=self.random_state + 2,
        )
        self.estimators_ = [rf, et, hgb]
        for est in self.estimators_:
            est.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64)
        preds = np.stack([est.predict(X) for est in self.estimators_], axis=0)
        return np.mean(preds, axis=0)

    def predict_members(self, X) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64)
        return np.stack([est.predict(X) for est in self.estimators_], axis=0)


def make_rank_extra_trees(
    *,
    n_estimators: int = 400,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
) -> RankExtraTreesRegressor:
    return RankExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )


def make_ensemble_trees(
    *,
    random_state: int = 0,
    rf_params: dict[str, Any] | None = None,
    et_params: dict[str, Any] | None = None,
    hgb_params: dict[str, Any] | None = None,
) -> EnsembleTreesRegressor:
    return EnsembleTreesRegressor(
        random_state=random_state,
        rf_params=rf_params,
        et_params=et_params,
        hgb_params=hgb_params,
    )


def _midranks(y: np.ndarray) -> np.ndarray:
    """Average ranks for ties; higher y ⇒ higher rank (1..n style via argsort)."""
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty(len(y), dtype=np.float64)
    ranks[order] = np.arange(1, len(y) + 1, dtype=np.float64)
    # Average ties
    vals, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sum_ranks = np.bincount(inverse, weights=ranks)
        avg = sum_ranks / counts
        ranks = avg[inverse]
    return ranks


def select_feature_indices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_features: int,
    random_state: int = 0,
) -> np.ndarray:
    """
    Train-fold-only feature pruning via ExtraTrees importances.

    Returns column indices to keep (length <= max_features).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n_features = X_train.shape[1]
    if n_features <= max_features:
        return np.arange(n_features, dtype=int)

    probe = make_extra_trees(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        random_state=random_state,
    )
    probe.fit(X_train, y_train)
    importances = np.asarray(probe.feature_importances_, dtype=np.float64)
    # Break ties with absolute Pearson correlation to y (train only)
    corrs = np.zeros(n_features, dtype=np.float64)
    y_c = y_train - np.mean(y_train)
    y_norm = np.linalg.norm(y_c) + 1e-12
    for j in range(n_features):
        col = X_train[:, j]
        c = col - np.mean(col)
        corrs[j] = abs(float(np.dot(c, y_c) / ((np.linalg.norm(c) + 1e-12) * y_norm)))
    order = np.lexsort((-corrs, -importances))
    return np.sort(order[:max_features])


def predict_with_uncertainty(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return (mean, uncertainty) where uncertainty is a std-like score or None.

    Notes
    -----
    - GP: predictive standard deviation (model-based).
    - RF / ExtraTrees / RankExtraTrees: std across trees (ensemble disagreement).
    - EnsembleTrees: std across member model means.
    - Additionally for tree ensembles, uncertainty is at least half the 10–90%
      quantile spread when that is larger (more conservative AL signal).
    """
    X = np.asarray(X, dtype=np.float64)

    if isinstance(model, EnsembleTreesRegressor):
        members = model.predict_members(X)
        mean = np.mean(members, axis=0)
        std = np.std(members, axis=0)
        return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)

    if isinstance(model, RankExtraTreesRegressor):
        check_is_fitted(model, "estimator_")
        return predict_with_uncertainty(model.estimator_, X)

    if isinstance(model, Pipeline) and "model" in model.named_steps:
        Xt = X
        if "scaler" in model.named_steps:
            Xt = model.named_steps["scaler"].transform(X)
        final = model.named_steps["model"]
    else:
        Xt = X
        final = model

    if isinstance(final, GaussianProcessRegressor):
        mean, std = final.predict(Xt, return_std=True)
        return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)

    if isinstance(final, (RandomForestRegressor, ExtraTreesRegressor)):
        preds = np.stack([est.predict(Xt) for est in final.estimators_], axis=0)
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        q10 = np.percentile(preds, 10, axis=0)
        q90 = np.percentile(preds, 90, axis=0)
        spread = 0.5 * (q90 - q10)
        unc = np.maximum(std, spread)
        return np.asarray(mean, dtype=np.float64), np.asarray(unc, dtype=np.float64)

    mean = model.predict(X)
    return np.asarray(mean, dtype=np.float64), None


def tree_feature_importances(model: Any) -> np.ndarray | None:
    """Extract feature importances from a tree model or pipeline containing one."""
    if isinstance(model, RankExtraTreesRegressor):
        return np.asarray(model.feature_importances_, dtype=np.float64)
    if isinstance(model, EnsembleTreesRegressor):
        # Average RF+ET importances when available
        imps: list[np.ndarray] = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                imps.append(np.asarray(est.feature_importances_, dtype=np.float64))
        if not imps:
            return None
        return np.mean(np.stack(imps, axis=0), axis=0)
    tree_model = model
    if isinstance(model, Pipeline) and "model" in model.named_steps:
        tree_model = model.named_steps["model"]
    if isinstance(tree_model, (RandomForestRegressor, ExtraTreesRegressor)):
        return np.asarray(tree_model.feature_importances_, dtype=np.float64)
    return None
