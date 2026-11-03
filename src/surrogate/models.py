"""Classical surrogate regressors (RF, ExtraTrees, GP)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        n_jobs=-1,
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
        n_jobs=-1,
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


def predict_with_uncertainty(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return (mean, uncertainty) where uncertainty is a std-like score or None.

    Notes
    -----
    - GP: predictive standard deviation (model-based).
    - RF / ExtraTrees: std across trees (ensemble disagreement; **not** assumed calibrated).
    - Others: uncertainty is None.
    """
    X = np.asarray(X, dtype=np.float64)

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
        return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)

    mean = model.predict(X)
    return np.asarray(mean, dtype=np.float64), None


def tree_feature_importances(model: Any) -> np.ndarray | None:
    """Extract feature importances from a tree model or pipeline containing one."""
    tree_model = model
    if isinstance(model, Pipeline) and "model" in model.named_steps:
        tree_model = model.named_steps["model"]
    if isinstance(tree_model, (RandomForestRegressor, ExtraTreesRegressor)):
        return np.asarray(tree_model.feature_importances_, dtype=np.float64)
    return None
