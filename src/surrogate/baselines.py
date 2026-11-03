"""Naive and linear baselines for reward-performance prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class MeanPredictor(BaseEstimator, RegressorMixin):
    """Predict the training-set mean for every sample."""

    def fit(self, X: Any, y: Any) -> "MeanPredictor":
        X, y = check_X_y(X, y, multi_output=False, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        finite = np.asarray(y, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("MeanPredictor requires at least one finite training target")
        self.mean_ = float(np.mean(finite))
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "mean_")
        X = check_array(X)
        return np.full(shape=(X.shape[0],), fill_value=self.mean_, dtype=np.float64)


class MedianPredictor(BaseEstimator, RegressorMixin):
    """Predict the training-set median for every sample."""

    def fit(self, X: Any, y: Any) -> "MedianPredictor":
        X, y = check_X_y(X, y, multi_output=False, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        finite = np.asarray(y, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("MedianPredictor requires at least one finite training target")
        self.median_ = float(np.median(finite))
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "median_")
        X = check_array(X)
        return np.full(shape=(X.shape[0],), fill_value=self.median_, dtype=np.float64)


def make_ridge_baseline(
    alphas: list[float] | None = None,
    *,
    scale_features: bool = True,
) -> Pipeline:
    alphas = alphas or [0.1, 1.0, 10.0, 100.0]
    ridge = RidgeCV(alphas=np.asarray(alphas, dtype=np.float64))
    if scale_features:
        return Pipeline([("scaler", StandardScaler()), ("model", ridge)])
    return Pipeline([("model", ridge)])
