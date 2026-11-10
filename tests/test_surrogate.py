"""Tests for surrogate baselines, metrics, and grouped CV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from rsr.surrogate.baselines import MeanPredictor, MedianPredictor, make_ridge_baseline
from rsr.surrogate.evaluation import (
    evaluate_predictions,
    pairwise_ranking_accuracy,
    regression_metrics,
    top_k_accuracy,
)
from rsr.surrogate.models import (
    make_hist_gradient_boosting,
    make_random_forest,
    predict_with_uncertainty,
)
from rsr.surrogate.training import run_grouped_cv, run_phase2_experiments


def test_mean_median_predictors() -> None:
    X = np.random.randn(20, 3)
    y = np.linspace(0, 1, 20)
    mean_m = MeanPredictor().fit(X, y)
    med_m = MedianPredictor().fit(X, y)
    assert np.allclose(mean_m.predict(X[:5]), y.mean())
    assert np.allclose(med_m.predict(X[:5]), np.median(y))


def test_ridge_and_rf_fit_predict() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=40)
    ridge = make_ridge_baseline().fit(X, y)
    pred = ridge.predict(X)
    assert pred.shape == y.shape
    rf = make_random_forest(n_estimators=20, random_state=0).fit(X, y)
    mean, std = predict_with_uncertainty(rf, X)
    assert mean.shape == y.shape
    assert std is not None and std.shape == y.shape


def test_regression_metrics_perfect() -> None:
    y = np.array([0.1, 0.2, 0.3, 0.4])
    m = regression_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["ranking_status"] == "ok"


def test_top_k_and_pairwise_perfect() -> None:
    y = np.array([0.1, 0.4, 0.2, 0.9, 0.3])
    assert top_k_accuracy(y, y, k=2, higher_is_better=True) == pytest.approx(1.0)
    assert pairwise_ranking_accuracy(y, y, higher_is_better=True) == pytest.approx(1.0)
    # Lower-is-better: reverse labels should still score 1.0 when preds match
    assert top_k_accuracy(y, y, k=1, higher_is_better=False) == pytest.approx(1.0)
    em = evaluate_predictions(y, y, target_name="success_rate")
    assert em["top_1_accuracy"] == pytest.approx(1.0)
    assert em["top_3_accuracy"] == pytest.approx(1.0)
    assert em["pairwise_accuracy"] == pytest.approx(1.0)


def test_hist_gradient_boosting_fit() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=40)
    model = make_hist_gradient_boosting(max_iter=30, random_state=0).fit(X, y)
    pred, std = predict_with_uncertainty(model, X)
    assert pred.shape == y.shape
    assert std is None


def test_constant_prediction_ranking_is_nan() -> None:
    y = np.array([0.1, 0.2, 0.3, 0.4])
    p = np.array([0.25, 0.25, 0.25, 0.25])
    m = regression_metrics(y, p)
    assert np.isnan(m["spearman"])
    assert m["ranking_status"] == "constant_prediction"
    from rsr.surrogate.evaluation import evaluate_predictions

    em = evaluate_predictions(y, p, target_name="success_rate", model_name="mean")
    assert np.isnan(em["spearman"])
    assert em["ranking_status"] == "constant_predictor_model"


def test_grouped_cv_runs(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 24
    X = rng.normal(size=(n, 6))
    # Family-correlated target
    families = np.array(["a", "b", "c", "d"])[np.arange(n) % 4]
    y = (families == "a").astype(float) * 0.5 + rng.normal(scale=0.05, size=n)
    meta = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(n)],
            "family_id": families,
        }
    )
    config = {
        "random_seed": 0,
        "validation": {"strategy": "leave_one_family_out", "n_cv_folds": 4},
        "hyperparameters": {
            "random_forest": {
                "n_estimators": 30,
                "max_depth": [None],
                "min_samples_leaf": [1],
            }
        },
        "preprocessing": {"scale_features": True},
    }
    rows, preds, _ = run_grouped_cv(
        X,
        y,
        families,
        model_name="random_forest",
        feature_set="static",
        target="success_rate",
        config=config,
        meta=meta,
    )
    assert any(r["split"] == "lofo_oof" for r in rows)
    assert any("fold_spearman_mean" in r for r in rows if r["split"] == "lofo_oof")
    assert len(preds) == n


def test_phase2_on_tiny_feature_bundle(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 16
    families = np.array(["f1", "f2", "f3", "f4"])[np.arange(n) % 4]
    meta = pd.DataFrame(
        {"candidate_id": [f"c{i}" for i in range(n)], "family_id": families}
    )
    static = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"s{i}" for i in range(4)])
    behavioral = pd.DataFrame(rng.normal(size=(n, 3)), columns=[f"b{i}" for i in range(3)])
    hybrid = pd.concat([static, behavioral], axis=1)
    targets = pd.DataFrame(
        {
            "success_rate": np.clip(0.3 + 0.2 * (families == "f1") + rng.normal(0, 0.05, n), 0, 1),
            "collision_rate": np.clip(rng.random(n) * 0.3, 0, 1),
            "timeout_rate": np.clip(rng.random(n) * 0.3, 0, 1),
            "mean_goal_time": 10 + rng.random(n) * 5,
            "mean_path_length": 5 + rng.random(n) * 2,
        }
    )
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    meta.to_csv(feat_dir / "features_meta.csv", index=False)
    static.to_csv(feat_dir / "features_static.csv", index=False)
    behavioral.to_csv(feat_dir / "features_behavioral.csv", index=False)
    hybrid.to_csv(feat_dir / "features_hybrid.csv", index=False)
    targets.to_csv(feat_dir / "features_targets.csv", index=False)

    config = {
        "random_seed": 0,
        "feature_sets": ["static"],
        "targets": ["success_rate"],
        "baselines": ["mean", "ridge"],
        "surrogates": ["random_forest"],
        "validation": {
            "strategy": "leave_one_family_out",
            "n_cv_folds": 4,
            "also_random_split": True,
            "random_test_size": 0.25,
        },
        "hyperparameters": {
            "ridge": {"alphas": [0.1, 1.0]},
            "random_forest": {
                "n_estimators": 20,
                "max_depth": [None],
                "min_samples_leaf": [1],
            },
            "gaussian_process": {"normalize_y": True, "alpha": 1e-6},
        },
        "preprocessing": {"scale_features": True},
        "output": {
            "results_dir": str(tmp_path / "out"),
            "metrics_csv": str(tmp_path / "out" / "metrics.csv"),
            "predictions_dir": str(tmp_path / "out" / "predictions"),
        },
    }
    result = run_phase2_experiments(feat_dir, config, output_dir=tmp_path / "out")
    assert not result["metrics"].empty
    assert (tmp_path / "out" / "metrics.csv").is_file()
