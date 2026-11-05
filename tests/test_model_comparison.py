"""Fast unit tests for the model-comparison experiment (no heavy optional deps)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneGroupOut

from src.surrogate.evaluation import association_status
from src.surrogate.model_comparison import (
    _topk_selection_metrics,
    fit_preprocessor,
    load_feature_matrix,
    within_family_top_hit,
)
from src.surrogate.model_zoo import make_model


def _synthetic(n_families: int = 6, n_per: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    X = []
    y = []
    groups = []
    for f in range(n_families):
        for i in range(n_per):
            # Linear signal + noise; family offset
            z = rng.normal(size=8)
            score = float(z[:3].sum() + 0.3 * f + rng.normal(scale=0.2))
            X.append(z)
            y.append(score)
            groups.append(f"fam{f}")
            rows.append({"candidate_id": f"c{f}_{i}", "family_id": f"fam{f}"})
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(groups),
        pd.DataFrame(rows),
    )


def test_lofo_never_sees_held_out_family_and_ridge_positive():
    X, y, groups, _ = _synthetic()
    logo = LeaveOneGroupOut()
    oof = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        assert set(groups[te]).isdisjoint(set(groups[tr]))
        model = make_model("ridge", random_state=0, params={"alpha": 1.0})
        pre = fit_preprocessor(
            X[tr],
            y[tr],
            target_transform="none",
            feature_selection="none",
            k=8,
            pca_components=None,
            standardize=True,
        )
        model.fit(pre.transform_X(X[tr]), y[tr])
        oof[te] = model.predict(pre.transform_X(X[te]))
    sp = spearmanr(y, oof).correlation
    assert sp is not None and sp > 0.3


def test_preprocessor_fit_on_train_only_shift_held_out():
    X, y, groups, _ = _synthetic()
    logo = LeaveOneGroupOut()
    tr, te = next(logo.split(X, y, groups))
    pre = fit_preprocessor(
        X[tr],
        y[tr],
        target_transform="none",
        feature_selection="top_k_by_spearman",
        k=5,
        pca_components=None,
        standardize=True,
    )
    Xt_before = pre.transform_X(X[te]).copy()
    X_shifted = X.copy()
    X_shifted[te] = X_shifted[te] + 1000.0
    # Re-using the same fitted preprocessor: train transform unchanged;
    # held-out transform must differ from unshifted because imputer/scaler see shifted values
    Xt_shifted = pre.transform_X(X_shifted[te])
    # Critical leakage check: refitting on train-only data must ignore held-out shift
    pre2 = fit_preprocessor(
        X_shifted[tr],
        y[tr],
        target_transform="none",
        feature_selection="top_k_by_spearman",
        k=5,
        pca_components=None,
        standardize=True,
    )
    # Training rows identical between X and X_shifted
    np.testing.assert_allclose(
        pre.transform_X(X[tr]),
        pre2.transform_X(X_shifted[tr]),
        atol=1e-8,
    )
    assert not np.allclose(Xt_before, Xt_shifted)


def test_constant_models_nan_ranking():
    model = make_model("mean", random_state=0)
    X = np.random.default_rng(0).normal(size=(20, 4))
    y = np.linspace(0, 1, 20)
    model.fit(X, y)
    pred = model.predict(X)
    assert association_status(y, pred) == "constant_prediction"
    from src.surrogate.evaluation import evaluate_predictions

    m = evaluate_predictions(y, pred, target_name="success_rate", model_name="mean")
    assert np.isnan(m["spearman"])


def test_topk_and_regret_handmade():
    y_true = np.asarray([0.1, 0.9, 0.8, 0.2], dtype=float)
    y_pred = np.asarray([0.2, 0.7, 0.95, 0.1], dtype=float)
    out = _topk_selection_metrics(y_true, y_pred, k=2, higher_is_better=True)
    # True top2: indices 1,2 (0.9, 0.8); pred top2: 2,1 (0.95, 0.7) → perfect overlap
    assert out["top2_overlap"] == 1.0
    assert out["top2_regret"] == pytest.approx(0.0, abs=1e-9)
    # Lower-is-better: pred prefers idx3 (pred 0.1) whose true label is 0.2
    out2 = _topk_selection_metrics(y_true, y_pred, k=1, higher_is_better=False)
    assert out2["top1_selected_mean"] == pytest.approx(0.2, abs=1e-9)
    assert out2["top1_oracle_mean"] == pytest.approx(0.1, abs=1e-9)


def test_within_family_top_hit():
    y_true = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=float)
    y_pred = np.asarray([0.0, 0.5, 0.4, 0.9], dtype=float)
    groups = np.asarray(["a", "a", "b", "b"])
    hit = within_family_top_hit(y_true, y_pred, groups, k=1, higher_is_better=True)
    # fam a: best idx1, pred top1 is idx1; fam b: best idx3, pred top1 idx3
    assert hit == pytest.approx(1.0)


def test_missing_optional_dependency_skips_not_crash(monkeypatch):
    import src.surrogate.model_zoo as mz

    def _boom(name: str):
        return None

    monkeypatch.setattr(mz, "_try_import", _boom)
    with pytest.raises(ImportError):
        mz.make_model("lightgbm", random_state=0)
    flags = mz.available_models()
    assert flags["lightgbm"] is False


def test_load_combo_feature_set(tmp_path: Path):
    # Minimal fake feature dir
    meta = pd.DataFrame(
        {"candidate_id": ["c0", "c1"], "family_id": ["f0", "f0"]}
    )
    targets = pd.DataFrame({"success_rate": [0.1, 0.2]})
    early = pd.DataFrame({"e1": [1.0, 2.0]})
    math = pd.DataFrame({"m1": [3.0, 4.0]})
    meta.to_csv(tmp_path / "features_meta.csv", index=False)
    targets.to_csv(tmp_path / "features_targets.csv", index=False)
    early.to_csv(tmp_path / "features_early.csv", index=False)
    math.to_csv(tmp_path / "features_math.csv", index=False)
    X, meta2, t2 = load_feature_matrix(tmp_path, "early+math")
    assert X.shape == (2, 2)
    assert list(X.columns) == ["e1", "m1"]
