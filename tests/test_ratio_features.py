"""Tests for Phase-3 ratio features and within-family prune mask."""

from __future__ import annotations

import numpy as np

from rsr.features.ratio_features import extract_ratio_features_from_weights
from rsr.surrogate.model_comparison import _within_family_std_mask


def test_ratio_features_finite_and_scale_invariant_shares():
    w = {
        "progress": 2.0,
        "goal": 1.0,
        "collision": -4.0,
        "obstacle": -1.0,
        "smoothness": 0.0,
        "time": -0.5,
        "energy": 0.0,
        "risk": 0.0,
        "potential": 0.0,
        "nonlinear_progress": 0.0,
        "distance": 0.0,
        "human_proximity": 0.0,
        "human_intrusion": 0.0,
        "crowded_progress": 0.0,
    }
    f = extract_ratio_features_from_weights(w)
    assert f["ratio_progress_over_collision"] > 0
    assert abs(f["share_progress"] + f["share_goal"] + f["share_collision"] + f["share_obstacle"] + f["share_time"] - 1.0) < 1e-6
    assert all(np.isfinite(v) for v in f.values())


def test_within_family_std_mask_drops_flat_cols():
    # col0 varies within family; col1 constant within each family
    X = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )
    groups = np.asarray(["a", "a", "b", "b"])
    keep = _within_family_std_mask(X, groups, min_within_std=0.1)
    assert keep[0]
    assert not keep[1]
