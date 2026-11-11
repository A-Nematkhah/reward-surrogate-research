"""Phase 6c: no code_hash leakage across folds; seeds depend on run_id."""

from __future__ import annotations

import numpy as np
import pytest

from reward_triage.gate2_harness import (
    assert_no_code_hash_leak,
    dedupe_by_code_hash,
    lolo_oof,
    seeds_for_run,
)
from reward_triage.label_cache import LabelCache
from reward_triage.store import SeedLabel


def test_seeds_for_run_independent():
    a = seeds_for_run("phase6_2d_r1b", n_seeds=3)
    b = seeds_for_run("phase6_2d_r2", n_seeds=3)
    assert a != b
    assert len(a) == 3
    assert seeds_for_run("phase6_2d_r1b") == a


def test_assert_no_code_hash_leak_raises():
    with pytest.raises(AssertionError):
        assert_no_code_hash_leak({"abc"}, {"abc", "def"})
    assert_no_code_hash_leak({"abc"}, {"def"})


def test_dedupe_averages_labels():
    import pandas as pd

    df = pd.DataFrame(
        {
            "code_hash": ["h1", "h1", "h2"],
            "y_true": [0.1, 0.3, 0.5],
            "seed_se": [0.01, 0.01, 0.02],
            "generation": [0, 0, 1],
            "lineage_id": ["l1", "l2", "l3"],
            "run_id": ["r1", "r2", "r1"],
            "candidate_id": ["c1", "c2", "c3"],
            "fold_key": ["h1::l1", "h1::l2", "h2::l3"],
            "decision": ["train", "train", "skip"],
            "probability": [1.0, 1.0, 0.0],
            "live_score": [0.2, 0.4, None],
            "f0": [1.0, 3.0, 5.0],
        }
    )
    out = dedupe_by_code_hash(df)
    assert len(out) == 2
    row = out[out["code_hash"] == "h1"].iloc[0]
    assert abs(float(row["y_true"]) - 0.2) < 1e-9
    assert abs(float(row["f0"]) - 2.0) < 1e-9


def test_lolo_code_hash_no_leak():
    rng = np.random.default_rng(0)
    n = 12
    rows = []
    for i in range(n):
        rows.append(
            {
                "candidate_id": f"c{i}",
                "code_hash": f"h{i}",
                "lineage_id": f"lin{i}",
                "run_id": "r",
                "generation": 0,
                "y_true": float(rng.uniform()),
                "seed_se": 0.02,
                "fold_key": f"h{i}",
                "decision": "train",
                "probability": 1.0,
                "live_score": None,
                **{f"f{j}": float(rng.normal()) for j in range(4)},
            }
        )
    import pandas as pd

    df = pd.DataFrame(rows)
    cols = [f"f{j}" for j in range(4)]
    oof = lolo_oof(
        df, cols, fold_col="code_hash", use_prior=False, enforce_no_hash_leak=True
    )
    assert len(oof) == n
    assert oof["y_pred"].notna().all()


def test_label_cache_roundtrip(tmp_path):
    cache = LabelCache(tmp_path / "cache")
    labels = [
        SeedLabel(
            seed=0,
            success=0.5,
            collision=0.1,
            timeout=0.1,
            return_mean=1.0,
            eval_episodes=10,
            train_steps=50000,
            wall_time_s=1.0,
        )
    ]
    assert cache.get("abcd") is None
    cache.put("abcd", labels)
    got = cache.get("abcd")
    assert got is not None
    assert got[0].success == 0.5
    assert cache.stats()["hits"] == 1
    assert cache.stats()["saved_ppo_steps"] == 50000
