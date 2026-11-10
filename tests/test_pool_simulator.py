"""Tests for the OOF pool simulator."""

from __future__ import annotations

import pandas as pd

from rsr.evaluation.pool_simulator import draw_pool, load_oof_candidates, simulate_pools


def _toy_candidates() -> pd.DataFrame:
    rows = []
    for fam_i, fam in enumerate(["a", "b", "c", "d"]):
        for j in range(6):
            rows.append(
                {
                    "candidate_id": f"{fam}_{j}",
                    "family_id": fam,
                    "y_true": 0.1 * j + 0.01 * fam_i,
                    "y_pred": 0.08 * j + 0.02 * fam_i,
                }
            )
    return pd.DataFrame(rows)


def test_draw_pool_adds_groups_when_too_small() -> None:
    import numpy as np

    cand = _toy_candidates()
    # Request 1 group but pool_size=10 > family size 6 -> must realize >1 groups
    rng = np.random.default_rng(0)
    pool = draw_pool(cand, pool_size=10, n_groups=1, rng=rng, expand_groups=True)
    assert pool is not None
    assert pool.n_groups >= 2
    assert len(pool.candidate_ids) == 10
    assert len(set(pool.candidate_ids)) == 10  # no duplicates


def test_draw_pool_strict_within_family_no_expand() -> None:
    import numpy as np

    cand = _toy_candidates()
    rng = np.random.default_rng(0)
    # Family size is 6; N=10 cannot be filled without expand -> None
    assert (
        draw_pool(cand, pool_size=10, n_groups=1, rng=rng, expand_groups=False) is None
    )
    pool = draw_pool(cand, pool_size=6, n_groups=1, rng=rng, expand_groups=False)
    assert pool is not None
    assert pool.n_groups == 1
    assert len(set(pool.candidate_ids)) == 6


def test_simulate_pools_smoke() -> None:
    cand = _toy_candidates()
    per, summary = simulate_pools(
        cand,
        pool_sizes=(8,),
        n_groups_list=(1, 2),
        ks=(3,),
        n_pools=20,
        selectors=("surrogate", "random", "oracle"),
        seed=0,
        n_boot=50,
    )
    assert not per.empty
    assert set(per["selector"]) == {"surrogate", "random", "oracle"}
    assert not summary.empty
    assert (summary["N"] == 8).all()


def test_load_oof_candidates_slice() -> None:
    oof = pd.DataFrame(
        {
            "repeat": [0, 0, 1, 1],
            "fold": [0, 0, 0, 0],
            "family_id": ["f", "f", "f", "g"],
            "candidate_id": ["a", "b", "a", "c"],
            "model": ["extra_trees"] * 4,
            "feature_set": ["early+math+behavioral"] * 4,
            "target": ["success_rate"] * 4,
            "y_true": [0.1, 0.2, 0.1, 0.3],
            "y_pred": [0.15, 0.25, 0.11, 0.28],
            "score": [0.15, 0.25, 0.11, 0.28],
        }
    )
    m = load_oof_candidates(oof, repeat=0)
    assert len(m) == 2
    assert set(m["candidate_id"]) == {"a", "b"}
