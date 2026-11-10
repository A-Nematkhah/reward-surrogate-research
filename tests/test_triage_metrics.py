"""Unit tests for generation-level triage metrics."""

from __future__ import annotations

import numpy as np
import pytest

from rsr.evaluation.cost import triage_cost
from rsr.evaluation.triage_metrics import (
    pairwise_accuracy,
    recall_at_k,
    regret_at_k,
    select_topk_indices,
    spearman_safe,
    top1_hit,
)


def test_n2_recall_and_regret() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0.9, 0.1])
    s = np.array([0.2, 0.8])  # ranks inverted
    assert recall_at_k(y, s, 1, rng=rng) == 0.0
    assert regret_at_k(y, s, 1, rng=rng) == pytest.approx(0.8)
    assert recall_at_k(y, s, 2, rng=rng) == 1.0
    assert regret_at_k(y, s, 2, rng=rng) == pytest.approx(0.0)


def test_n3_top1_hit() -> None:
    rng = np.random.default_rng(1)
    y = np.array([0.5, 0.8, 0.2])
    s = np.array([0.1, 0.9, 0.0])
    assert top1_hit(y, s, rng=rng) == 1.0


def test_constant_group_spearman_nan() -> None:
    y = np.array([0.5, 0.5, 0.5, 0.5])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.isnan(spearman_safe(y, s))
    # n=3 even with variation -> NaN (min_n=4)
    assert np.isnan(spearman_safe([0.1, 0.2, 0.3], [0.3, 0.2, 0.1]))


def test_ties_seeded_break() -> None:
    y = np.array([0.9, 0.5, 0.5])
    s = np.array([0.1, 0.7, 0.7])  # tie for top score
    a = select_topk_indices(s, 1, rng=np.random.default_rng(0))
    b = select_topk_indices(s, 1, rng=np.random.default_rng(0))
    c = select_topk_indices(s, 1, rng=np.random.default_rng(1))
    assert a.tolist() == b.tolist()
    # Different seeds may pick different tied indices
    assert a.shape == (1,)
    assert c.shape == (1,)
    assert set(a.tolist()) <= {1, 2}
    assert set(c.tolist()) <= {1, 2}


def test_k_ge_n_keeps_all() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0.2, 0.9, 0.4])
    s = np.array([0.0, 0.0, 1.0])
    assert recall_at_k(y, s, 10, rng=rng) == 1.0
    assert regret_at_k(y, s, 10, rng=rng) == pytest.approx(0.0)
    kept = select_topk_indices(s, 10, rng=rng)
    assert set(kept.tolist()) == {0, 1, 2}


def test_pairwise_accuracy_perfect_and_weighted() -> None:
    y = np.array([0.1, 0.5, 0.9])
    s = np.array([0.0, 0.4, 0.8])
    assert pairwise_accuracy(y, s) == pytest.approx(1.0)
    se = np.array([0.02, 0.02, 0.02])
    assert pairwise_accuracy(y, s, se=se) == pytest.approx(1.0)


def test_cost_savings() -> None:
    info = triage_cost(16, 5, feature_steps_per_candidate=0, full_run_steps=50_000)
    assert info["savings_frac"] == pytest.approx(1.0 - 5 / 16)
    assert info["meets_50pct_savings"] == 1.0
    early = triage_cost(16, 5, feature_steps_per_candidate=12_000, full_run_steps=50_000)
    # cost = 12k*16 + 50k*5 = 192k + 250k = 442k; baseline 800k; savings 0.4475
    assert early["savings_frac"] == pytest.approx(1.0 - 442_000 / 800_000)
    assert early["meets_50pct_savings"] == 0.0
