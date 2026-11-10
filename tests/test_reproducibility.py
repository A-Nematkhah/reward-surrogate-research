"""Tests for seed control utilities."""

from __future__ import annotations

import random

import numpy as np
import pytest

from rsr.utils.reproducibility import make_rng, seed_everything, seed_worker


def test_seed_everything_repeatable_numpy() -> None:
    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_seed_everything_repeatable_python_random() -> None:
    seed_everything(7)
    a = [random.random() for _ in range(5)]
    seed_everything(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_different_seeds_differ() -> None:
    seed_everything(1)
    a = np.random.rand(10)
    seed_everything(2)
    b = np.random.rand(10)
    assert not np.allclose(a, b)


def test_seed_everything_metadata() -> None:
    meta = seed_everything(0, deterministic_torch=True)
    assert meta["seed"] == 0
    assert meta["numpy_seeded"] is True
    assert "has_torch" in meta


def test_seed_everything_rejects_negative() -> None:
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_make_rng_isolated() -> None:
    rng_a = make_rng(42)
    rng_b = make_rng(42)
    np.testing.assert_array_equal(rng_a.random(4), rng_b.random(4))


def test_seed_worker_runs() -> None:
    seed_everything(99)
    # Should not raise
    seed_worker(0)
