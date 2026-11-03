"""Tests for reward candidate generation and diversity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.rewards.candidate_generator import (
    generate_candidates,
    load_candidates,
    make_reward_fn,
    save_candidates,
    summarize_candidates,
)
from src.rewards.components import COMPONENT_NAMES
from src.utils.config import load_named_config


@pytest.fixture
def reward_config(config_dir: Path) -> dict:
    return load_named_config("reward_generation", config_dir=config_dir)


def test_generate_expected_count_and_ids(reward_config: dict) -> None:
    candidates = generate_candidates(reward_config, seed=42)
    assert len(candidates) == reward_config["n_candidates"]
    assert 24 <= len(candidates) <= 150

    ids = [c.candidate_id for c in candidates]
    assert len(ids) == len(set(ids))

    family_counts = {}
    for c in candidates:
        family_counts[c.family_id] = family_counts.get(c.family_id, 0) + 1
        assert set(c.weights) == set(COMPONENT_NAMES)
        assert c.generation_seed == 42
        assert "candidate_seed" in c.metadata

    for family_id, cfg in reward_config["families"].items():
        assert family_counts[family_id] == cfg["n"]


def test_generation_is_reproducible(reward_config: dict) -> None:
    a = generate_candidates(reward_config, seed=7)
    b = generate_candidates(reward_config, seed=7)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]

    c = generate_candidates(reward_config, seed=8)
    assert [x.weights for x in a] != [x.weights for x in c]


def test_weights_respect_family_ranges(reward_config: dict) -> None:
    candidates = generate_candidates(reward_config, seed=42)
    ranges_by_family = {
        fid: cfg["weight_ranges"] for fid, cfg in reward_config["families"].items()
    }
    for cand in candidates:
        ranges = ranges_by_family[cand.family_id]
        for name in COMPONENT_NAMES:
            w = cand.weights[name]
            if name not in ranges:
                assert w == 0.0, f"{cand.candidate_id}.{name} expected 0, got {w}"
                continue
            lo, hi = ranges[name]
            assert lo - 1e-9 <= w <= hi + 1e-9, (
                f"{cand.candidate_id}.{name}={w} outside [{lo}, {hi}]"
            )


def test_includes_weak_and_near_zero_families(reward_config: dict) -> None:
    candidates = generate_candidates(reward_config, seed=42)
    families = {c.family_id for c in candidates}
    assert "weak_unbalanced" in families
    assert "near_zero" in families
    assert "safety_heavy" in families
    assert "progress_heavy" in families
    for structural in (
        "energy_aware",
        "risk_sensitive",
        "potential_shaping",
        "nonlinear_progress",
        "distance_dense",
    ):
        assert structural in families


def test_diversity_summary(reward_config: dict) -> None:
    candidates = generate_candidates(reward_config, seed=42)
    summary = summarize_candidates(candidates)
    assert summary["n_candidates"] == len(candidates)
    assert summary["n_families"] == len(reward_config["families"])
    # At least one component should have meaningful spread
    spreads = [
        s["max"] - s["min"] for s in summary["weight_stats"].values()
    ]
    assert max(spreads) > 1.0


def test_save_and_load_roundtrip(reward_config: dict, tmp_path: Path) -> None:
    candidates = generate_candidates(reward_config, seed=42)
    json_path = tmp_path / "candidates.json"
    cand_dir = tmp_path / "candidates"
    save_candidates(candidates, json_path=json_path, candidates_dir=cand_dir)

    assert json_path.is_file()
    assert len(list(cand_dir.glob("*.json"))) == len(candidates)

    loaded = load_candidates(json_path)
    assert len(loaded) == len(candidates)
    assert loaded[0].candidate_id == candidates[0].candidate_id
    assert loaded[0].weights == candidates[0].weights

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["n_candidates"] == len(candidates)


def test_make_reward_fn_callable(reward_config: dict) -> None:
    candidates = generate_candidates(reward_config, seed=42)
    rf = make_reward_fn(candidates[0])
    transition = {
        "prev_position": np.array([0.0, 0.0]),
        "position": np.array([0.5, 0.0]),
        "velocity": np.array([0.5, 0.0]),
        "action": np.array([0.5, 0.0]),
        "prev_action": np.array([0.0, 0.0]),
        "prev_goal_distance": 2.0,
        "goal_distance": 1.5,
        "success": False,
        "collision": False,
        "out_of_bounds": False,
        "timeout": False,
        "obstacles": [],
        "robot_radius": 0.15,
    }
    reward = rf(None, transition)
    assert np.isfinite(reward)


def test_mismatched_n_candidates_raises(reward_config: dict) -> None:
    bad = dict(reward_config)
    bad["n_candidates"] = 999
    with pytest.raises(ValueError, match="n_candidates"):
        generate_candidates(bad, seed=0)
