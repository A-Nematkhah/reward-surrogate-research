"""Tests for static/behavioral feature extraction and leakage guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.behavioral_features import extract_behavioral_features
from src.features.early_behavioral import collect_early_rollout_features
from src.features.feature_builder import (
    aggregate_phase1_dataset,
    build_feature_tables,
    ensure_probe_set,
)
from src.features.interaction_features import extract_interaction_features
from src.features.math_features import extract_math_features
from src.features.probe_trajectories import (
    generate_probe_set,
    load_probe_set,
    save_probe_set,
)
from src.features.static_features import (
    extract_static_features_from_spec,
    static_feature_matrix_from_dataframe,
)
from src.rewards.reward_function import RewardSpec
from src.utils.config import load_named_config


@pytest.fixture
def env_config(config_dir: Path) -> dict:
    return load_named_config("environment", config_dir=config_dir)


@pytest.fixture
def probe_config() -> dict:
    return {
        "seed": 7,
        "n_trajectories_per_policy": 2,
        "max_steps": 30,
        "policies": ["random", "goal_seeking", "obstacle_avoiding"],
        "probe_set_path": "experiments/phase2/probe_trajectories.npz",
        "probe_config_path": "experiments/phase2/probe_config.json",
    }


@pytest.fixture
def sample_spec() -> RewardSpec:
    return RewardSpec(
        candidate_id="balanced_00",
        family_id="balanced",
        weights={
            "progress": 1.5,
            "goal": 10.0,
            "collision": -10.0,
            "obstacle": -1.0,
            "smoothness": -0.2,
            "time": -0.01,
        },
    )


def test_static_features_no_metric_keys(sample_spec: RewardSpec) -> None:
    feats = extract_static_features_from_spec(sample_spec)
    assert "w_progress" in feats
    assert "ratio_goal_over_collision" in feats
    assert "success_rate" not in feats
    assert all(np.isfinite(v) for v in feats.values())


def test_static_matrix_rejects_leakage_columns() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "family_id": ["balanced", "near_zero"],
            "weight_progress": [1.0, 0.0],
            "weight_goal": [10.0, 0.0],
            "weight_collision": [-10.0, 0.0],
            "weight_obstacle": [-1.0, 0.0],
            "weight_smoothness": [-0.1, 0.0],
            "weight_time": [-0.01, 0.0],
        }
    )
    feat_df, cols = static_feature_matrix_from_dataframe(
        df, include_family_one_hot=False
    )
    assert "family_balanced" not in cols
    assert "w_progress" in cols or "w_progress" in feat_df.columns
    assert "success_rate" not in feat_df.columns

    feat_oh, cols_oh = static_feature_matrix_from_dataframe(
        df, include_family_one_hot=True
    )
    assert "family_balanced" in cols_oh
    assert "success_rate" not in feat_oh.columns


def test_probe_generation_reproducible(env_config: dict, probe_config: dict) -> None:
    a = generate_probe_set(env_config, probe_config)
    b = generate_probe_set(env_config, probe_config)
    assert len(a) == 6
    assert {ep.policy_name for ep in a} == {
        "random",
        "goal_seeking",
        "obstacle_avoiding",
    }
    assert [ep.seed for ep in a] == [ep.seed for ep in b]
    assert len(a[0].transitions) == len(b[0].transitions)
    if a[0].transitions:
        np.testing.assert_allclose(
            a[0].transitions[0]["position"], b[0].transitions[0]["position"]
        )


def test_probe_save_load(tmp_path: Path, env_config: dict, probe_config: dict) -> None:
    episodes = generate_probe_set(env_config, probe_config)
    npz_path = tmp_path / "probes.npz"
    cfg_path = tmp_path / "probes.json"
    save_probe_set(
        episodes, npz_path=npz_path, config_path=cfg_path, probe_config=probe_config
    )
    loaded = load_probe_set(npz_path)
    assert len(loaded) == len(episodes)
    assert loaded[0].policy_name == episodes[0].policy_name
    assert len(loaded[0].transitions) == len(episodes[0].transitions)


def test_behavioral_features_finite(
    env_config: dict, probe_config: dict, sample_spec: RewardSpec
) -> None:
    episodes = generate_probe_set(env_config, probe_config)
    feats = extract_behavioral_features(sample_spec, episodes)
    assert "total_mean" in feats
    assert "comp_progress_mean" in feats
    assert "policy_random_total_mean" in feats
    assert all(np.isfinite(v) for v in feats.values())
    assert "success_rate" not in feats


def test_math_and_interaction_features_finite(
    env_config: dict, probe_config: dict, sample_spec: RewardSpec
) -> None:
    episodes = generate_probe_set(env_config, probe_config)
    math_feats = extract_math_features(sample_spec, episodes)
    ix_feats = extract_interaction_features(sample_spec, episodes)
    assert "math_sparsity" in math_feats
    assert "math_distance_sensitivity_mean" in math_feats
    assert "ix_w_progress_x_collision" in ix_feats
    assert all(np.isfinite(v) for v in math_feats.values())
    assert all(np.isfinite(v) for v in ix_feats.values())
    assert "success_rate" not in math_feats
    assert "success_rate" not in ix_feats


def test_build_feature_tables_shapes(
    env_config: dict, probe_config: dict, tmp_path: Path
) -> None:
    # Tiny fake phase1 dataset
    rows = []
    for i, (cid, fam, sr) in enumerate(
        [
            ("balanced_00", "balanced", 0.8),
            ("near_zero_00", "near_zero", 0.1),
            ("safety_heavy_00", "safety_heavy", 0.3),
        ]
    ):
        rows.append(
            {
                "candidate_id": cid,
                "family_id": fam,
                "training_seed": 0,
                "training_status": "completed",
                "weight_progress": 1.0 - 0.1 * i,
                "weight_goal": 10.0,
                "weight_collision": -10.0,
                "weight_obstacle": -1.0,
                "weight_smoothness": -0.1,
                "weight_time": -0.01,
                "success_rate": sr,
                "collision_rate": 0.1,
                "timeout_rate": 0.1,
                "mean_goal_time": 10.0,
                "mean_path_length": 5.0,
            }
        )
        rows.append({**rows[-1], "training_seed": 1, "success_rate": sr - 0.05})

    df = pd.DataFrame(rows)
    episodes = generate_probe_set(env_config, probe_config)
    tables = build_feature_tables(
        df, episodes, seed_aggregation="mean", include_family_one_hot=False
    )
    assert len(tables["meta"]) == 3
    assert tables["static"].shape[0] == 3
    assert tables["behavioral"].shape[0] == 3
    assert tables["hybrid"].shape[1] == (
        tables["static"].shape[1]
        + tables["behavioral"].shape[1]
        + tables["math"].shape[1]
        + tables["interaction"].shape[1]
    )
    assert "math" in tables and tables["math"].shape[0] == 3
    assert "interaction" in tables and tables["interaction"].shape[0] == 3
    assert not any(c.startswith("family_") for c in tables["static"].columns)
    assert "success_rate" in tables["targets"].columns
    assert "success_rate" not in tables["hybrid"].columns


def test_ensure_probe_set_reuses_file(
    env_config: dict, probe_config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # rewrite paths relative to tmp project root
    local_probe = {
        **probe_config,
        "probe_set_path": "probes.npz",
        "probe_config_path": "probes.json",
    }
    first = ensure_probe_set(env_config, local_probe, project_root=tmp_path)
    second = ensure_probe_set(env_config, local_probe, project_root=tmp_path)
    assert len(first) == len(second)
    assert (tmp_path / "probes.npz").is_file()


class _RandomPolicy:
    """Minimal predict() stub for early-rollout feature tests."""

    def predict(self, obs, deterministic=False):  # noqa: ANN001
        del deterministic
        rng = np.random.default_rng(abs(hash(obs.tobytes())) % (2**32))
        return rng.uniform(-1.0, 1.0, size=2).astype(np.float64), None


def test_early_rollout_features_finite(env_config: dict, sample_spec: RewardSpec) -> None:
    feats = collect_early_rollout_features(
        _RandomPolicy(),
        env_config=env_config,
        reward_spec=sample_spec,
        n_episodes=3,
        seed_start=123,
        deterministic=False,
    )
    assert feats["early_n_episodes"] == 3.0
    assert "early_goal_progress_mean" in feats
    assert "early_return_std" in feats
    assert "success_rate" not in feats
    assert all(k.startswith("early_") for k in feats)
    assert all(np.isfinite(v) for v in feats.values())


def test_aggregate_mean() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "family_id": "balanced",
                "training_seed": 0,
                "training_status": "completed",
                "weight_progress": 1.0,
                "weight_goal": 1.0,
                "weight_collision": -1.0,
                "weight_obstacle": 0.0,
                "weight_smoothness": 0.0,
                "weight_time": 0.0,
                "success_rate": 0.4,
                "collision_rate": 0.1,
                "timeout_rate": 0.5,
                "mean_goal_time": 10.0,
                "mean_path_length": 5.0,
            },
            {
                "candidate_id": "a",
                "family_id": "balanced",
                "training_seed": 1,
                "training_status": "completed",
                "weight_progress": 1.0,
                "weight_goal": 1.0,
                "weight_collision": -1.0,
                "weight_obstacle": 0.0,
                "weight_smoothness": 0.0,
                "weight_time": 0.0,
                "success_rate": 0.6,
                "collision_rate": 0.1,
                "timeout_rate": 0.3,
                "mean_goal_time": 12.0,
                "mean_path_length": 6.0,
            },
        ]
    )
    agg = aggregate_phase1_dataset(df, how="mean")
    assert len(agg) == 1
    assert agg.iloc[0]["success_rate"] == pytest.approx(0.5)
