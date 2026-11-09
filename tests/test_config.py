"""Tests for YAML configuration loading and merging."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    CONFIG_FILES,
    ConfigError,
    apply_dot_overrides,
    deep_update,
    load_config,
    load_merged_configs,
    load_named_config,
    resolve_config_path,
    save_config,
)


def test_all_default_configs_exist(config_dir: Path) -> None:
    for filename in CONFIG_FILES.values():
        path = config_dir / filename
        assert path.is_file(), f"Missing config: {path}"


def test_load_each_named_config(config_dir: Path) -> None:
    for name in CONFIG_FILES:
        config = load_named_config(name, config_dir=config_dir)
        assert isinstance(config, dict)
        assert len(config) > 0


def test_load_merged_configs_keys(config_dir: Path) -> None:
    merged = load_merged_configs(config_dir=config_dir)
    assert set(merged.keys()) == set(CONFIG_FILES.keys())
    assert merged["ppo"]["algorithm"] == "PPO"
    assert merged["environment"]["env_id"] == "RewardNavigation2D-v0"
    assert merged["reward_generation"]["n_candidates"] == 48
    assert "success_rate" in merged["evaluation"]["metrics"]
    assert merged["surrogate"]["surrogates"] == ["extra_trees"]


def test_deep_update_nested() -> None:
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    overrides = {"b": {"c": 99}, "e": 5}
    result = deep_update(base, overrides)
    assert result == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}
    # Original must be unchanged
    assert base["b"]["c"] == 2


def test_save_and_reload_config(tmp_path: Path) -> None:
    config = {"alpha": 1, "nested": {"beta": True}}
    path = tmp_path / "subdir" / "test.yaml"
    save_config(config, path)
    assert path.is_file()
    reloaded = load_config(path)
    assert reloaded == config


def test_apply_dot_overrides() -> None:
    config = {"ppo": {"total_timesteps": 50000, "n_seeds": 2}}
    updated = apply_dot_overrides(
        config,
        ["ppo.total_timesteps=1000", "ppo.n_seeds=1"],
    )
    assert updated["ppo"]["total_timesteps"] == 1000
    assert updated["ppo"]["n_seeds"] == 1
    # Original unchanged
    assert config["ppo"]["total_timesteps"] == 50000


def test_apply_dot_overrides_invalid() -> None:
    with pytest.raises(ConfigError):
        apply_dot_overrides({}, ["no_equals_sign"])


def test_resolve_unknown_config_name(config_dir: Path) -> None:
    with pytest.raises(ConfigError):
        resolve_config_path("does_not_exist", config_dir=config_dir)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_reward_family_counts_match_pool_intent(config_dir: Path) -> None:
    """Families should sum near the configured candidate pool size."""
    config = load_named_config("reward_generation", config_dir=config_dir)
    family_total = sum(fam["n"] for fam in config["families"].values())
    assert family_total == config["n_candidates"]
    assert family_total == 48


def test_crowdnav_reward_pool_is_24(config_dir: Path) -> None:
    config = load_named_config("reward_generation_crowdnav", config_dir=config_dir)
    family_total = sum(fam["n"] for fam in config["families"].values())
    assert family_total == config["n_candidates"] == 24
    assert all(fam["n"] == 2 for fam in config["families"].values())
    expected = {
        "balanced",
        "progress_heavy",
        "progress_dense",
        "time_pressure",
        "human_clearance",
        "crowded_progress",
        "risk_sensitive",
        "efficiency",
        "energy_aware",
        "nonlinear_progress",
        "safety_moderate",
        "weak_contrast",
    }
    assert set(config["families"]) == expected
    for name in ("human_proximity", "human_intrusion", "crowded_progress"):
        assert name in config["components"]


def test_irsim_reward_pool_is_24(config_dir: Path) -> None:
    config = load_named_config("reward_generation_irsim", config_dir=config_dir)
    family_total = sum(fam["n"] for fam in config["families"].values())
    assert family_total == config["n_candidates"] == 24
    assert all(fam["n"] == 2 for fam in config["families"].values())
    assert len(config["families"]) == 12


def test_ppo_crowdnav_budget(config_dir: Path) -> None:
    cfg = load_named_config("ppo_crowdnav", config_dir=config_dir)
    assert int(cfg["total_timesteps"]) == 150000
    assert cfg["early_behavioral"]["checkpoints"] == [5000, 12000, 25000]


def test_configs_are_valid_yaml(config_dir: Path) -> None:
    for filename in CONFIG_FILES.values():
        with (config_dir / filename).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        assert isinstance(data, dict)
