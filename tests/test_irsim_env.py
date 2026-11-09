"""Smoke tests for IRSimNav-v0 (skipped if irsim is unavailable)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.environment.navigation_2d import make_env
from src.utils.config import PROJECT_ROOT, load_merged_configs

IRSIM_WORLD = PROJECT_ROOT / "env" / "worlds" / "irsim_pipeline_nav.yaml"

pytest.importorskip("irsim")


@pytest.fixture()
def irsim_config() -> dict:
    cfg = load_merged_configs(names=["environment"])
    env = dict(cfg["environment"])
    env["env_id"] = "IRSimNav-v0"
    env.setdefault("irsim", {})
    env["irsim"]["world_yaml"] = str(IRSIM_WORLD)
    return env


def test_irsim_world_yaml_exists() -> None:
    assert IRSIM_WORLD.is_file()


def test_irsim_reset_step_spaces(irsim_config: dict) -> None:
    env = make_env(irsim_config)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert info["env_id"] == "IRSimNav-v0"
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert next_obs.shape == env.observation_space.shape
        assert np.isfinite(reward)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert "termination_reason" in step_info
    finally:
        env.close()


def test_irsim_does_not_change_default_env_id() -> None:
    cfg = load_merged_configs(names=["environment"])
    assert cfg["environment"]["env_id"] == "RewardNavigation2D-v0"


def test_make_env_2d_still_default() -> None:
    cfg = load_merged_configs(names=["environment"])["environment"]
    env = make_env(cfg)
    try:
        obs, _ = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape
        assert cfg["env_id"] == "RewardNavigation2D-v0"
    finally:
        env.close()
