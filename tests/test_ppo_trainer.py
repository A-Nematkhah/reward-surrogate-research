"""Integration tests for PPO training and resume (small timestep budget)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from src.rewards.reward_function import RewardSpec
from src.training.experiment_runner import run_phase1
from src.training.ppo_trainer import (
    is_run_complete,
    load_trained_model,
    train_candidate_seed,
)
from src.utils.config import load_named_config


@pytest.fixture
def tiny_configs(config_dir: Path) -> tuple[dict, dict, dict]:
    env_config = load_named_config("environment", config_dir=config_dir)
    ppo_config = load_named_config("ppo", config_dir=config_dir)
    eval_config = load_named_config("evaluation", config_dir=config_dir)

    # Tiny budget for CI / local smoke tests
    early = dict(ppo_config.get("early_behavioral") or {})
    early.update(
        {
            "enabled": True,
            "timesteps": 256,
            "checkpoints": [128, 256],
            "n_episodes": 2,
            "stable_stats": True,
        }
    )
    ppo_config = {
        **ppo_config,
        "total_timesteps": 512,
        "n_envs": 2,
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 2,
        "n_seeds": 1,
        "seeds": [0],
        "device": "cpu",
        "early_behavioral": early,
    }
    eval_config = {
        **eval_config,
        "n_eval_episodes": 4,
        "eval_seed_start": 20000,
        "save_episode_details": True,
    }
    return env_config, ppo_config, eval_config


@pytest.fixture
def simple_spec() -> RewardSpec:
    return RewardSpec(
        candidate_id="test_balanced_00",
        family_id="balanced",
        weights={
            "progress": 1.5,
            "goal": 10.0,
            "collision": -10.0,
            "obstacle": -1.0,
            "smoothness": -0.1,
            "time": -0.01,
        },
        scales={
            "progress": 1.0,
            "goal": 1.0,
            "collision": 1.0,
            "obstacle": 1.0,
            "smoothness": 1.0,
            "time": 1.0,
        },
        generation_seed=0,
    )


def test_train_one_candidate_saves_checkpoint(
    tiny_configs: tuple[dict, dict, dict],
    simple_spec: RewardSpec,
    tmp_path: Path,
) -> None:
    env_config, ppo_config, eval_config = tiny_configs
    run_dir = tmp_path / "run"
    status = train_candidate_seed(
        simple_spec,
        env_config=env_config,
        ppo_config=ppo_config,
        eval_config=eval_config,
        seed=0,
        run_dir=run_dir,
        resume_if_complete=False,
    )
    assert status["training_status"] == "completed"
    assert (run_dir / "model.zip").is_file()
    assert (run_dir / "status.json").is_file()
    assert (run_dir / "evaluation_metrics.json").is_file()
    assert "success_rate" in status["metrics"]
    assert is_run_complete(run_dir)

    model = load_trained_model(run_dir)
    assert model is not None


def test_resume_skips_completed_run(
    tiny_configs: tuple[dict, dict, dict],
    simple_spec: RewardSpec,
    tmp_path: Path,
) -> None:
    env_config, ppo_config, eval_config = tiny_configs
    run_dir = tmp_path / "run"
    first = train_candidate_seed(
        simple_spec,
        env_config=env_config,
        ppo_config=ppo_config,
        eval_config=eval_config,
        seed=0,
        run_dir=run_dir,
    )
    second = train_candidate_seed(
        simple_spec,
        env_config=env_config,
        ppo_config=ppo_config,
        eval_config=eval_config,
        seed=0,
        run_dir=run_dir,
        resume_if_complete=True,
    )
    assert first["training_status"] == "completed"
    assert second["training_status"] == "completed"
    # Second call should return persisted status without rewriting failure
    assert second.get("metrics", {}).get("success_rate") == first["metrics"]["success_rate"]


def test_phase1_runner_max_jobs(
    tiny_configs: tuple[dict, dict, dict],
    simple_spec: RewardSpec,
    tmp_path: Path,
) -> None:
    env_config, ppo_config, eval_config = tiny_configs
    summary = run_phase1(
        candidates=[simple_spec],
        env_config=env_config,
        ppo_config=ppo_config,
        eval_config=eval_config,
        experiments_root=tmp_path / "phase1",
        max_jobs=1,
        resume=True,
        stop_on_failure=True,
    )
    assert summary["n_jobs"] == 1
    assert summary["n_failed"] == 0
    assert (tmp_path / "phase1" / "phase1_summary.json").is_file()
    with (tmp_path / "phase1" / "phase1_summary.json").open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["n_jobs"] == 1
