"""Tests for dataset building and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rsr.dataset.builder import (
    build_and_save_phase1_dataset,
    build_dataset_from_runs,
    status_to_record,
)
from rsr.dataset.validator import DatasetValidationError, validate_dataset


def _write_run(
    root: Path,
    candidate_id: str,
    family_id: str,
    seed: int,
    *,
    success_rate: float,
    status: str = "completed",
) -> None:
    run_dir = root / "runs" / candidate_id / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidate_id": candidate_id,
        "family_id": family_id,
        "training_seed": seed,
        "training_status": status,
        "total_timesteps": 1000,
        "train_duration_sec": 1.5,
        "eval_duration_sec": 0.2,
        "reward_weights": {
            "progress": 1.0,
            "goal": 10.0,
            "collision": -10.0,
            "obstacle": -1.0,
            "smoothness": -0.1,
            "time": -0.01,
        },
        "metrics": {
            "success_rate": success_rate,
            "collision_rate": 0.1,
            "out_of_bounds_rate": 0.0,
            "timeout_rate": 1.0 - success_rate - 0.1,
            "mean_episode_length": 50.0,
            "mean_path_length": 8.0,
            "mean_goal_time": 20.0,
            "mean_min_obstacle_distance": 0.5,
            "mean_evaluation_return": 5.0,
        }
        if status == "completed"
        else None,
        "failure": None
        if status == "completed"
        else {"type": "RuntimeError", "message": "boom"},
        "model_path": str(run_dir / "model.zip"),
        "device": "cpu",
    }
    if status == "completed":
        (run_dir / "model.zip").write_bytes(b"fake")
    with (run_dir / "status.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_build_dataset_includes_failures(tmp_path: Path) -> None:
    _write_run(tmp_path, "balanced_00", "balanced", 0, success_rate=0.8)
    _write_run(
        tmp_path, "near_zero_00", "near_zero", 0, success_rate=0.0, status="failed"
    )
    df = build_dataset_from_runs(tmp_path)
    assert len(df) == 2
    assert set(df["training_status"]) == {"completed", "failed"}


def test_validate_dataset_ok(tmp_path: Path) -> None:
    _write_run(tmp_path, "balanced_00", "balanced", 0, success_rate=0.9)
    _write_run(tmp_path, "near_zero_00", "near_zero", 0, success_rate=0.1)
    df = build_dataset_from_runs(tmp_path)
    report = validate_dataset(df, min_completed=2, min_families=2)
    assert report["ok"] is True
    assert report["stats"]["success_rate_span"] == pytest.approx(0.8)


def test_validate_empty_raises() -> None:
    with pytest.raises(DatasetValidationError):
        validate_dataset(pd.DataFrame())


def test_build_and_save(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    _write_run(tmp_path, "a", "balanced", 0, success_rate=0.5)
    _write_run(tmp_path, "b", "safety_heavy", 0, success_rate=0.2)
    df, paths = build_and_save_phase1_dataset(tmp_path, output_dir=tmp_path / "dataset")
    assert paths["csv"].is_file()
    assert paths["parquet"].is_file()
    assert len(df) == 2


def test_status_to_record_weights() -> None:
    record = status_to_record(
        {
            "candidate_id": "x",
            "family_id": "balanced",
            "training_seed": 0,
            "training_status": "completed",
            "reward_weights": {"progress": 1.2},
            "metrics": {"success_rate": 0.5},
        }
    )
    assert record["weight_progress"] == 1.2
    assert record["success_rate"] == 0.5
