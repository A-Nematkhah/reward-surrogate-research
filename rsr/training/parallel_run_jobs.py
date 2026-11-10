"""Picklable job callables for parallel early-refresh / reeval (Windows-safe)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsr.evaluation.evaluator import evaluate_policy
from rsr.rewards.reward_function import RewardSpec
from rsr.training.ppo_trainer import (
    collect_early_features_only,
    load_trained_model,
    train_candidate_seed,
)
from rsr.utils.config import relpath_to_project


def phase1_train_job(job: dict[str, Any]) -> dict[str, Any]:
    """Train one Phase 1 (candidate, seed). Exceptions stay inside the worker."""
    try:
        reward_spec = RewardSpec.from_dict(job["reward_spec"])
        return train_candidate_seed(
            reward_spec,
            env_config=job["env_config"],
            ppo_config=job["ppo_config"],
            eval_config=job["eval_config"],
            seed=int(job["seed"]),
            run_dir=job["run_dir"],
            resume_if_complete=bool(job.get("resume", True)),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "candidate_id": job.get("candidate_id"),
            "family_id": job.get("family_id"),
            "training_seed": int(job["seed"]),
            "training_status": "failed",
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }


def refresh_early_job(job: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(job["run_dir"])
    try:
        reward_spec = RewardSpec.from_dict(job["reward_spec"])
        collect_early_features_only(
            env_config=job["env_config"],
            reward_spec=reward_spec,
            ppo_config=job["ppo_config"],
            seed=int(job["seed"]),
            run_path=run_dir,
        )
        return {
            "run_dir": str(run_dir),
            "candidate_id": reward_spec.candidate_id,
            "seed": int(job["seed"]),
            "ok": True,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "run_dir": str(run_dir),
            "candidate_id": job.get("candidate_id"),
            "seed": int(job["seed"]),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def reeval_run_job(job: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(job["run_dir"])
    try:
        status_path = run_dir / "status.json"
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        if status.get("training_status") != "completed":
            return {
                "run_dir": str(run_dir),
                "skipped": True,
                "ok": True,
                "reason": "not_completed",
            }
        if not (run_dir / "model.zip").is_file():
            return {
                "run_dir": str(run_dir),
                "skipped": True,
                "ok": True,
                "reason": "missing_model",
            }

        spec_path = run_dir / "reward_spec.json"
        if spec_path.is_file():
            with spec_path.open("r", encoding="utf-8") as handle:
                reward_spec = RewardSpec.from_dict(json.load(handle))
        else:
            reward_spec = RewardSpec(
                candidate_id=str(status["candidate_id"]),
                family_id=str(status.get("family_id", "unknown")),
                weights=dict(status.get("reward_weights") or {}),
                scales={k: 1.0 for k in (status.get("reward_weights") or {})},
            )

        model = load_trained_model(run_dir)
        eval_result = evaluate_policy(
            model,
            env_config=job["env_config"],
            reward_spec=reward_spec,
            eval_config=job["eval_config"],
            output_dir=run_dir,
        )
        status["metrics"] = eval_result["metrics"]
        status["eval_duration_sec"] = float(eval_result["eval_duration_sec"])
        status["n_eval_episodes"] = int(eval_result["n_eval_episodes"])
        status["reeval"] = {
            "n_eval_episodes": int(eval_result["n_eval_episodes"]),
            "eval_duration_sec": float(eval_result["eval_duration_sec"]),
            "model_path": relpath_to_project(run_dir / "model.zip"),
        }
        with status_path.open("w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2)
        return {
            "run_dir": relpath_to_project(run_dir),
            "candidate_id": reward_spec.candidate_id,
            "success_rate": eval_result["metrics"].get("success_rate"),
            "n_eval_episodes": eval_result["n_eval_episodes"],
            "skipped": False,
            "ok": True,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "run_dir": str(run_dir),
            "skipped": True,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "reason": f"{type(exc).__name__}: {exc}",
        }
