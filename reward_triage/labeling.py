"""PPO labeling for code rewards (local / LLM), including early features."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from reward_triage.code_reward import CodeRewardFunction
from reward_triage.store import SeedLabel
from rsr.features.early_behavioral import (
    EARLY_FEATURES_FILENAME,
    flatten_early_payload,
    load_early_payload,
)
from rsr.training.ppo_trainer import train_reward_fn_candidate_seed

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(path)
    return data


def prepare_ppo_config(
    ppo_config: Mapping[str, Any],
    *,
    total_timesteps: int | None = None,
    enable_early: bool = True,
) -> dict[str, Any]:
    """Copy PPO config; optionally shorten budget and clip early checkpoints."""
    cfg = dict(ppo_config)
    if total_timesteps is not None:
        cfg["total_timesteps"] = int(total_timesteps)
    pilot = os.environ.get("RSR_PILOT_TIMESTEPS")
    if pilot:
        cfg["total_timesteps"] = int(pilot)

    early = dict(cfg.get("early_behavioral") or {})
    budget = int(cfg["total_timesteps"])
    if not enable_early:
        early["enabled"] = False
    else:
        early["enabled"] = True
        raw = list(early.get("checkpoints") or [2000, 5000, 12000])
        ckpts = sorted({int(c) for c in raw if int(c) <= budget})
        if not ckpts:
            early["enabled"] = False
        else:
            early["checkpoints"] = ckpts
            early["timesteps"] = int(early.get("timesteps", ckpts[-1]))
            if int(early["timesteps"]) > budget:
                early["timesteps"] = ckpts[-1]
    cfg["early_behavioral"] = early
    return cfg


def label_code_candidate_seeds(
    *,
    code: str,
    candidate_id: str,
    seeds: Sequence[int],
    run_root: Path,
    env_config: Mapping[str, Any],
    ppo_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    family_id: str = "local",
    enable_early: bool = True,
) -> list[SeedLabel]:
    """Train+eval each seed; return SeedLabel rows (failures recorded, not raised)."""
    ppo_cfg = prepare_ppo_config(ppo_config, enable_early=enable_early)
    eval_cfg = dict(eval_config)
    if os.environ.get("RSR_PILOT_TIMESTEPS"):
        eval_cfg["n_eval_episodes"] = int(os.environ.get("RSR_PILOT_EVAL_EPISODES", "16"))

    def _factory() -> CodeRewardFunction:
        return CodeRewardFunction(code)

    labels: list[SeedLabel] = []
    for seed in seeds:
        run_dir = Path(run_root) / candidate_id / f"seed_{int(seed)}"
        try:
            status = train_reward_fn_candidate_seed(
                candidate_id=candidate_id,
                family_id=family_id,
                reward_factory=_factory,
                env_config=env_config,
                ppo_config=ppo_cfg,
                eval_config=eval_cfg,
                seed=int(seed),
                run_dir=run_dir,
                resume_if_complete=True,
            )
        except Exception as exc:  # noqa: BLE001
            labels.append(
                SeedLabel(
                    seed=int(seed),
                    success=0.0,
                    collision=0.0,
                    timeout=0.0,
                    return_mean=0.0,
                    eval_episodes=0,
                    train_steps=int(ppo_cfg["total_timesteps"]),
                    wall_time_s=0.0,
                    status="failed",
                    failure={"type": type(exc).__name__, "message": str(exc)},
                    run_dir=str(run_dir),
                )
            )
            continue

        metrics = dict(status.get("metrics") or {})
        early_feats: dict[str, float] | None = None
        early_path = run_dir / EARLY_FEATURES_FILENAME
        if early_path.is_file():
            early_feats = flatten_early_payload(load_early_payload(early_path))

        labels.append(
            SeedLabel(
                seed=int(seed),
                success=float(metrics.get("success_rate", 0.0)),
                collision=float(metrics.get("collision_rate", 0.0)),
                timeout=float(metrics.get("timeout_rate", 0.0)),
                return_mean=float(metrics.get("mean_evaluation_return", 0.0)),
                eval_episodes=int(metrics.get("n_episodes", 0) or 0),
                train_steps=int(status.get("total_timesteps", ppo_cfg["total_timesteps"])),
                wall_time_s=float(status.get("train_duration_sec", 0.0))
                + float(status.get("eval_duration_sec", 0.0)),
                status=str(status.get("training_status", "completed")),
                failure=status.get("failure"),
                early_features=early_feats,
                run_dir=str(run_dir),
            )
        )
    return labels


def load_default_configs(
    *,
    env_path: Path | None = None,
    ppo_path: Path | None = None,
    eval_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        _load_yaml(env_path or ROOT / "configs" / "environment.yaml"),
        _load_yaml(ppo_path or ROOT / "configs" / "ppo.yaml"),
        _load_yaml(eval_path or ROOT / "configs" / "evaluation.yaml"),
    )
