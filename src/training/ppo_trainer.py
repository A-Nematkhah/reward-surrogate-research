"""Stable-Baselines3 PPO training for a single reward candidate × seed."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from src.environment.navigation_2d import make_env
from src.evaluation.evaluator import evaluate_policy
from src.features.early_behavioral import (
    collect_early_rollout_features,
    save_early_features,
)
from src.rewards.reward_function import RewardSpec, WeightedRewardFunction
from src.utils.config import relpath_to_project
from src.utils.logging import get_logger
from src.utils.reproducibility import seed_everything

logger = get_logger("ppo_trainer")

ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}


def _resolve_activation(name: str) -> type[nn.Module]:
    key = str(name).lower()
    if key not in ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation_fn '{name}'. Expected one of {sorted(ACTIVATION_MAP)}"
        )
    return ACTIVATION_MAP[key]


def make_vec_env(
    env_config: Mapping[str, Any],
    reward_spec: RewardSpec,
    *,
    n_envs: int,
    seed: int,
) -> VecEnv:
    """Create a DummyVecEnv with one WeightedRewardFunction copy per env."""

    def _make_thunk(rank: int) -> Callable[[], Any]:
        def _init() -> Any:
            reward_fn = WeightedRewardFunction.from_spec(reward_spec)
            env = make_env(dict(env_config), reward_fn=reward_fn)
            env.reset(seed=seed + rank)
            return Monitor(env)

        return _init

    env_fns = [_make_thunk(i) for i in range(n_envs)]
    vec = DummyVecEnv(env_fns)
    vec.seed(seed)
    return vec


def build_ppo(
    vec_env: VecEnv,
    ppo_config: Mapping[str, Any],
    *,
    seed: int,
) -> PPO:
    policy_cfg = ppo_config.get("policy", {})
    activation = _resolve_activation(policy_cfg.get("activation_fn", "tanh"))
    net_arch = policy_cfg.get("net_arch", {"pi": [64, 64], "vf": [64, 64]})

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(ppo_config["learning_rate"]),
        n_steps=int(ppo_config["n_steps"]),
        batch_size=int(ppo_config["batch_size"]),
        n_epochs=int(ppo_config["n_epochs"]),
        gamma=float(ppo_config["gamma"]),
        gae_lambda=float(ppo_config["gae_lambda"]),
        clip_range=float(ppo_config["clip_range"]),
        ent_coef=float(ppo_config["ent_coef"]),
        vf_coef=float(ppo_config["vf_coef"]),
        max_grad_norm=float(ppo_config["max_grad_norm"]),
        policy_kwargs={
            "net_arch": net_arch,
            "activation_fn": activation,
        },
        verbose=0,
        seed=int(seed),
        device=str(ppo_config.get("device", "auto")),
        tensorboard_log=None,
    )
    return model


def run_dir_for(
    experiments_root: str | Path,
    candidate_id: str,
    seed: int,
) -> Path:
    return Path(experiments_root) / "runs" / candidate_id / f"seed_{seed}"


def is_run_complete(
    run_dir: str | Path,
    *,
    expected_timesteps: int | None = None,
    require_early_features: bool = False,
) -> bool:
    """
    A run is complete when status.json reports success and the model exists.

    If ``expected_timesteps`` is provided, the saved budget must match so that
    smoke runs with a smaller budget are not treated as final results.
    """
    run_dir = Path(run_dir)
    status_path = run_dir / "status.json"
    model_path = run_dir / "model.zip"
    if not status_path.is_file() or not model_path.is_file():
        return False
    if require_early_features and not (run_dir / "early_behavioral_features.json").is_file():
        return False
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if status.get("training_status") != "completed":
        return False
    if expected_timesteps is not None:
        saved = status.get("total_timesteps")
        if saved is None or int(saved) != int(expected_timesteps):
            return False
    return True


def train_candidate_seed(
    reward_spec: RewardSpec,
    *,
    env_config: Mapping[str, Any],
    ppo_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    seed: int,
    run_dir: str | Path,
    resume_if_complete: bool = True,
) -> dict[str, Any]:
    """
    Train PPO for one (candidate, seed), evaluate, and persist artifacts.

    Failures are recorded in ``status.json`` and re-raised (not silently ignored).
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    expected_timesteps = int(ppo_config["total_timesteps"])
    early_cfg = dict(ppo_config.get("early_behavioral") or {})
    early_enabled = bool(early_cfg.get("enabled", False))
    if resume_if_complete and is_run_complete(
        run_path,
        expected_timesteps=expected_timesteps,
        require_early_features=early_enabled,
    ):
        logger.info(
            "Skipping completed run %s seed=%s (timesteps=%s)",
            reward_spec.candidate_id,
            seed,
            expected_timesteps,
        )
        with (run_path / "status.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    status: dict[str, Any] = {
        "candidate_id": reward_spec.candidate_id,
        "family_id": reward_spec.family_id,
        "training_seed": int(seed),
        "training_status": "running",
        "failure": None,
        "reward_weights": dict(reward_spec.weights),
        "total_timesteps": int(ppo_config["total_timesteps"]),
        "early_behavioral_enabled": early_enabled,
    }
    _write_json(run_path / "status.json", status)
    _write_json(run_path / "reward_spec.json", reward_spec.to_dict())
    _write_json(
        run_path / "run_config.json",
        {
            "environment": dict(env_config),
            "ppo": dict(ppo_config),
            "evaluation": dict(eval_config),
        },
    )

    vec_env: Optional[VecEnv] = None
    try:
        seed_meta = seed_everything(
            int(seed),
            deterministic_torch=bool(ppo_config.get("seed_torch_deterministic", True)),
        )
        _write_json(run_path / "seed_metadata.json", seed_meta)

        vec_env = make_vec_env(
            env_config,
            reward_spec,
            n_envs=int(ppo_config["n_envs"]),
            seed=int(seed),
        )
        model = build_ppo(vec_env, ppo_config, seed=int(seed))

        t0 = time.perf_counter()
        early_features: Optional[dict[str, float]] = None
        if early_enabled:
            early_steps = int(early_cfg.get("timesteps", 5000))
            if early_steps <= 0:
                raise ValueError("early_behavioral.timesteps must be positive")
            if early_steps > expected_timesteps:
                raise ValueError(
                    "early_behavioral.timesteps cannot exceed ppo.total_timesteps"
                )
            model.learn(
                total_timesteps=early_steps,
                progress_bar=False,
                log_interval=int(ppo_config.get("log_interval", 10)),
            )
            early_features = collect_early_rollout_features(
                model,
                env_config=env_config,
                reward_spec=reward_spec,
                n_episodes=int(early_cfg.get("n_episodes", 12)),
                seed_start=int(early_cfg.get("seed_offset", 50000)) + int(seed) * 1000,
                deterministic=bool(early_cfg.get("deterministic", False)),
            )
            early_features["early_warmup_timesteps"] = float(early_steps)
            save_early_features(run_path / "early_behavioral_features.json", early_features)
            remaining = expected_timesteps - early_steps
            if remaining > 0:
                model.learn(
                    total_timesteps=remaining,
                    progress_bar=False,
                    log_interval=int(ppo_config.get("log_interval", 10)),
                    reset_num_timesteps=False,
                )
        else:
            model.learn(
                total_timesteps=expected_timesteps,
                progress_bar=False,
                log_interval=int(ppo_config.get("log_interval", 10)),
            )
        train_duration = time.perf_counter() - t0

        model_path = run_path / "model.zip"
        model.save(str(model_path))

        eval_result = evaluate_policy(
            model,
            env_config=env_config,
            reward_spec=reward_spec,
            eval_config=eval_config,
            output_dir=run_path,
        )

        status.update(
            {
                "training_status": "completed",
                "failure": None,
                "train_duration_sec": float(train_duration),
                "eval_duration_sec": float(eval_result["eval_duration_sec"]),
                "model_path": relpath_to_project(model_path),
                "metrics": eval_result["metrics"],
                "device": str(model.device),
                "early_behavioral_path": (
                    relpath_to_project(run_path / "early_behavioral_features.json")
                    if early_features is not None
                    else None
                ),
            }
        )
        _write_json(run_path / "status.json", status)
        logger.info(
            "Completed %s seed=%s success_rate=%.3f (train %.1fs)",
            reward_spec.candidate_id,
            seed,
            eval_result["metrics"].get("success_rate", float("nan")),
            train_duration,
        )
        return status

    except Exception as exc:  # noqa: BLE001 — must record failures explicitly
        status.update(
            {
                "training_status": "failed",
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _write_json(run_path / "status.json", status)
        logger.exception(
            "Training failed for %s seed=%s: %s",
            reward_spec.candidate_id,
            seed,
            exc,
        )
        raise
    finally:
        if vec_env is not None:
            vec_env.close()


def load_trained_model(run_dir: str | Path) -> PPO:
    run_path = Path(run_dir)
    model_path = run_path / "model.zip"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    return PPO.load(str(model_path))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
