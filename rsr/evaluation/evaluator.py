"""Evaluate a trained policy on task metrics (not just return)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

import numpy as np

from rsr.environment.navigation_2d import RewardNavigation2D, make_env
from rsr.evaluation.metrics import (
    aggregate_episode_metrics,
    episodes_to_dataframe,
    metrics_to_serializable,
)
from rsr.rewards.reward_function import RewardSpec, WeightedRewardFunction
from rsr.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from stable_baselines3.common.base_class import BaseAlgorithm

logger = get_logger("evaluator")

PolicyLike = Any


def resolve_eval_seeds(eval_config: Mapping[str, Any]) -> list[int]:
    if eval_config.get("eval_seeds") is not None:
        return [int(s) for s in eval_config["eval_seeds"]]
    n = int(eval_config["n_eval_episodes"])
    start = int(eval_config.get("eval_seed_start", 10000))
    return list(range(start, start + n))


def evaluate_policy(
    model: PolicyLike,
    *,
    env_config: Mapping[str, Any],
    reward_spec: RewardSpec,
    eval_config: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Roll out ``model`` on a fixed set of evaluation seeds.

    Returns a dict with aggregate metrics, per-episode records, and timing.
    """
    seeds = resolve_eval_seeds(eval_config)
    deterministic = bool(eval_config.get("deterministic", True))
    reward_fn = WeightedRewardFunction.from_spec(reward_spec)

    t0 = time.perf_counter()
    episodes: list[dict[str, Any]] = []

    for ep_idx, seed in enumerate(seeds):
        env = make_env(dict(env_config), reward_fn=reward_fn)
        obs, info = env.reset(seed=int(seed))
        terminated = truncated = False
        episode_return = 0.0
        last_info = info

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, last_info = env.step(action)
            episode_return += float(reward)

        clearance = last_info.get("min_obstacle_clearance")
        episodes.append(
            {
                "episode_index": ep_idx,
                "eval_seed": int(seed),
                "termination_reason": last_info.get("termination_reason"),
                "success": bool(last_info.get("success", False)),
                "collision": bool(last_info.get("collision", False)),
                "out_of_bounds": bool(last_info.get("out_of_bounds", False)),
                "timeout": bool(last_info.get("timeout", False)),
                "episode_length": int(last_info.get("step_count", 0)),
                "path_length": float(last_info.get("path_length", 0.0)),
                "episode_return": float(episode_return),
                "min_obstacle_clearance": (
                    float(clearance) if clearance is not None else None
                ),
                "final_goal_distance": float(last_info.get("goal_distance", np.nan)),
                "n_obstacles": int(last_info.get("n_obstacles", 0)),
            }
        )
        env.close()

    eval_duration = time.perf_counter() - t0
    metrics = aggregate_episode_metrics(episodes)
    result = {
        "candidate_id": reward_spec.candidate_id,
        "family_id": reward_spec.family_id,
        "metrics": metrics_to_serializable(metrics),
        "episodes": episodes,
        "eval_duration_sec": float(eval_duration),
        "n_eval_episodes": len(episodes),
        "eval_seeds": seeds,
        "deterministic": deterministic,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics_name = eval_config.get("output", {}).get(
            "metrics_json", "evaluation_metrics.json"
        )
        details_name = eval_config.get("output", {}).get(
            "episode_details_csv", "episode_details.csv"
        )
        metrics_path = out / metrics_name
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "candidate_id": result["candidate_id"],
                    "family_id": result["family_id"],
                    "metrics": result["metrics"],
                    "eval_duration_sec": result["eval_duration_sec"],
                    "n_eval_episodes": result["n_eval_episodes"],
                    "eval_seeds": result["eval_seeds"],
                    "deterministic": result["deterministic"],
                },
                handle,
                indent=2,
            )
        if bool(eval_config.get("save_episode_details", True)):
            episodes_to_dataframe(episodes).to_csv(out / details_name, index=False)
        logger.info(
            "Evaluation %s: success=%.2f collision=%.2f timeout=%.2f (%.1fs)",
            reward_spec.candidate_id,
            metrics["success_rate"],
            metrics["collision_rate"],
            metrics["timeout_rate"],
            eval_duration,
        )

    return result
