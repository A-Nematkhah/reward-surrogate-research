"""Early short-rollout behavioral features from a partially trained policy.

These features are computed after a cheap PPO warmup (not full training) and
are valid surrogate inputs: they do not use full-budget evaluation metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

from src.environment.navigation_2d import make_env
from src.rewards.reward_function import RewardSpec, WeightedRewardFunction
from src.utils.logging import get_logger

logger = get_logger("early_behavioral")

PolicyLike = Union[BaseAlgorithm, Any]


def collect_early_rollout_features(
    model: PolicyLike,
    *,
    env_config: Mapping[str, Any],
    reward_spec: RewardSpec,
    n_episodes: int,
    seed_start: int,
    deterministic: bool = False,
) -> dict[str, float]:
    """
    Roll out a partially trained policy and summarize cheap behavioral signals.

    Feature names are prefixed with ``early_`` so they cannot collide with
    full-budget evaluation metric column names used as Phase 2 targets.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")

    reward_fn = WeightedRewardFunction.from_spec(reward_spec)

    goal_deltas: list[float] = []
    returns: list[float] = []
    lengths: list[float] = []
    paths: list[float] = []
    clearances: list[float] = []
    n_success = n_collision = n_timeout = n_oob = 0

    for ep_idx in range(n_episodes):
        seed = int(seed_start) + ep_idx
        env = make_env(dict(env_config), reward_fn=reward_fn)
        obs, info = env.reset(seed=seed)
        start_goal_dist = float(info.get("goal_distance", np.nan))
        terminated = truncated = False
        episode_return = 0.0
        last_info = info

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, last_info = env.step(action)
            episode_return += float(reward)

        end_goal_dist = float(last_info.get("goal_distance", np.nan))
        if np.isfinite(start_goal_dist) and np.isfinite(end_goal_dist):
            goal_deltas.append(start_goal_dist - end_goal_dist)
        returns.append(episode_return)
        lengths.append(float(last_info.get("step_count", 0)))
        paths.append(float(last_info.get("path_length", 0.0)))
        clearance = last_info.get("min_obstacle_clearance")
        if clearance is not None and np.isfinite(clearance):
            clearances.append(float(clearance))

        if bool(last_info.get("success", False)):
            n_success += 1
        if bool(last_info.get("collision", False)):
            n_collision += 1
        if bool(last_info.get("timeout", False)):
            n_timeout += 1
        if bool(last_info.get("out_of_bounds", False)):
            n_oob += 1
        env.close()

    n = float(n_episodes)
    returns_arr = np.asarray(returns, dtype=np.float64)
    deltas_arr = np.asarray(goal_deltas, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.float64)
    paths_arr = np.asarray(paths, dtype=np.float64)

    features = {
        "early_timesteps_marker": 1.0,
        "early_n_episodes": n,
        "early_success_frac": n_success / n,
        "early_collision_frac": n_collision / n,
        "early_timeout_frac": n_timeout / n,
        "early_oob_frac": n_oob / n,
        "early_goal_progress_mean": float(np.mean(deltas_arr)) if deltas_arr.size else 0.0,
        "early_goal_progress_std": float(np.std(deltas_arr)) if deltas_arr.size else 0.0,
        "early_return_mean": float(np.mean(returns_arr)),
        "early_return_std": float(np.std(returns_arr)),
        "early_episode_length_mean": float(np.mean(lengths_arr)),
        "early_path_length_mean": float(np.mean(paths_arr)),
        "early_min_clearance_mean": (
            float(np.mean(clearances)) if clearances else 0.0
        ),
    }
    return features


def load_aggregated_early_features(
    experiments_root: str | Path,
    candidate_ids: Sequence[str],
    *,
    how: str = "mean",
) -> pd.DataFrame:
    """
    Load per-seed ``early_behavioral_features.json`` and aggregate per candidate.

    Row order matches ``candidate_ids``. Missing files raise FileNotFoundError.
    """
    if how not in {"mean", "median"}:
        raise ValueError(f"Unsupported aggregation: {how}")

    root = Path(experiments_root) / "runs"
    rows: list[dict[str, float]] = []
    for candidate_id in candidate_ids:
        cand_dir = root / str(candidate_id)
        if not cand_dir.is_dir():
            raise FileNotFoundError(f"Missing run directory for {candidate_id}: {cand_dir}")
        seed_feats: list[dict[str, float]] = []
        for seed_dir in sorted(cand_dir.glob("seed_*")):
            path = seed_dir / "early_behavioral_features.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing early behavioral features for {candidate_id}: {path}"
                )
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            seed_feats.append({k: float(v) for k, v in payload.items()})
        if not seed_feats:
            raise FileNotFoundError(f"No seed runs with early features for {candidate_id}")
        frame = pd.DataFrame(seed_feats)
        agg = frame.mean(numeric_only=True) if how == "mean" else frame.median(numeric_only=True)
        rows.append(agg.to_dict())

    out = pd.DataFrame(rows)
    out.reset_index(drop=True, inplace=True)
    return out


def save_early_features(path: str | Path, features: Mapping[str, float]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: (None if isinstance(v, float) and not np.isfinite(v) else float(v))
        for k, v in features.items()
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
    return path
