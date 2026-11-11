"""Early short-rollout behavioral features from a partially trained policy.

These features are computed after cheap PPO warmups (not full training) and
are valid surrogate inputs: they do not use full-budget evaluation metrics.

Schema v2 supports multiple checkpoints (e.g. 2k / 5k / 12k) plus robust
rollout summaries (median / IQR / MAD) and cross-checkpoint deltas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from rsr.environment.navigation_2d import make_env
from rsr.rewards.reward_function import RewardSpec, WeightedRewardFunction
from rsr.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from stable_baselines3.common.base_class import BaseAlgorithm

logger = get_logger("early_behavioral")

PolicyLike = Any

EARLY_SCHEMA_VERSION = 2
EARLY_FEATURES_FILENAME = "early_behavioral_features.json"


def _finite_stats(arr: np.ndarray, *, prefix: str) -> dict[str, float]:
    """Mean/std plus robust summaries to reduce fragile-mean noise."""
    out: dict[str, float] = {
        f"{prefix}_mean": 0.0,
        f"{prefix}_std": 0.0,
        f"{prefix}_median": 0.0,
        f"{prefix}_iqr": 0.0,
        f"{prefix}_mad": 0.0,
        f"{prefix}_p10": 0.0,
        f"{prefix}_p90": 0.0,
        f"{prefix}_winsor_mean": 0.0,
    }
    if arr.size == 0:
        return out
    x = arr.astype(np.float64, copy=False)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return out
    med = float(np.median(x))
    q10, q25, q75, q90 = [float(v) for v in np.percentile(x, [10, 25, 75, 90])]
    mad = float(np.median(np.abs(x - med)))
    clipped = np.clip(x, q10, q90)
    out[f"{prefix}_mean"] = float(np.mean(x))
    out[f"{prefix}_std"] = float(np.std(x))
    out[f"{prefix}_median"] = med
    out[f"{prefix}_iqr"] = float(q75 - q25)
    out[f"{prefix}_mad"] = mad
    out[f"{prefix}_p10"] = q10
    out[f"{prefix}_p90"] = q90
    out[f"{prefix}_winsor_mean"] = float(np.mean(clipped))
    return out


def collect_early_rollout_features(
    model: PolicyLike,
    *,
    env_config: Mapping[str, Any],
    n_episodes: int,
    seed_start: int,
    reward_spec: RewardSpec | None = None,
    deterministic: bool = False,
    prefix: str = "early",
    include_stable_stats: bool = True,
    reward_fn: Any | None = None,
) -> dict[str, float]:
    """
    Roll out a partially trained policy and summarize cheap behavioral signals.

    Feature names are prefixed (default ``early_`` / ``early_t{steps}_``) so they
    cannot collide with full-budget evaluation metric column names.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")

    if reward_fn is None:
        if reward_spec is None:
            raise ValueError("collect_early_rollout_features requires reward_spec or reward_fn")
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
    clear_arr = np.asarray(clearances, dtype=np.float64)

    p = prefix.rstrip("_")
    features: dict[str, float] = {
        f"{p}_n_episodes": n,
        f"{p}_success_frac": n_success / n,
        f"{p}_collision_frac": n_collision / n,
        f"{p}_timeout_frac": n_timeout / n,
        f"{p}_oob_frac": n_oob / n,
    }

    if include_stable_stats:
        features.update(_finite_stats(deltas_arr, prefix=f"{p}_goal_progress"))
        features.update(_finite_stats(returns_arr, prefix=f"{p}_return"))
        features.update(_finite_stats(lengths_arr, prefix=f"{p}_episode_length"))
        features.update(_finite_stats(paths_arr, prefix=f"{p}_path_length"))
        features.update(_finite_stats(clear_arr, prefix=f"{p}_min_clearance"))
        # Keep legacy-friendly aliases used by older analyses / tests.
        features[f"{p}_goal_progress_mean"] = features[f"{p}_goal_progress_mean"]
        features[f"{p}_goal_progress_std"] = features[f"{p}_goal_progress_std"]
        features[f"{p}_return_mean"] = features[f"{p}_return_mean"]
        features[f"{p}_return_std"] = features[f"{p}_return_std"]
        features[f"{p}_episode_length_mean"] = features[f"{p}_episode_length_mean"]
        features[f"{p}_path_length_mean"] = features[f"{p}_path_length_mean"]
        features[f"{p}_min_clearance_mean"] = features[f"{p}_min_clearance_mean"]
    else:
        features[f"{p}_goal_progress_mean"] = (
            float(np.mean(deltas_arr)) if deltas_arr.size else 0.0
        )
        features[f"{p}_goal_progress_std"] = (
            float(np.std(deltas_arr)) if deltas_arr.size else 0.0
        )
        features[f"{p}_return_mean"] = float(np.mean(returns_arr))
        features[f"{p}_return_std"] = float(np.std(returns_arr))
        features[f"{p}_episode_length_mean"] = float(np.mean(lengths_arr))
        features[f"{p}_path_length_mean"] = float(np.mean(paths_arr))
        features[f"{p}_min_clearance_mean"] = (
            float(np.mean(clear_arr)) if clear_arr.size else 0.0
        )

    return features


def checkpoint_prefix(timesteps: int) -> str:
    return f"early_t{int(timesteps)}"


def derive_checkpoint_dynamics(
    by_checkpoint: Mapping[int, Mapping[str, float]],
) -> dict[str, float]:
    """Cross-checkpoint deltas / slopes for success and return median."""
    steps = sorted(int(s) for s in by_checkpoint)
    derived: dict[str, float] = {
        "early_n_checkpoints": float(len(steps)),
    }
    if len(steps) < 2:
        return derived

    def _get(step: int, key_suffix: str, default: float = 0.0) -> float:
        row = by_checkpoint[step]
        key = f"{checkpoint_prefix(step)}_{key_suffix}"
        val = row.get(key, default)
        return float(val) if val is not None and np.isfinite(val) else default

    for a, b in zip(steps[:-1], steps[1:]):
        tag = f"early_d{a}_{b}"
        derived[f"{tag}_success_frac"] = _get(b, "success_frac") - _get(a, "success_frac")
        derived[f"{tag}_return_median"] = _get(b, "return_median") - _get(
            a, "return_median"
        )
        derived[f"{tag}_goal_progress_median"] = _get(
            b, "goal_progress_median"
        ) - _get(a, "goal_progress_median")
        dt = float(b - a)
        if dt > 0:
            derived[f"{tag}_success_slope"] = derived[f"{tag}_success_frac"] / dt

    first, last = steps[0], steps[-1]
    span = float(last - first)
    derived["early_span_success_frac"] = _get(last, "success_frac") - _get(
        first, "success_frac"
    )
    derived["early_span_return_median"] = _get(last, "return_median") - _get(
        first, "return_median"
    )
    if span > 0:
        derived["early_span_success_slope"] = derived["early_span_success_frac"] / span
    return derived


def build_multi_checkpoint_payload(
    by_checkpoint: Mapping[int, Mapping[str, float]],
    *,
    primary_timesteps: int | None = None,
) -> dict[str, Any]:
    """Package per-checkpoint rollouts + dynamics into schema v2 JSON payload."""
    steps = sorted(int(s) for s in by_checkpoint)
    flat: dict[str, float] = {}
    for step in steps:
        flat.update({k: float(v) for k, v in by_checkpoint[step].items()})
    flat.update(derive_checkpoint_dynamics(by_checkpoint))

    primary = int(primary_timesteps) if primary_timesteps is not None else (
        steps[-1] if steps else 0
    )
    # Compatibility aliases (unscoped early_*) from the primary checkpoint.
    if primary in by_checkpoint:
        pfx = checkpoint_prefix(primary)
        for key, value in by_checkpoint[primary].items():
            if key.startswith(pfx + "_"):
                alias = "early_" + key[len(pfx) + 1 :]
                flat[alias] = float(value)
        flat["early_warmup_timesteps"] = float(primary)
        flat["early_timesteps_marker"] = 1.0

    return {
        "schema_version": EARLY_SCHEMA_VERSION,
        "checkpoints": steps,
        "primary_timesteps": primary,
        "features": flat,
    }


def flatten_early_payload(payload: Mapping[str, Any]) -> dict[str, float]:
    """Normalize schema v1 (flat) or v2 (nested) into a float feature dict."""
    if "features" in payload and isinstance(payload["features"], Mapping):
        return {
            str(k): float(v)
            for k, v in payload["features"].items()
            if v is not None and np.isfinite(float(v))
        }
    # Schema v1: already flat numeric map (ignore non-numeric meta if any)
    out: dict[str, float] = {}
    for k, v in payload.items():
        if k in {"schema_version", "checkpoints", "primary_timesteps", "features"}:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out[str(k)] = fv
    return out


def early_payload_matches(
    payload: Mapping[str, Any],
    *,
    expected_checkpoints: Sequence[int] | None,
    expected_schema: int = EARLY_SCHEMA_VERSION,
) -> bool:
    if int(payload.get("schema_version", 1)) < int(expected_schema):
        return False
    if expected_checkpoints is None:
        return True
    have = [int(x) for x in (payload.get("checkpoints") or [])]
    want = [int(x) for x in expected_checkpoints]
    return have == want


def resolve_early_checkpoints(early_cfg: Mapping[str, Any]) -> list[int]:
    """Return sorted unique checkpoint timesteps from config."""
    raw = early_cfg.get("checkpoints")
    if raw:
        steps = sorted({int(x) for x in raw})
    else:
        steps = [int(early_cfg.get("timesteps", 5000))]
    if any(s <= 0 for s in steps):
        raise ValueError("early_behavioral checkpoints must be positive")
    return steps


def load_aggregated_early_features(
    experiments_root: str | Path,
    candidate_ids: Sequence[str],
    *,
    how: str = "mean",
) -> pd.DataFrame:
    """
    Load per-seed early feature files and aggregate per candidate.

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
            path = seed_dir / EARLY_FEATURES_FILENAME
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing early behavioral features for {candidate_id}: {path}"
                )
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            seed_feats.append(flatten_early_payload(payload))
        if not seed_feats:
            raise FileNotFoundError(f"No seed runs with early features for {candidate_id}")
        frame = pd.DataFrame(seed_feats)
        agg = frame.mean(numeric_only=True) if how == "mean" else frame.median(numeric_only=True)
        rows.append(agg.to_dict())

    out = pd.DataFrame(rows)
    out.reset_index(drop=True, inplace=True)
    return out


def save_early_features(
    path: str | Path,
    features: Mapping[str, Any],
) -> Path:
    """Save schema v1 flat map or schema v2 payload to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            fv = float(obj)
            return None if not np.isfinite(fv) else fv
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    with path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize(features), handle, indent=2)
    return path


def load_early_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
