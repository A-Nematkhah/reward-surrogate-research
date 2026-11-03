"""Fixed behavioral probe trajectories independent of candidate PPO policies.

Leakage restrictions:
- Probes come only from reward-independent reference policies.
- Never use a candidate's trained PPO rollouts.
- Probe generation does not use evaluation metrics as inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.environment.dynamics import Circle, clearance_to_circle, distance, normalize_action
from src.environment.navigation_2d import make_env
from src.utils.config import relpath_to_project
from src.utils.logging import get_logger
from src.utils.reproducibility import make_rng

logger = get_logger("probe_trajectories")

PolicyFn = Callable[[Any, np.ndarray, dict[str, Any], np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class ProbeEpisode:
    policy_name: str
    episode_index: int
    seed: int
    transitions: list[dict[str, Any]]
    termination_reason: str


def policy_random(
    env: Any, obs: np.ndarray, info: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    del obs, info
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float64)


def policy_goal_seeking(
    env: Any, obs: np.ndarray, info: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    del obs, rng
    robot = np.asarray(info["robot_position"], dtype=np.float64)
    target = np.asarray(info["target_position"], dtype=np.float64)
    delta = target - robot
    norm = float(np.linalg.norm(delta))
    if norm < 1e-8:
        return np.zeros(2, dtype=np.float64)
    return normalize_action(delta / norm)


def policy_obstacle_avoiding(
    env: Any, obs: np.ndarray, info: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Blend goal seeking with repulsion from the nearest obstacle."""
    del obs, rng
    robot = np.asarray(info["robot_position"], dtype=np.float64)
    target = np.asarray(info["target_position"], dtype=np.float64)
    goal_dir = target - robot
    goal_norm = float(np.linalg.norm(goal_dir))
    goal_u = goal_dir / goal_norm if goal_norm > 1e-8 else np.zeros(2)

    obstacles = getattr(env, "obstacles", [])
    robot_radius = float(getattr(env, "robot_radius", 0.15))
    avoid = np.zeros(2, dtype=np.float64)
    if obstacles:
        nearest = min(
            obstacles,
            key=lambda c: distance(robot, c.center),
        )
        clearance = clearance_to_circle(robot, nearest, point_radius=robot_radius)
        away = robot - nearest.center
        away_norm = float(np.linalg.norm(away))
        if away_norm > 1e-8:
            away_u = away / away_norm
            # Stronger repulsion when close (clearance under ~1.5)
            strength = float(np.clip(1.5 - clearance, 0.0, 1.5))
            avoid = away_u * strength

    action = goal_u + 1.25 * avoid
    return normalize_action(action)


POLICY_REGISTRY: dict[str, PolicyFn] = {
    "random": policy_random,
    "goal_seeking": policy_goal_seeking,
    "obstacle_avoiding": policy_obstacle_avoiding,
}


def _transition_from_step(
    *,
    prev_pos: np.ndarray,
    prev_action: np.ndarray,
    action: np.ndarray,
    env: Any,
    info: dict[str, Any],
    prev_goal_distance: float,
) -> dict[str, Any]:
    return {
        "prev_position": prev_pos.astype(np.float64),
        "position": np.asarray(info["robot_position"], dtype=np.float64),
        "velocity": np.asarray(info.get("robot_velocity", np.zeros(2)), dtype=np.float64),
        "action": np.asarray(action, dtype=np.float64),
        "prev_action": np.asarray(prev_action, dtype=np.float64),
        "prev_goal_distance": float(prev_goal_distance),
        "goal_distance": float(info["goal_distance"]),
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "out_of_bounds": bool(info.get("out_of_bounds", False)),
        "timeout": bool(info.get("timeout", False)),
        "step_count": int(info.get("step_count", 0)),
        "obstacles": [
            Circle(float(o.x), float(o.y), float(o.radius)) for o in env.obstacles
        ],
        "target": env.target,
        "robot_radius": float(env.robot_radius),
        "min_obstacle_clearance": info.get("min_obstacle_clearance"),
    }


def rollout_probe_episode(
    env_config: Mapping[str, Any],
    *,
    policy_name: str,
    seed: int,
    episode_index: int,
    max_steps: int,
) -> ProbeEpisode:
    if policy_name not in POLICY_REGISTRY:
        raise KeyError(f"Unknown probe policy '{policy_name}'")
    policy = POLICY_REGISTRY[policy_name]
    rng = make_rng(seed)

    # Zero reward during probe collection — we only need transitions.
    def _zero_reward(env, transition):  # noqa: ANN001
        del env, transition
        return 0.0

    env = make_env(dict(env_config), reward_fn=_zero_reward)
    obs, info = env.reset(seed=seed)
    transitions: list[dict[str, Any]] = []
    prev_action = np.zeros(2, dtype=np.float64)
    terminated = truncated = False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        prev_pos = np.asarray(info["robot_position"], dtype=np.float64).copy()
        prev_goal_distance = distance(prev_pos, env.target.center)
        action = policy(env, obs, info, rng)
        obs, _, terminated, truncated, info = env.step(action)
        transitions.append(
            _transition_from_step(
                prev_pos=prev_pos,
                prev_action=prev_action,
                action=action,
                env=env,
                info=info,
                prev_goal_distance=prev_goal_distance,
            )
        )
        prev_action = np.asarray(action, dtype=np.float64)
        steps += 1

    reason = str(info.get("termination_reason", "none"))
    env.close()
    return ProbeEpisode(
        policy_name=policy_name,
        episode_index=episode_index,
        seed=seed,
        transitions=transitions,
        termination_reason=reason,
    )


def generate_probe_set(
    env_config: Mapping[str, Any],
    probe_config: Mapping[str, Any],
) -> list[ProbeEpisode]:
    """Generate the fixed probe trajectory set from config."""
    seed0 = int(probe_config["seed"])
    n_per = int(probe_config["n_trajectories_per_policy"])
    max_steps = int(probe_config.get("max_steps", env_config.get("episode", {}).get("max_steps", 200)))
    policies = list(probe_config["policies"])

    episodes: list[ProbeEpisode] = []
    ep_idx = 0
    for p_idx, policy_name in enumerate(policies):
        for i in range(n_per):
            # Distinct, reproducible seeds independent of PPO training seeds
            seed = seed0 + 1000 * (p_idx + 1) + i
            episode = rollout_probe_episode(
                env_config,
                policy_name=policy_name,
                seed=seed,
                episode_index=ep_idx,
                max_steps=max_steps,
            )
            episodes.append(episode)
            ep_idx += 1

    logger.info(
        "Generated %d probe episodes (%d policies × %d)",
        len(episodes),
        len(policies),
        n_per,
    )
    return episodes


def _serialize_transition(transition: Mapping[str, Any]) -> dict[str, Any]:
    obstacles = []
    for obs in transition.get("obstacles") or []:
        if isinstance(obs, Circle):
            obstacles.append([float(obs.x), float(obs.y), float(obs.radius)])
        else:
            obstacles.append([float(obs[0]), float(obs[1]), float(obs[2])])
    target = transition.get("target")
    if isinstance(target, Circle):
        target_list = [float(target.x), float(target.y), float(target.radius)]
    elif target is None:
        target_list = None
    else:
        target_list = [float(target[0]), float(target[1]), float(target[2])]

    return {
        "prev_position": np.asarray(transition["prev_position"], dtype=np.float64).tolist(),
        "position": np.asarray(transition["position"], dtype=np.float64).tolist(),
        "velocity": np.asarray(transition["velocity"], dtype=np.float64).tolist(),
        "action": np.asarray(transition["action"], dtype=np.float64).tolist(),
        "prev_action": np.asarray(transition["prev_action"], dtype=np.float64).tolist(),
        "prev_goal_distance": float(transition["prev_goal_distance"]),
        "goal_distance": float(transition["goal_distance"]),
        "success": bool(transition["success"]),
        "collision": bool(transition["collision"]),
        "out_of_bounds": bool(transition["out_of_bounds"]),
        "timeout": bool(transition["timeout"]),
        "step_count": int(transition["step_count"]),
        "obstacles": obstacles,
        "target": target_list,
        "robot_radius": float(transition["robot_radius"]),
        "min_obstacle_clearance": transition.get("min_obstacle_clearance"),
    }


def _deserialize_transition(data: Mapping[str, Any]) -> dict[str, Any]:
    obstacles = [
        Circle(float(o[0]), float(o[1]), float(o[2])) for o in (data.get("obstacles") or [])
    ]
    target_raw = data.get("target")
    target = (
        Circle(float(target_raw[0]), float(target_raw[1]), float(target_raw[2]))
        if target_raw is not None
        else None
    )
    return {
        "prev_position": np.asarray(data["prev_position"], dtype=np.float64),
        "position": np.asarray(data["position"], dtype=np.float64),
        "velocity": np.asarray(data["velocity"], dtype=np.float64),
        "action": np.asarray(data["action"], dtype=np.float64),
        "prev_action": np.asarray(data["prev_action"], dtype=np.float64),
        "prev_goal_distance": float(data["prev_goal_distance"]),
        "goal_distance": float(data["goal_distance"]),
        "success": bool(data["success"]),
        "collision": bool(data["collision"]),
        "out_of_bounds": bool(data["out_of_bounds"]),
        "timeout": bool(data["timeout"]),
        "step_count": int(data["step_count"]),
        "obstacles": obstacles,
        "target": target,
        "robot_radius": float(data["robot_radius"]),
        "min_obstacle_clearance": data.get("min_obstacle_clearance"),
    }


def save_probe_set(
    episodes: Sequence[ProbeEpisode],
    *,
    npz_path: str | Path,
    config_path: str | Path,
    probe_config: Mapping[str, Any],
) -> tuple[Path, Path]:
    """
    Save probe trajectories.

    NPZ stores a JSON-serialized object array for portability; config JSON
    records generation settings and integrity metadata.
    """
    npz_path = Path(npz_path)
    config_path = Path(config_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for ep in episodes:
        payload.append(
            {
                "policy_name": ep.policy_name,
                "episode_index": ep.episode_index,
                "seed": ep.seed,
                "termination_reason": ep.termination_reason,
                "n_transitions": len(ep.transitions),
                "transitions": [_serialize_transition(t) for t in ep.transitions],
            }
        )

    # NPZ with object array of JSON strings (one episode per entry)
    encoded = np.array([json.dumps(item) for item in payload], dtype=object)
    np.savez_compressed(npz_path, episodes_json=encoded)

    meta = {
        "probe_config": dict(probe_config),
        "n_episodes": len(episodes),
        "n_transitions_total": int(sum(len(ep.transitions) for ep in episodes)),
        "policies": sorted({ep.policy_name for ep in episodes}),
        "seeds": [ep.seed for ep in episodes],
        "leakage_notes": [
            "Probes generated by reward-independent reference policies only.",
            "No candidate PPO trajectories were used.",
            "No evaluation metrics were used as probe inputs.",
        ],
        "npz_path": relpath_to_project(npz_path),
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    logger.info("Wrote probe set: %s (%d episodes)", npz_path, len(episodes))
    return npz_path, config_path


def load_probe_set(npz_path: str | Path) -> list[ProbeEpisode]:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    encoded = data["episodes_json"]
    episodes: list[ProbeEpisode] = []
    for item in encoded:
        payload = json.loads(str(item))
        transitions = [_deserialize_transition(t) for t in payload["transitions"]]
        episodes.append(
            ProbeEpisode(
                policy_name=payload["policy_name"],
                episode_index=int(payload["episode_index"]),
                seed=int(payload["seed"]),
                transitions=transitions,
                termination_reason=str(payload["termination_reason"]),
            )
        )
    return episodes
