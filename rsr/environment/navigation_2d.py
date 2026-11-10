"""Gymnasium 2D navigation environment: RewardNavigation2D-v0."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rsr.environment.dynamics import (
    Circle,
    Workspace,
    clearance_to_circle,
    distance,
    integrate_velocity,
    is_valid_placement,
    nearest_obstacles,
    normalize_action,
    point_circle_collision,
    sample_point_in_workspace,
)

# Termination / truncation reason codes stored in info["termination_reason"]
REASON_SUCCESS = "success"
REASON_COLLISION = "collision"
REASON_OUT_OF_BOUNDS = "out_of_bounds"
REASON_TIMEOUT = "timeout"
REASON_NONE = "none"

RewardFn = Callable[["RewardNavigation2D", dict[str, Any]], float]


def default_reward_fn(env: "RewardNavigation2D", transition: dict[str, Any]) -> float:
    """
    Simple shaped reward for demos and environment tests.

    Full configurable multi-component rewards are implemented in Step 3.
    Components (kept numerically stable and finite):
      progress  : previous_dist - current_dist
      goal      : +10 on success
      collision : -10 on collision / out-of-bounds
      time      : -0.01 per step
    """
    progress = float(transition["prev_goal_distance"] - transition["goal_distance"])
    reward = 1.0 * progress - 0.01
    if transition["success"]:
        reward += 10.0
    if transition["collision"] or transition["out_of_bounds"]:
        reward -= 10.0
    if not np.isfinite(reward):
        raise RuntimeError(f"Non-finite default reward: {reward}")
    return float(reward)


class RewardNavigation2D(gym.Env):
    """
    Continuous 2D point-robot navigation among circular obstacles.

    Observation (normalized when configured):
      [robot_x, robot_y,
       vel_x, vel_y,
       rel_target_x, rel_target_y,
       dist_target,
       for each of n_nearest obstacles:
           rel_obs_x, rel_obs_y, dist_obs]
    Missing obstacle slots are filled with a far-away sentinel.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        *,
        reward_fn: Optional[RewardFn] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = self._validate_config(config or {})
        self.reward_fn: RewardFn = reward_fn or default_reward_fn
        self.render_mode = render_mode

        ws = self.config["workspace"]
        self.workspace = Workspace(
            width=float(ws["width"]),
            height=float(ws["height"]),
            centered=bool(ws.get("centered", True)),
        )
        self.robot_radius = float(self.config["robot"]["radius"])
        self.max_speed = float(self.config["robot"]["max_speed"])
        self.target_radius = float(self.config["target"]["radius"])
        self.n_nearest = int(self.config["obstacles"]["n_nearest"])
        self.max_steps = int(self.config["episode"]["max_steps"])
        goal_tol = self.config["episode"].get("goal_tolerance")
        self.goal_tolerance = (
            float(goal_tol)
            if goal_tol is not None
            else self.robot_radius + self.target_radius
        )

        obs_dim = 7 + 3 * self.n_nearest
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Episode state
        self._robot_pos = np.zeros(2, dtype=np.float64)
        self._robot_vel = np.zeros(2, dtype=np.float64)
        self._target = Circle(0.0, 0.0, self.target_radius)
        self._obstacles: list[Circle] = []
        self._step_count = 0
        self._path_length = 0.0
        self._min_obstacle_clearance = np.inf
        self._last_action = np.zeros(2, dtype=np.float64)
        self._trajectory: list[np.ndarray] = []
        self._np_random: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        options = options or {}
        self._step_count = 0
        self._path_length = 0.0
        self._min_obstacle_clearance = np.inf
        self._robot_vel = np.zeros(2, dtype=np.float64)
        self._last_action = np.zeros(2, dtype=np.float64)
        self._trajectory = []

        self._place_entities(options)
        self._trajectory.append(self._robot_pos.copy())
        self._update_min_clearance()

        obs = self._get_obs()
        info = self._base_info()
        info["termination_reason"] = REASON_NONE
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        action = normalize_action(np.asarray(action, dtype=np.float64))
        prev_pos = self._robot_pos.copy()
        prev_action = self._last_action.copy()
        prev_goal_dist = distance(self._robot_pos, self._target.center)

        new_pos, velocity = integrate_velocity(
            self._robot_pos,
            action,
            max_speed=self.max_speed,
            dt=1.0,
        )
        self._robot_pos = new_pos
        self._robot_vel = velocity
        self._last_action = action.copy()
        self._step_count += 1
        self._path_length += distance(prev_pos, self._robot_pos)
        self._trajectory.append(self._robot_pos.copy())
        self._update_min_clearance()

        goal_dist = distance(self._robot_pos, self._target.center)
        success = goal_dist <= self.goal_tolerance
        collision = self._check_collision()
        out_of_bounds = not self.workspace.contains(
            self._robot_pos, margin=self.robot_radius
        )
        timeout = self._step_count >= self.max_steps

        # Termination priority: success > collision > out_of_bounds; timeout truncates
        terminated = False
        truncated = False
        reason = REASON_NONE
        if success and self.config["termination"].get("success_on_goal", True):
            terminated = True
            reason = REASON_SUCCESS
        elif collision and self.config["termination"].get("collide_on_obstacle", True):
            terminated = True
            reason = REASON_COLLISION
        elif out_of_bounds and self.config["termination"].get(
            "fail_on_out_of_bounds", True
        ):
            terminated = True
            reason = REASON_OUT_OF_BOUNDS
        elif timeout:
            truncated = True
            reason = REASON_TIMEOUT

        transition = {
            "prev_position": prev_pos,
            "position": self._robot_pos.copy(),
            "velocity": self._robot_vel.copy(),
            "action": action.copy(),
            "prev_action": prev_action,
            "prev_goal_distance": prev_goal_dist,
            "goal_distance": goal_dist,
            "success": success and reason == REASON_SUCCESS,
            "collision": reason == REASON_COLLISION,
            "out_of_bounds": reason == REASON_OUT_OF_BOUNDS,
            "timeout": reason == REASON_TIMEOUT,
            "step_count": self._step_count,
            "obstacles": list(self._obstacles),
            "target": self._target,
            "robot_radius": self.robot_radius,
            "min_obstacle_clearance": self._current_min_clearance(),
            "n_humans_close": self._count_obstacles_within(1.0),
            "comfort_radius": 1.0,
            "nearest": nearest_obstacles(
                self._robot_pos,
                self._obstacles,
                n=self.n_nearest,
                point_radius=self.robot_radius,
            ),
        }
        reward = float(self.reward_fn(self, transition))
        if not np.isfinite(reward):
            raise RuntimeError(f"Non-finite reward from reward_fn: {reward}")

        obs = self._get_obs()
        info = self._base_info()
        info.update(
            {
                "termination_reason": reason,
                "success": reason == REASON_SUCCESS,
                "collision": reason == REASON_COLLISION,
                "out_of_bounds": reason == REASON_OUT_OF_BOUNDS,
                "timeout": reason == REASON_TIMEOUT,
                "goal_distance": goal_dist,
                "prev_goal_distance": prev_goal_dist,
                "progress": prev_goal_dist - goal_dist,
                "path_length": self._path_length,
                "min_obstacle_clearance": (
                    float(self._min_obstacle_clearance)
                    if np.isfinite(self._min_obstacle_clearance)
                    else None
                ),
                "reward": reward,
                "step_count": self._step_count,
            }
        )
        # Log per-component values when using WeightedRewardFunction (or compatible)
        last_components = getattr(self.reward_fn, "last_components", None)
        last_weighted = getattr(self.reward_fn, "last_weighted", None)
        if isinstance(last_components, dict):
            info["reward_components"] = dict(last_components)
        if isinstance(last_weighted, dict):
            info["reward_weighted"] = dict(last_weighted)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering helpers (state accessors used by visualization)
    # ------------------------------------------------------------------

    @property
    def robot_position(self) -> np.ndarray:
        return self._robot_pos.copy()

    @property
    def robot_velocity(self) -> np.ndarray:
        return self._robot_vel.copy()

    @property
    def target(self) -> Circle:
        return self._target

    @property
    def obstacles(self) -> list[Circle]:
        return list(self._obstacles)

    @property
    def trajectory(self) -> list[np.ndarray]:
        return [p.copy() for p in self._trajectory]

    def get_wrapper_attr(self, name: str) -> Any:  # pragma: no cover - gym helper
        return getattr(self, name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        required_top = [
            "workspace",
            "robot",
            "target",
            "obstacles",
            "episode",
            "observation",
            "termination",
            "randomization",
        ]
        # Allow partial configs by merging with safe defaults for tests
        defaults = _default_env_config()
        merged = _deep_merge(defaults, config)
        for key in required_top:
            if key not in merged:
                raise ValueError(f"Missing environment config section: {key}")
        if merged["robot"]["max_speed"] <= 0:
            raise ValueError("robot.max_speed must be positive")
        if merged["episode"]["max_steps"] < 1:
            raise ValueError("episode.max_steps must be >= 1")
        if merged["obstacles"]["n_nearest"] < 0:
            raise ValueError("obstacles.n_nearest must be >= 0")
        if merged["obstacles"]["min_count"] < 0:
            raise ValueError("obstacles.min_count must be >= 0")
        if merged["obstacles"]["max_count"] < merged["obstacles"]["min_count"]:
            raise ValueError("obstacles.max_count must be >= min_count")
        return merged

    def _place_entities(self, options: dict[str, Any]) -> None:
        rng = self._np_random
        obs_cfg = self.config["obstacles"]
        rand_cfg = self.config["randomization"]
        min_clearance = float(obs_cfg["min_clearance"])

        # Target
        if "target_position" in options:
            tpos = np.asarray(options["target_position"], dtype=np.float64).reshape(2)
        elif rand_cfg.get("randomize_target", True):
            tpos = sample_point_in_workspace(
                rng, self.workspace, margin=self.target_radius + min_clearance * 0.5
            )
        else:
            tpos = np.array([0.35 * self.workspace.width, 0.0], dtype=np.float64)
        self._target = Circle(float(tpos[0]), float(tpos[1]), self.target_radius)

        # Obstacles
        if "obstacles" in options:
            self._obstacles = [
                Circle(float(o[0]), float(o[1]), float(o[2]))
                for o in options["obstacles"]
            ]
        elif rand_cfg.get("randomize_obstacles", True):
            self._obstacles = self._sample_obstacles(rng, min_clearance)
        else:
            self._obstacles = [
                Circle(-1.5, 1.0, 0.6),
                Circle(1.5, -1.0, 0.5),
                Circle(0.0, 2.0, 0.4),
            ]

        # Robot start
        if "robot_position" in options:
            rpos = np.asarray(options["robot_position"], dtype=np.float64).reshape(2)
        elif rand_cfg.get("randomize_start", True):
            rpos = self._sample_robot_start(rng, min_clearance)
        else:
            rpos = np.array([-0.35 * self.workspace.width, 0.0], dtype=np.float64)
        self._robot_pos = rpos

    def _sample_obstacles(
        self, rng: np.random.Generator, min_clearance: float
    ) -> list[Circle]:
        cfg = self.config["obstacles"]
        n = int(rng.integers(cfg["min_count"], cfg["max_count"] + 1))
        obstacles: list[Circle] = []
        others_seed = [self._target]
        for _ in range(n):
            placed = False
            for _attempt in range(100):
                radius = float(rng.uniform(cfg["min_radius"], cfg["max_radius"]))
                center = sample_point_in_workspace(
                    rng, self.workspace, margin=radius
                )
                if is_valid_placement(
                    center,
                    radius,
                    workspace=self.workspace,
                    others=others_seed + obstacles,
                    min_clearance=min_clearance,
                ):
                    obstacles.append(
                        Circle(float(center[0]), float(center[1]), radius)
                    )
                    placed = True
                    break
            if not placed:
                # Fewer obstacles than requested is acceptable; do not fail silently
                # beyond exhausting attempts — caller still gets a valid scene.
                break
        return obstacles

    def _sample_robot_start(
        self, rng: np.random.Generator, min_clearance: float
    ) -> np.ndarray:
        others = [self._target, *self._obstacles]
        for _attempt in range(200):
            center = sample_point_in_workspace(
                rng, self.workspace, margin=self.robot_radius
            )
            if is_valid_placement(
                center,
                self.robot_radius,
                workspace=self.workspace,
                others=others,
                min_clearance=min_clearance,
            ):
                # Also require meaningful distance from target so episode is non-trivial
                if distance(center, self._target.center) > (
                    self.goal_tolerance + 2.0 * min_clearance
                ):
                    return center
        raise RuntimeError(
            "Failed to sample a valid robot start pose after 200 attempts. "
            "Relax obstacle density or clearance."
        )

    def _check_collision(self) -> bool:
        for obs in self._obstacles:
            if point_circle_collision(
                self._robot_pos, obs, point_radius=self.robot_radius
            ):
                return True
        return False

    def _current_min_clearance(self) -> float:
        if not self._obstacles:
            return float("inf")
        return min(
            clearance_to_circle(self._robot_pos, obs, point_radius=self.robot_radius)
            for obs in self._obstacles
        )

    def _count_obstacles_within(self, comfort_radius: float) -> int:
        if not self._obstacles:
            return 0
        count = 0
        for obs in self._obstacles:
            clearance = clearance_to_circle(
                self._robot_pos, obs, point_radius=self.robot_radius
            )
            if clearance < float(comfort_radius):
                count += 1
        return int(count)

    def _update_min_clearance(self) -> None:
        clearance = self._current_min_clearance()
        if clearance < self._min_obstacle_clearance:
            self._min_obstacle_clearance = clearance

    def _get_obs(self) -> np.ndarray:
        normalize = bool(self.config["observation"].get("normalize", True))
        half = self.workspace.half_extents
        scale_pos = half if normalize else np.ones(2, dtype=np.float64)
        scale_speed = self.max_speed if normalize else 1.0
        diag = self.workspace.diagonal
        scale_dist = diag if normalize else 1.0

        robot = self._robot_pos / scale_pos
        if self.config["observation"].get("include_velocity", True):
            vel = self._robot_vel / scale_speed
        else:
            vel = self._last_action.copy()  # already in [-1, 1]

        rel_target = (self._target.center - self._robot_pos) / scale_pos
        dist_target = distance(self._robot_pos, self._target.center) / scale_dist

        nearest = nearest_obstacles(
            self._robot_pos, self._obstacles, n=self.n_nearest
        )
        obs_feats: list[float] = []
        for i in range(self.n_nearest):
            if i < len(nearest):
                circle, dist_c = nearest[i]
                rel = (circle.center - self._robot_pos) / scale_pos
                dist_n = dist_c / scale_dist
            else:
                # Far-away sentinel: relative offset along +x, max distance
                rel = np.array([1.0, 0.0], dtype=np.float64)
                dist_n = 1.0 if normalize else diag
            obs_feats.extend([float(rel[0]), float(rel[1]), float(dist_n)])

        obs = np.array(
            [
                float(robot[0]),
                float(robot[1]),
                float(vel[0]),
                float(vel[1]),
                float(rel_target[0]),
                float(rel_target[1]),
                float(dist_target),
                *obs_feats,
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(obs)):
            raise RuntimeError(f"Non-finite observation: {obs}")
        return obs

    def _base_info(self) -> dict[str, Any]:
        return {
            "robot_position": self._robot_pos.copy(),
            "robot_velocity": self._robot_vel.copy(),
            "target_position": self._target.center.copy(),
            "n_obstacles": len(self._obstacles),
            "workspace": {
                "width": self.workspace.width,
                "height": self.workspace.height,
                "centered": self.workspace.centered,
            },
        }


def _default_env_config() -> dict[str, Any]:
    return {
        "env_id": "RewardNavigation2D-v0",
        "workspace": {"width": 10.0, "height": 10.0, "centered": True},
        "robot": {"radius": 0.15, "max_speed": 1.0},
        "target": {"radius": 0.25},
        "obstacles": {
            "min_count": 2,
            "max_count": 5,
            "min_radius": 0.3,
            "max_radius": 0.8,
            "n_nearest": 3,
            "min_clearance": 0.5,
        },
        "episode": {"max_steps": 200, "goal_tolerance": None},
        "observation": {"normalize": True, "include_velocity": True},
        "termination": {
            "success_on_goal": True,
            "collide_on_obstacle": True,
            "fail_on_out_of_bounds": True,
        },
        "randomization": {
            "randomize_start": True,
            "randomize_target": True,
            "randomize_obstacles": True,
        },
        "seed": 0,
    }


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def make_env(
    config: Optional[dict[str, Any]] = None,
    *,
    reward_fn: Optional[RewardFn] = None,
    render_mode: Optional[str] = None,
):
    """Factory for the configured navigation env.

    - ``CrowdNavPlusPlus-v0`` → CrowdNav++ adapter (``env/crowdnav``)
    - ``IRSimNav-v0`` → IR-SIM adapter (``env/ir-sim``)
    - anything else → RewardNavigation2D
    """
    env_id = str((config or {}).get("env_id", "RewardNavigation2D-v0"))
    if env_id == "CrowdNavPlusPlus-v0":
        from rsr.environment.crowdnav_env import CrowdNavPlusPlusEnv

        return CrowdNavPlusPlusEnv(
            config=config, reward_fn=reward_fn, render_mode=render_mode
        )
    if env_id == "IRSimNav-v0":
        from rsr.environment.irsim_env import IRSimNavEnv

        return IRSimNavEnv(
            config=config, reward_fn=reward_fn, render_mode=render_mode
        )
    return RewardNavigation2D(config=config, reward_fn=reward_fn, render_mode=render_mode)


# Gymnasium registration (idempotent)
def register_env() -> None:
    env_id = "RewardNavigation2D-v0"
    registry = gym.envs.registration.registry
    if env_id not in registry:
        gym.register(
            id=env_id,
            entry_point="rsr.environment.navigation_2d:RewardNavigation2D",
        )


register_env()
