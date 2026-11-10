"""Gymnasium adapter that runs the project reward on CrowdNav++.

The inner simulator is the vendored CrowdNav++ ``CrowdSimVarNum`` env.
Humans use ORCA (``rvo2``).
The robot is still controlled by our PPO action, and the scalar reward is
``WeightedRewardFunction`` (the same components as RewardNavigation2D).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rsr.environment.dynamics import Circle
from rsr.environment.navigation_2d import (
    REASON_COLLISION,
    REASON_NONE,
    REASON_OUT_OF_BOUNDS,
    REASON_SUCCESS,
    REASON_TIMEOUT,
    default_reward_fn,
)
from rsr.utils.config import PROJECT_ROOT

CROWDNAV_ROOT = PROJECT_ROOT / "env" / "crowdnav"
ENV_ID = "CrowdNavPlusPlus-v0"

RewardFn = Callable[[Any, dict[str, Any]], float]
_BOOTED = False


def _boot_crowdnav() -> None:
    global _BOOTED
    if _BOOTED:
        return
    if not CROWDNAV_ROOT.is_dir():
        raise FileNotFoundError(f"CrowdNav++ clone missing: {CROWDNAV_ROOT}")
    root = str(CROWDNAV_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    # Config() parses argv at import time.
    sys.argv = [sys.argv[0]]
    import crowd_sim  # noqa: F401

    _BOOTED = True


def _flatten_obs(obs: dict[str, Any], width: int) -> np.ndarray:
    parts = [
        np.asarray(obs["robot_node"], dtype=np.float32).reshape(-1),
        np.asarray(obs["temporal_edges"], dtype=np.float32).reshape(-1),
        np.asarray(obs["spatial_edges"], dtype=np.float32).reshape(-1),
        np.asarray(obs["detected_human_num"], dtype=np.float32).reshape(-1),
        np.asarray(obs["visible_masks"], dtype=np.float32).reshape(-1),
    ]
    vec = np.concatenate(parts)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    if vec.size < width:
        vec = np.pad(vec, (0, width - vec.size))
    return vec[:width].astype(np.float32)


class CrowdNavPlusPlusEnv(gym.Env):
    """CrowdNav++ dynamics with this project's weighted reward and info API."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        *,
        reward_fn: Optional[RewardFn] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        del render_mode
        _boot_crowdnav()
        from crowd_nav.configs.config import Config
        from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum

        logging.getLogger().setLevel(logging.WARNING)
        self.config = dict(config or {})
        crowd_cfg = dict(self.config.get("crowdnav") or {})
        self.human_num = int(crowd_cfg.get("human_num", 5))
        self.max_steps = int(
            (self.config.get("episode") or {}).get("max_steps", 80)
        )
        self.reward_fn = reward_fn or default_reward_fn

        cn_config = Config()
        cn_config.sim.predict_method = "none"
        cn_config.env.use_wrapper = False
        cn_config.sim.human_num = self.human_num
        cn_config.sim.human_num_range = 0
        cn_config.sim.render = False
        cn_config.env.randomize_attributes = False
        cn_config.env.time_step = 0.25
        cn_config.env.time_limit = self.max_steps * cn_config.env.time_step
        cn_config.humans.policy = "orca"
        cn_config.humans.random_goal_changing = False
        cn_config.humans.end_goal_changing = False
        cn_config.robot.policy = "srnn"
        # When False, ORCA humans ignore the robot (no reactive avoidance of the agent).
        cn_config.robot.visible = bool(crowd_cfg.get("robot_visible", False))
        cn_config.action_space.kinematics = "holonomic"

        self._cn = CrowdSimVarNum()
        self._cn.configure(cn_config)
        self._cn.nenv = 1
        self._cn.phase = "train"
        self._cn.thisSeed = 0

        # robot(7) + vel(2) + humans*2 + count(1) + visibility
        self._obs_width = 7 + 2 + self.human_num * 2 + 1 + self.human_num
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_width,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._step_count = 0
        self._path_length = 0.0
        self._min_clearance = np.inf
        self._prev_goal_distance = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        del options
        if seed is not None:
            self._cn.thisSeed = int(seed)
        self._cn.nenv = 1
        self._cn.phase = "train"
        raw = self._cn.reset(phase="train")
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._step_count = 0
        self._path_length = 0.0
        self._min_clearance = np.inf
        self._prev_goal_distance = self._goal_distance()
        info = self._info(REASON_NONE, 0.0)
        return _flatten_obs(raw, self._obs_width), info

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(2)
        action_arr = np.clip(action_arr, -1.0, 1.0)
        prev_pos = self._position()
        prev_goal = self._goal_distance()
        _obs, _native_reward, done, native_info = self._cn.step(action_arr.copy())
        episode_info = native_info.get("info", native_info) if isinstance(native_info, dict) else native_info
        reason = _reason_from_info(episode_info, done=bool(done), steps=self._step_count + 1, limit=self.max_steps)

        pos = self._position()
        self._step_count += 1
        self._path_length += float(np.linalg.norm(pos - prev_pos))
        goal_dist = self._goal_distance()
        clearance = self._min_human_clearance()
        if clearance < self._min_clearance:
            self._min_clearance = clearance
        comfort = float((self.config.get("crowdnav") or {}).get("comfort_radius", 1.0))
        n_close = self._count_humans_within(comfort)

        success = reason == REASON_SUCCESS
        collision = reason == REASON_COLLISION
        out_of_bounds = reason == REASON_OUT_OF_BOUNDS
        timeout = reason == REASON_TIMEOUT
        terminated = success or collision or out_of_bounds
        truncated = timeout

        obstacles = [
            (float(h.px), float(h.py), float(h.radius))
            for h in self._cn.humans
        ]
        transition = {
            "prev_position": prev_pos,
            "position": pos,
            "velocity": np.array([self._cn.robot.vx, self._cn.robot.vy], dtype=np.float64),
            "action": action_arr.astype(np.float64),
            "prev_action": self._prev_action.copy(),
            "prev_goal_distance": prev_goal,
            "goal_distance": goal_dist,
            "success": success,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "timeout": timeout,
            "step_count": self._step_count,
            "obstacles": obstacles,
            "target": Circle(float(self._cn.robot.gx), float(self._cn.robot.gy), float(self._cn.robot.radius)),
            "robot_radius": float(self._cn.robot.radius),
            "min_obstacle_clearance": clearance,
            "n_humans_close": n_close,
            "comfort_radius": comfort,
        }
        reward = float(self.reward_fn(self, transition))
        self._prev_action = action_arr.astype(np.float64)
        self._prev_goal_distance = goal_dist
        info = self._info(reason, reward)
        info["goal_distance"] = goal_dist
        info["prev_goal_distance"] = prev_goal
        return _flatten_obs(_obs, self._obs_width), reward, terminated, truncated, info

    def close(self):
        return None

    @property
    def robot_radius(self) -> float:
        return float(self._cn.robot.radius)

    @property
    def target(self) -> Circle:
        return Circle(
            float(self._cn.robot.gx),
            float(self._cn.robot.gy),
            float(self._cn.robot.radius),
        )

    @property
    def obstacles(self) -> list[Circle]:
        return [
            Circle(float(h.px), float(h.py), float(h.radius))
            for h in self._cn.humans
        ]

    def _position(self) -> np.ndarray:
        return np.array([self._cn.robot.px, self._cn.robot.py], dtype=np.float64)

    def _goal_distance(self) -> float:
        return float(
            np.linalg.norm(
                self._position()
                - np.array([self._cn.robot.gx, self._cn.robot.gy], dtype=np.float64)
            )
        )

    def _min_human_clearance(self) -> float:
        robot = self._cn.robot
        if not self._cn.humans:
            return float("inf")
        clearances = []
        for human in self._cn.humans:
            dist = float(np.hypot(human.px - robot.px, human.py - robot.py))
            clearances.append(dist - float(human.radius) - float(robot.radius))
        return float(min(clearances))

    def _count_humans_within(self, comfort_radius: float) -> int:
        robot = self._cn.robot
        count = 0
        for human in self._cn.humans:
            dist = float(np.hypot(human.px - robot.px, human.py - robot.py))
            clearance = dist - float(human.radius) - float(robot.radius)
            if clearance < float(comfort_radius):
                count += 1
        return int(count)

    def _info(self, reason: str, reward: float) -> dict[str, Any]:
        info = {
            "termination_reason": reason,
            "success": reason == REASON_SUCCESS,
            "collision": reason == REASON_COLLISION,
            "out_of_bounds": reason == REASON_OUT_OF_BOUNDS,
            "timeout": reason == REASON_TIMEOUT,
            "goal_distance": self._goal_distance(),
            "prev_goal_distance": self._prev_goal_distance,
            "path_length": self._path_length,
            "min_obstacle_clearance": (
                float(self._min_clearance) if np.isfinite(self._min_clearance) else None
            ),
            "n_humans_close": self._count_humans_within(
                float((self.config.get("crowdnav") or {}).get("comfort_radius", 1.0))
            ),
            "comfort_radius": float(
                (self.config.get("crowdnav") or {}).get("comfort_radius", 1.0)
            ),
            "reward": reward,
            "step_count": self._step_count,
            "robot_position": self._position(),
            "target_position": np.array(
                [self._cn.robot.gx, self._cn.robot.gy], dtype=np.float64
            ),
            "robot_velocity": np.array(
                [self._cn.robot.vx, self._cn.robot.vy], dtype=np.float64
            ),
        }
        last_components = getattr(self.reward_fn, "last_components", None)
        last_weighted = getattr(self.reward_fn, "last_weighted", None)
        if isinstance(last_components, dict):
            info["reward_components"] = dict(last_components)
        if isinstance(last_weighted, dict):
            info["reward_weighted"] = dict(last_weighted)
        return info


def _reason_from_info(episode_info: Any, *, done: bool, steps: int, limit: int) -> str:
    name = type(episode_info).__name__
    if name == "ReachGoal":
        return REASON_SUCCESS
    if name == "Collision":
        return REASON_COLLISION
    if name == "OutRoad":
        return REASON_OUT_OF_BOUNDS
    if name == "Timeout" or steps >= limit:
        return REASON_TIMEOUT
    if done:
        return REASON_TIMEOUT
    return REASON_NONE
