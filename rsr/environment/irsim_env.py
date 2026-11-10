"""Gymnasium adapter for IR-SIM (`IRSimNav-v0`).

Keeps the same reward / info contract as RewardNavigation2D and CrowdNav++
so Phase-1 PPO and feature extraction work unchanged. The IR-SIM backend
lives under ``env/ir-sim/`` and is isolated from other backends.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rsr.environment.dynamics import (
    Circle,
    Workspace,
    clearance_to_circle,
    distance,
    is_valid_placement,
    nearest_obstacles,
    point_circle_collision,
    sample_point_in_workspace,
)
from rsr.environment.navigation_2d import (
    REASON_COLLISION,
    REASON_NONE,
    REASON_OUT_OF_BOUNDS,
    REASON_SUCCESS,
    REASON_TIMEOUT,
    default_reward_fn,
)
from rsr.utils.config import PROJECT_ROOT

ENV_ID = "IRSimNav-v0"
IRSIM_ROOT = PROJECT_ROOT / "env" / "ir-sim"
# Kept outside the nested ir-sim clone so Git can track it.
DEFAULT_WORLD = PROJECT_ROOT / "env" / "worlds" / "irsim_pipeline_nav.yaml"

RewardFn = Callable[[Any, dict[str, Any]], float]
_BOOTED = False


def _boot_irsim() -> None:
    """Import irsim without putting ``env/ir-sim`` on ``sys.path``.

    Adding the clone root to ``sys.path`` can shadow project modules
    (e.g. ``tests``). Prefer an editable install: ``pip install -e env/ir-sim``.
    """
    global _BOOTED
    if _BOOTED:
        return
    if not IRSIM_ROOT.is_dir():
        raise FileNotFoundError(f"IR-SIM clone missing: {IRSIM_ROOT}")
    try:
        import irsim  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "irsim is not installed. Run: pip install -e env/ir-sim "
            "(do not rely on sys.path hacks; they collide with other backends)."
        ) from exc
    _BOOTED = True


def _as_xy(state: Any) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float64).reshape(-1)
    return arr[:2].copy()


def _resolve_world_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    candidate = (PROJECT_ROOT / path).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"IR-SIM world YAML missing: {raw}")


def _pick_float(
    irsim_cfg: dict[str, Any],
    key: str,
    *,
    fallback: Any,
    default: float,
) -> float:
    """Prefer explicit ``environment.irsim.*`` over shared 2D robot/target keys."""
    if key in irsim_cfg and irsim_cfg[key] is not None:
        return float(irsim_cfg[key])
    if fallback is not None:
        return float(fallback)
    return float(default)


class IRSimNavEnv(gym.Env):
    """IR-SIM dynamics with this project's weighted reward and info API."""

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
        _boot_irsim()
        import irsim

        # Quiet IR-SIM log noise during PPO rollouts.
        logging.getLogger("irsim").setLevel(logging.WARNING)

        self.config = dict(config or {})
        irsim_cfg = dict(self.config.get("irsim") or {})
        robot_cfg = dict(self.config.get("robot") or {})
        target_cfg = dict(self.config.get("target") or {})
        obs_cfg = dict(self.config.get("obstacles") or {})
        episode_cfg = dict(self.config.get("episode") or {})

        # ``environment.irsim.*`` wins over shared 2D keys so CrowdNav/2D YAML
        # values (e.g. robot.radius=0.15) cannot silently reshape IR-SIM.
        self.max_speed = _pick_float(
            irsim_cfg, "max_speed", fallback=robot_cfg.get("max_speed"), default=1.0
        )
        self.robot_radius = _pick_float(
            irsim_cfg, "robot_radius", fallback=None, default=0.2
        )
        self.target_radius = _pick_float(
            irsim_cfg, "goal_radius", fallback=None, default=0.25
        )
        self.n_nearest = int(
            irsim_cfg.get(
                "n_nearest",
                obs_cfg.get("n_nearest", 3),
            )
        )
        self.max_steps = int(
            irsim_cfg.get(
                "max_steps",
                episode_cfg.get("max_steps", 200),
            )
        )
        self.goal_tolerance = _pick_float(
            irsim_cfg,
            "goal_tolerance",
            fallback=None,
            default=self.robot_radius + self.target_radius,
        )
        self.comfort_radius = _pick_float(
            irsim_cfg, "comfort_radius", fallback=None, default=1.0
        )
        self.min_clearance = _pick_float(
            irsim_cfg,
            "min_clearance",
            fallback=obs_cfg.get("min_clearance"),
            default=0.5,
        )
        world_path = _resolve_world_path(
            irsim_cfg.get("world_yaml")
            or self.config.get("world_yaml")
            or DEFAULT_WORLD
        )

        width = float((self.config.get("workspace") or {}).get("width", 10.0))
        height = float((self.config.get("workspace") or {}).get("height", 10.0))
        # IR-SIM world uses origin at corner [0,0]; never inherit 2D centered=True.
        self.workspace = Workspace(
            width=width,
            height=height,
            centered=False,
        )

        self.reward_fn = reward_fn or default_reward_fn
        self._np_random = np.random.default_rng(
            int(self.config.get("seed", 0) or 0)
        )

        self._sim = irsim.make(str(world_path), display=False, headless=True)
        # Trust the loaded world geometry for collision/reward consistency.
        sim_radius = getattr(self._sim.robot, "radius", None)
        if sim_radius is not None:
            self.robot_radius = float(sim_radius)
        try:
            self._sim.robot.goal_threshold = float(self.goal_tolerance)
        except Exception:
            pass
        self._obstacle_objs = [
            obj for obj in self._sim.objects if getattr(obj, "role", None) != "robot"
        ]
        if not self._obstacle_objs:
            raise RuntimeError("IR-SIM world has no obstacles; expected static circles.")

        # obs: robot(2) + vel(2) + rel_goal(2) + goal_dist(1) + n_nearest*(rel2+dist)
        self._obs_width = 7 + self.n_nearest * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_width,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._prev_action = np.zeros(2, dtype=np.float64)
        self._robot_vel = np.zeros(2, dtype=np.float64)
        self._step_count = 0
        self._path_length = 0.0
        self._min_clearance_seen = np.inf
        self._prev_goal_distance = 0.0
        self._target = Circle(8.0, 8.0, self.target_radius)
        self._obstacles: list[Circle] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        if seed is not None:
            self._np_random = np.random.default_rng(int(seed))
            try:
                self._sim.set_random_seed(int(seed))
            except Exception:
                pass

        self._sim.reset(random=False)
        self._place_entities(options)
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._robot_vel = np.zeros(2, dtype=np.float64)
        self._step_count = 0
        self._path_length = 0.0
        self._min_clearance_seen = np.inf
        self._prev_goal_distance = self._goal_distance()
        self._update_min_clearance()
        info = self._info(REASON_NONE, 0.0)
        return self._get_obs(), info

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float64).reshape(2)
        action_arr = np.clip(action_arr, -1.0, 1.0)
        prev_pos = self._position()
        prev_goal = self._goal_distance()
        prev_action = self._prev_action.copy()

        velocity = action_arr * self.max_speed
        self._sim.step(velocity.astype(np.float64))
        self._robot_vel = velocity.copy()
        self._step_count += 1

        pos = self._position()
        self._path_length += float(np.linalg.norm(pos - prev_pos))
        self._sync_obstacle_circles()
        self._update_min_clearance()
        goal_dist = self._goal_distance()

        success = goal_dist <= self.goal_tolerance or bool(self._sim.robot.arrive)
        collision = bool(self._sim.robot.collision) or self._check_collision()
        out_of_bounds = not self.workspace.contains(pos, margin=self.robot_radius)
        timeout = self._step_count >= self.max_steps

        reason = REASON_NONE
        terminated = False
        truncated = False
        if success:
            terminated = True
            reason = REASON_SUCCESS
        elif collision:
            terminated = True
            reason = REASON_COLLISION
        elif out_of_bounds:
            terminated = True
            reason = REASON_OUT_OF_BOUNDS
        elif timeout:
            truncated = True
            reason = REASON_TIMEOUT

        transition = {
            "prev_position": prev_pos,
            "position": pos,
            "velocity": self._robot_vel.copy(),
            "action": action_arr.copy(),
            "prev_action": prev_action,
            "prev_goal_distance": prev_goal,
            "goal_distance": goal_dist,
            "success": reason == REASON_SUCCESS,
            "collision": reason == REASON_COLLISION,
            "out_of_bounds": reason == REASON_OUT_OF_BOUNDS,
            "timeout": reason == REASON_TIMEOUT,
            "step_count": self._step_count,
            "obstacles": list(self._obstacles),
            "target": self._target,
            "robot_radius": self.robot_radius,
            "min_obstacle_clearance": self._current_min_clearance(),
            "n_humans_close": self._count_obstacles_within(self.comfort_radius),
            "comfort_radius": self.comfort_radius,
            "nearest": nearest_obstacles(
                pos, self._obstacles, n=self.n_nearest, point_radius=self.robot_radius
            ),
        }
        reward = float(self.reward_fn(self, transition))
        self._prev_action = action_arr.copy()
        self._prev_goal_distance = goal_dist
        info = self._info(reason, reward)
        info["goal_distance"] = goal_dist
        info["prev_goal_distance"] = prev_goal
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        try:
            # Prefer silent cleanup; avoid EnvBase.end()'s timed wait when possible.
            if hasattr(self._sim, "quit_flag"):
                self._sim.quit_flag = True
        except Exception:
            pass
        return None

    @property
    def target(self) -> Circle:
        return self._target

    @property
    def obstacles(self) -> list[Circle]:
        return list(self._obstacles)

    @property
    def robot_position(self) -> np.ndarray:
        return self._position()

    def _place_entities(self, options: dict[str, Any]) -> None:
        rng = self._np_random
        margin = self.min_clearance

        if "target_position" in options:
            tpos = np.asarray(options["target_position"], dtype=np.float64).reshape(2)
        else:
            tpos = sample_point_in_workspace(
                rng, self.workspace, margin=self.target_radius + margin * 0.5
            )
        self._target = Circle(float(tpos[0]), float(tpos[1]), self.target_radius)
        self._sim.robot.set_goal([float(tpos[0]), float(tpos[1]), 0.0], init=True)

        placed: list[Circle] = [self._target]
        if "obstacles" in options:
            circles = [
                Circle(float(o[0]), float(o[1]), float(o[2]))
                for o in options["obstacles"]
            ]
            for obj, circle in zip(self._obstacle_objs, circles):
                obj.set_state([circle.x, circle.y, 0.0], init=True)
                placed.append(circle)
        else:
            for obj in self._obstacle_objs:
                radius = float(getattr(obj, "radius", 0.5))
                center = self._sample_valid_center(rng, radius, placed, margin)
                obj.set_state([float(center[0]), float(center[1]), 0.0], init=True)
                placed.append(Circle(float(center[0]), float(center[1]), radius))

        if "robot_position" in options:
            rpos = np.asarray(options["robot_position"], dtype=np.float64).reshape(2)
        else:
            rpos = self._sample_robot_start(rng, placed, margin)
        self._sim.robot.set_state([float(rpos[0]), float(rpos[1]), 0.0], init=True)
        self._sync_obstacle_circles()
        # Refresh geometry / collision tree after manual placement.
        if hasattr(self._sim, "refresh"):
            self._sim.refresh()

    def _sample_valid_center(
        self,
        rng: np.random.Generator,
        radius: float,
        others: list[Circle],
        min_clearance: float,
    ) -> np.ndarray:
        for _ in range(200):
            center = sample_point_in_workspace(rng, self.workspace, margin=radius)
            if is_valid_placement(
                center,
                radius,
                workspace=self.workspace,
                others=others,
                min_clearance=min_clearance,
            ):
                return center
        return sample_point_in_workspace(rng, self.workspace, margin=radius)

    def _sample_robot_start(
        self,
        rng: np.random.Generator,
        others: list[Circle],
        min_clearance: float,
    ) -> np.ndarray:
        for _ in range(200):
            center = sample_point_in_workspace(
                rng, self.workspace, margin=self.robot_radius
            )
            if not is_valid_placement(
                center,
                self.robot_radius,
                workspace=self.workspace,
                others=others,
                min_clearance=min_clearance,
            ):
                continue
            if distance(center, self._target.center) > (
                self.goal_tolerance + 2.0 * min_clearance
            ):
                return center
        return np.array([1.0, 1.0], dtype=np.float64)

    def _sync_obstacle_circles(self) -> None:
        circles: list[Circle] = []
        for obj in self._obstacle_objs:
            xy = _as_xy(obj.state)
            radius = float(getattr(obj, "radius", 0.5))
            circles.append(Circle(float(xy[0]), float(xy[1]), radius))
        self._obstacles = circles

    def _position(self) -> np.ndarray:
        return _as_xy(self._sim.robot.state)

    def _goal_distance(self) -> float:
        return float(distance(self._position(), self._target.center))

    def _check_collision(self) -> bool:
        pos = self._position()
        for obs in self._obstacles:
            if point_circle_collision(pos, obs, point_radius=self.robot_radius):
                return True
        return False

    def _current_min_clearance(self) -> float:
        if not self._obstacles:
            return float("inf")
        pos = self._position()
        return float(
            min(
                clearance_to_circle(pos, obs, point_radius=self.robot_radius)
                for obs in self._obstacles
            )
        )

    def _count_obstacles_within(self, comfort_radius: float) -> int:
        pos = self._position()
        count = 0
        for obs in self._obstacles:
            if (
                clearance_to_circle(pos, obs, point_radius=self.robot_radius)
                < float(comfort_radius)
            ):
                count += 1
        return int(count)

    def _update_min_clearance(self) -> None:
        clearance = self._current_min_clearance()
        if clearance < self._min_clearance_seen:
            self._min_clearance_seen = clearance

    def _get_obs(self) -> np.ndarray:
        half = np.array(
            [self.workspace.width / 2.0, self.workspace.height / 2.0], dtype=np.float64
        )
        # Map corner-origin coords into roughly [-1, 1] like centered 2D env.
        center = half
        scale_pos = half
        scale_speed = self.max_speed if self.max_speed > 0 else 1.0
        diag = float(np.linalg.norm([self.workspace.width, self.workspace.height]))
        scale_dist = diag if diag > 0 else 1.0

        pos = self._position()
        robot = (pos - center) / scale_pos
        vel = self._robot_vel / scale_speed
        rel_target = (self._target.center - pos) / scale_pos
        dist_target = distance(pos, self._target.center) / scale_dist

        nearest = nearest_obstacles(pos, self._obstacles, n=self.n_nearest)
        obs_feats: list[float] = [
            float(robot[0]),
            float(robot[1]),
            float(vel[0]),
            float(vel[1]),
            float(rel_target[0]),
            float(rel_target[1]),
            float(dist_target),
        ]
        for i in range(self.n_nearest):
            if i < len(nearest):
                circle, dist_c = nearest[i]
                rel = (circle.center - pos) / scale_pos
                dist_n = float(dist_c) / scale_dist
            else:
                rel = np.array([1.0, 0.0], dtype=np.float64)
                dist_n = 1.0
            obs_feats.extend([float(rel[0]), float(rel[1]), float(dist_n)])
        return np.asarray(obs_feats, dtype=np.float32)

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
                float(self._min_clearance_seen)
                if np.isfinite(self._min_clearance_seen)
                else None
            ),
            "n_humans_close": self._count_obstacles_within(self.comfort_radius),
            "comfort_radius": self.comfort_radius,
            "reward": reward,
            "step_count": self._step_count,
            "robot_position": self._position(),
            "target_position": self._target.center.copy(),
            "robot_velocity": self._robot_vel.copy(),
            "env_id": ENV_ID,
        }
        last_components = getattr(self.reward_fn, "last_components", None)
        last_weighted = getattr(self.reward_fn, "last_weighted", None)
        if isinstance(last_components, dict):
            info["reward_components"] = dict(last_components)
        if isinstance(last_weighted, dict):
            info["reward_weighted"] = dict(last_weighted)
        return info
