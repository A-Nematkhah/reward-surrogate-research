"""Tests for RewardNavigation2D-v0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.environment.dynamics import Circle, distance
from src.environment.navigation_2d import (
    REASON_COLLISION,
    REASON_OUT_OF_BOUNDS,
    REASON_SUCCESS,
    REASON_TIMEOUT,
    RewardNavigation2D,
    default_reward_fn,
    make_env,
)
from src.environment.visualization import plot_trajectory
from src.utils.config import load_named_config


@pytest.fixture
def env_config(config_dir: Path) -> dict:
    return load_named_config("environment", config_dir=config_dir)


@pytest.fixture
def env(env_config: dict) -> RewardNavigation2D:
    return make_env(env_config)


def test_reset_and_observation_validity(env: RewardNavigation2D) -> None:
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    assert "robot_position" in info
    assert info["n_obstacles"] >= env.config["obstacles"]["min_count"]


def test_action_space_validity(env: RewardNavigation2D) -> None:
    env.reset(seed=1)
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))


def test_step_rejects_nonfinite_handled_via_clip(env: RewardNavigation2D) -> None:
    env.reset(seed=2)
    # Out-of-range actions are clipped, not rejected
    obs, reward, *_ = env.step(np.array([10.0, -10.0], dtype=np.float32))
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)


def test_collision_detection(env_config: dict) -> None:
    env = make_env(env_config)
    env.reset(
        seed=0,
        options={
            "robot_position": [0.0, 0.0],
            "target_position": [4.0, 0.0],
            # After +1.0 step, robot is at x=1.0 inside this obstacle.
            "obstacles": [[1.0, 0.0, 0.5]],
        },
    )
    _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
    assert terminated
    assert not truncated
    assert info["termination_reason"] == REASON_COLLISION
    assert info["collision"] is True


def test_goal_detection(env_config: dict) -> None:
    env = make_env(env_config)
    env.reset(
        seed=0,
        options={
            "robot_position": [0.0, 0.0],
            "target_position": [0.8, 0.0],
            "obstacles": [[3.0, 3.0, 0.4]],
        },
    )
    # Goal tolerance = 0.15 + 0.25 = 0.40; after +1.0 step, dist to target = 0.2
    _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
    assert terminated
    assert not truncated
    assert info["termination_reason"] == REASON_SUCCESS
    assert info["success"] is True


def test_workspace_boundaries(env_config: dict) -> None:
    env = make_env(env_config)
    # Place near left boundary and drive out
    env.reset(
        seed=0,
        options={
            "robot_position": [-4.7, 0.0],
            "target_position": [0.0, 0.0],
            "obstacles": [[2.0, 2.0, 0.4]],
        },
    )
    _, _, terminated, truncated, info = env.step(np.array([-1.0, 0.0]))
    assert terminated
    assert not truncated
    assert info["termination_reason"] == REASON_OUT_OF_BOUNDS
    assert info["out_of_bounds"] is True


def test_timeout_truncation(env_config: dict) -> None:
    cfg = dict(env_config)
    cfg = {
        **cfg,
        "episode": {**cfg["episode"], "max_steps": 5},
        "randomization": {
            "randomize_start": False,
            "randomize_target": False,
            "randomize_obstacles": False,
        },
    }
    env = make_env(cfg)
    env.reset(
        seed=0,
        options={
            "robot_position": [-3.0, -3.0],
            "target_position": [3.0, 3.0],
            "obstacles": [[0.0, 2.5, 0.4]],
        },
    )
    terminated = truncated = False
    info: dict = {}
    for _ in range(5):
        # Move slowly away from a short path to avoid accidental success
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
        if terminated or truncated:
            break
    assert truncated
    assert not terminated
    assert info["termination_reason"] == REASON_TIMEOUT


def test_reproducibility_with_fixed_seeds(env_config: dict) -> None:
    env_a = make_env(env_config)
    env_b = make_env(env_config)

    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)

    actions = [np.array([0.2, -0.3]), np.array([-0.5, 0.1]), np.array([0.0, 0.8])]
    for action in actions:
        oa, ra, ta, ua, ia = env_a.step(action)
        ob, rb, tb, ub, ib = env_b.step(action)
        np.testing.assert_allclose(oa, ob)
        assert ra == pytest.approx(rb)
        assert ta == tb and ua == ub
        assert ia["termination_reason"] == ib["termination_reason"]


def test_default_reward_components_progress_and_goal(env_config: dict) -> None:
    """Verify default reward uses progress / goal / collision / time terms."""
    env = make_env(env_config)
    env.reset(
        seed=0,
        options={
            "robot_position": [0.0, 0.0],
            "target_position": [2.0, 0.0],
            "obstacles": [[0.0, 3.0, 0.4]],
        },
    )
    prev_dist = distance(env.robot_position, env.target.center)
    _, reward, terminated, _, info = env.step(np.array([1.0, 0.0]))
    new_dist = info["goal_distance"]
    progress = prev_dist - new_dist
    expected = 1.0 * progress - 0.01
    if terminated and info["success"]:
        expected += 10.0
    assert reward == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert np.isfinite(reward)


def test_default_reward_collision_penalty(env_config: dict) -> None:
    env = make_env(env_config)
    env.reset(
        seed=0,
        options={
            "robot_position": [0.0, 0.0],
            "target_position": [4.0, 0.0],
            "obstacles": [[0.5, 0.0, 0.5]],
        },
    )
    prev_dist = distance(env.robot_position, env.target.center)
    _, reward, terminated, _, info = env.step(np.array([1.0, 0.0]))
    assert terminated and info["collision"]
    progress = prev_dist - info["goal_distance"]
    expected = 1.0 * progress - 0.01 - 10.0
    assert reward == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_observation_pads_missing_obstacles(env_config: dict) -> None:
    cfg = {
        **env_config,
        "obstacles": {**env_config["obstacles"], "n_nearest": 3},
    }
    env = make_env(cfg)
    env.reset(
        seed=0,
        options={
            "robot_position": [0.0, 0.0],
            "target_position": [2.0, 0.0],
            "obstacles": [[1.0, 1.0, 0.3]],  # only one obstacle
        },
    )
    obs = env._get_obs()
    assert obs.shape[0] == 7 + 3 * 3
    # Last obstacle slot uses sentinel distance ~1.0 when normalized
    assert obs[-1] == pytest.approx(1.0, abs=1e-5)


def test_gymnasium_registry(env_config: dict) -> None:
    import gymnasium as gym

    from src.environment.navigation_2d import register_env

    register_env()
    # entry_point construction may need config; use make_env for configured env
    env = make_env(env_config)
    assert isinstance(env, gym.Env)
    assert "RewardNavigation2D-v0" in gym.envs.registration.registry


def test_custom_reward_fn_injected(env_config: dict) -> None:
    calls: list[dict] = []

    def custom_fn(env, transition):
        calls.append(transition)
        return 1.23

    env = make_env(env_config, reward_fn=custom_fn)
    env.reset(
        seed=0,
        options={
            "robot_position": [0.0, 0.0],
            "target_position": [2.0, 0.0],
            "obstacles": [[3.0, 3.0, 0.4]],
        },
    )
    _, reward, *_ = env.step(np.array([0.1, 0.0]))
    assert reward == pytest.approx(1.23)
    assert len(calls) == 1
    assert "goal_distance" in calls[0]


def test_visualization_saves(tmp_path: Path, env_config: dict) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    env = make_env(env_config)
    env.reset(seed=0)
    env.step(np.array([0.5, 0.2]))
    out = tmp_path / "traj.png"
    fig = plot_trajectory(env, save_path=out, title="test")
    assert out.is_file()
    assert out.stat().st_size > 0
    matplotlib.pyplot.close(fig)


def test_default_reward_fn_direct() -> None:
    class _Dummy:
        pass

    transition = {
        "prev_goal_distance": 2.0,
        "goal_distance": 1.5,
        "success": False,
        "collision": False,
        "out_of_bounds": False,
    }
    r = default_reward_fn(_Dummy(), transition)  # type: ignore[arg-type]
    assert r == pytest.approx(0.5 - 0.01)
