"""Tests for individual reward components and weighted reward functions."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from rsr.environment.dynamics import Circle
from rsr.environment.navigation_2d import make_env
from rsr.rewards.components import (
    COMPONENT_NAMES,
    CollisionComponent,
    CrowdedProgressComponent,
    GoalComponent,
    HumanIntrusionComponent,
    HumanProximityComponent,
    ObstacleProximityComponent,
    ProgressComponent,
    SmoothnessComponent,
    TimeComponent,
    compute_all_components,
)
from rsr.rewards.reward_function import RewardSpec, WeightedRewardFunction
from rsr.utils.config import load_named_config


def _transition(**overrides):
    base = {
        "prev_position": np.array([0.0, 0.0]),
        "position": np.array([1.0, 0.0]),
        "velocity": np.array([1.0, 0.0]),
        "action": np.array([1.0, 0.0]),
        "prev_action": np.array([0.0, 0.0]),
        "prev_goal_distance": 2.0,
        "goal_distance": 1.0,
        "success": False,
        "collision": False,
        "out_of_bounds": False,
        "timeout": False,
        "step_count": 1,
        "obstacles": [Circle(3.0, 0.0, 0.5)],
        "target": Circle(2.0, 0.0, 0.25),
        "robot_radius": 0.15,
        "min_obstacle_clearance": 1.35,
    }
    base.update(overrides)
    return base


def test_progress_component() -> None:
    assert ProgressComponent()(_transition()) == pytest.approx(1.0)


def test_goal_and_collision_components() -> None:
    assert GoalComponent()(_transition(success=True)) == 1.0
    assert GoalComponent()(_transition(success=False)) == 0.0
    assert CollisionComponent()(_transition(collision=True)) == 1.0
    assert CollisionComponent()(_transition(out_of_bounds=True)) == 1.0
    assert CollisionComponent()(_transition()) == 0.0


def test_obstacle_proximity_scales() -> None:
    comp = ObstacleProximityComponent(safe_distance=1.0)
    # Far obstacle: clearance large -> 0
    far = _transition(
        position=np.array([0.0, 0.0]),
        obstacles=[Circle(5.0, 0.0, 0.5)],
        robot_radius=0.15,
    )
    assert comp(far) == pytest.approx(0.0)

    # Near / penetrating: clearance <= 0 -> 1
    near = _transition(
        position=np.array([0.0, 0.0]),
        obstacles=[Circle(0.4, 0.0, 0.5)],
        robot_radius=0.15,
    )
    # clearance = 0.4 - 0.5 - 0.15 = -0.25 -> value 1
    assert comp(near) == pytest.approx(1.0)


def test_smoothness_and_time() -> None:
    t = _transition(action=np.array([1.0, 0.0]), prev_action=np.array([-1.0, 0.0]))
    # ||delta|| = 2, /sqrt(2) = sqrt(2)
    assert SmoothnessComponent()(t) == pytest.approx(np.sqrt(2.0))
    assert TimeComponent()(t) == 1.0


def test_human_and_crowded_progress_components() -> None:
    far = _transition(
        position=np.array([0.0, 0.0]),
        obstacles=[Circle(5.0, 0.0, 0.5)],
        robot_radius=0.15,
        prev_goal_distance=2.0,
        goal_distance=1.0,
        comfort_radius=1.0,
        n_humans_close=0,
    )
    assert HumanProximityComponent()(far) == pytest.approx(0.0)
    assert HumanIntrusionComponent()(far) == pytest.approx(0.0)
    assert CrowdedProgressComponent()(far) == pytest.approx(1.0)

    close = _transition(
        position=np.array([0.0, 0.0]),
        obstacles=[Circle(0.5, 0.0, 0.2)],
        robot_radius=0.15,
        prev_goal_distance=2.0,
        goal_distance=1.0,
        comfort_radius=1.0,
        n_humans_close=2,
    )
    # clearance = 0.5 - 0.2 - 0.15 = 0.15 → gate = 0.15
    prox = HumanProximityComponent()(close)
    assert 0.0 < prox <= 1.0
    intrusion = HumanIntrusionComponent()(close)
    assert 0.0 < intrusion <= 1.0
    gated = CrowdedProgressComponent()(close)
    assert gated == pytest.approx(1.0 * 0.15)
    # n_humans_close must not apply a second discount
    close_no_count = dict(close)
    close_no_count["n_humans_close"] = None
    assert CrowdedProgressComponent()(close_no_count) == pytest.approx(gated)

    penetrating = _transition(
        position=np.array([0.0, 0.0]),
        obstacles=[Circle(0.2, 0.0, 0.3)],
        robot_radius=0.15,
        comfort_radius=1.0,
    )
    # clearance negative → intrusion clipped to 1
    assert HumanIntrusionComponent()(penetrating) == pytest.approx(1.0)
    # transition comfort_radius overrides constructor default
    wide = dict(close)
    wide["comfort_radius"] = 2.0
    # Larger comfort → same clearance is a deeper relative intrusion → higher proximity
    assert HumanProximityComponent(comfort_radius=1.0)(wide) > prox
    assert HumanProximityComponent(comfort_radius=1.0)(wide) == pytest.approx(
        1.0 - 0.15 / 2.0
    )


def test_compute_all_components_finite() -> None:
    values = compute_all_components(_transition())
    assert set(values) == set(COMPONENT_NAMES)
    assert all(np.isfinite(v) for v in values.values())


def test_components_no_future_fields_required() -> None:
    """Components must not require future / privileged keys beyond the transition."""
    t = _transition()
    # Remove anything that looks like future info if present
    t.pop("future", None)
    values = compute_all_components(t)
    assert np.isfinite(sum(values.values()))


def test_weighted_reward_matches_manual_sum() -> None:
    weights = {
        "progress": 2.0,
        "goal": 10.0,
        "collision": -10.0,
        "obstacle": -1.0,
        "smoothness": -0.5,
        "time": -0.01,
    }
    rf = WeightedRewardFunction(weights, candidate_id="t", family_id="test")
    transition = _transition(success=True)
    total = rf(None, transition)
    comps = rf.last_components
    expected = sum(weights.get(n, 0.0) * comps[n] for n in COMPONENT_NAMES)
    assert total == pytest.approx(expected)
    assert rf.last_total == pytest.approx(expected)
    assert "progress" in rf.last_weighted


def test_reward_spec_roundtrip() -> None:
    spec = RewardSpec(
        candidate_id="balanced_00",
        family_id="balanced",
        weights={n: 1.0 for n in COMPONENT_NAMES},
        scales={n: 1.0 for n in COMPONENT_NAMES},
        generation_seed=42,
        metadata={"x": 1},
    )
    restored = RewardSpec.from_dict(spec.to_dict())
    assert restored.candidate_id == spec.candidate_id
    assert restored.weights == spec.weights
    rf = WeightedRewardFunction.from_spec(restored)
    assert rf.candidate_id == "balanced_00"


def test_env_logs_reward_components(config_dir) -> None:
    env_config = load_named_config("environment", config_dir=config_dir)
    weights = {
        "progress": 1.0,
        "goal": 10.0,
        "collision": -10.0,
        "obstacle": -1.0,
        "smoothness": -0.1,
        "time": -0.01,
    }
    rf = WeightedRewardFunction(weights, candidate_id="demo", family_id="balanced")
    env = make_env(env_config, reward_fn=rf)
    env.reset(
        seed=0,
        options={
            "robot_position": [0.0, 0.0],
            "target_position": [2.0, 0.0],
            "obstacles": [[0.0, 3.0, 0.4]],
        },
    )
    _, reward, _, _, info = env.step(np.array([1.0, 0.0]))
    assert "reward_components" in info
    assert "reward_weighted" in info
    assert set(info["reward_components"]) == set(COMPONENT_NAMES)
    assert reward == pytest.approx(info["reward"])
    assert np.isfinite(reward)


def test_invalid_scale_rejected() -> None:
    with pytest.raises(ValueError):
        WeightedRewardFunction({"progress": 1.0}, scales={"progress": 0.0})
