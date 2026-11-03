"""Reward components, weighted reward functions, and candidate generation."""

from src.rewards.candidate_generator import (
    generate_candidates,
    load_candidates,
    make_reward_fn,
    save_candidates,
    summarize_candidates,
)
from src.rewards.components import (
    COMPONENT_NAMES,
    CollisionComponent,
    GoalComponent,
    ObstacleProximityComponent,
    ProgressComponent,
    SmoothnessComponent,
    TimeComponent,
    build_default_components,
    compute_all_components,
)
from src.rewards.reward_function import RewardSpec, WeightedRewardFunction

__all__ = [
    "COMPONENT_NAMES",
    "CollisionComponent",
    "GoalComponent",
    "ObstacleProximityComponent",
    "ProgressComponent",
    "RewardSpec",
    "SmoothnessComponent",
    "TimeComponent",
    "WeightedRewardFunction",
    "build_default_components",
    "compute_all_components",
    "generate_candidates",
    "load_candidates",
    "make_reward_fn",
    "save_candidates",
    "summarize_candidates",
]
