"""2D navigation environment package."""

from rsr.environment.dynamics import Circle, Workspace
from rsr.environment.navigation_2d import (
    REASON_COLLISION,
    REASON_OUT_OF_BOUNDS,
    REASON_SUCCESS,
    REASON_TIMEOUT,
    RewardNavigation2D,
    default_reward_fn,
    make_env,
    register_env,
)
from rsr.environment.visualization import plot_trajectory, render_env

register_env()

__all__ = [
    "Circle",
    "REASON_COLLISION",
    "REASON_OUT_OF_BOUNDS",
    "REASON_SUCCESS",
    "REASON_TIMEOUT",
    "RewardNavigation2D",
    "Workspace",
    "default_reward_fn",
    "make_env",
    "plot_trajectory",
    "register_env",
    "render_env",
]
