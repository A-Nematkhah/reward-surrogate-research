"""Visualization helpers for RewardNavigation2D."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Rectangle

from rsr.environment.dynamics import Circle, Workspace
from rsr.environment.navigation_2d import RewardNavigation2D


def draw_scene(
    ax: Axes,
    *,
    workspace: Workspace,
    robot_pos: np.ndarray,
    robot_radius: float,
    target: Circle,
    obstacles: Sequence[Circle],
    trajectory: Optional[Sequence[np.ndarray]] = None,
) -> None:
    """Draw workspace, obstacles, target, robot, and optional trajectory on ``ax``."""
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(workspace.x_min, workspace.x_max)
    ax.set_ylim(workspace.y_min, workspace.y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("RewardNavigation2D")

    # Workspace border
    ax.add_patch(
        Rectangle(
            (workspace.x_min, workspace.y_min),
            workspace.width,
            workspace.height,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
    )

    for obs in obstacles:
        ax.add_patch(
            MplCircle(
                (obs.x, obs.y),
                obs.radius,
                facecolor="#9aa5b1",
                edgecolor="#4b5563",
                alpha=0.85,
            )
        )

    ax.add_patch(
        MplCircle(
            (target.x, target.y),
            target.radius,
            facecolor="#3f8f5f",
            edgecolor="#1f5c38",
            alpha=0.9,
            label="target",
        )
    )

    if trajectory and len(trajectory) > 1:
        traj = np.asarray(trajectory, dtype=np.float64)
        ax.plot(traj[:, 0], traj[:, 1], color="#2563eb", linewidth=1.5, alpha=0.9)

    ax.add_patch(
        MplCircle(
            (float(robot_pos[0]), float(robot_pos[1])),
            robot_radius,
            facecolor="#c45c26",
            edgecolor="#7a3414",
            zorder=5,
            label="robot",
        )
    )
    ax.grid(True, linestyle=":", alpha=0.4)


def render_env(
    env: RewardNavigation2D,
    *,
    ax: Optional[Axes] = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Figure:
    """Render the current environment state; optionally save to disk."""
    fig: Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    draw_scene(
        ax,
        workspace=env.workspace,
        robot_pos=env.robot_position,
        robot_radius=env.robot_radius,
        target=env.target,
        obstacles=env.obstacles,
        trajectory=env.trajectory,
    )

    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    return fig


def plot_trajectory(
    env: RewardNavigation2D,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    title: Optional[str] = None,
) -> Figure:
    """Convenience wrapper that renders and optionally titles the figure."""
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_scene(
        ax,
        workspace=env.workspace,
        robot_pos=env.robot_position,
        robot_radius=env.robot_radius,
        target=env.target,
        obstacles=env.obstacles,
        trajectory=env.trajectory,
    )
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    return fig
