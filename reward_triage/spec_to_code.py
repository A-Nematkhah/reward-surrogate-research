"""Convert parametric RewardSpec dicts into sandboxed candidate source."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Default component names used by the 2D parametric pipeline.
COMPONENT_NAMES: tuple[str, ...] = (
    "progress",
    "goal",
    "collision",
    "obstacle",
    "smoothness",
    "time",
)


def spec_to_code(spec: Mapping[str, Any]) -> str:
    """Emit ``compute_reward(transition)`` source for a parametric weighted reward.

    The generated code uses only numpy/math and transition dict fields so it
    runs inside the sandbox without importing ``rsr``.
    """
    weights = {name: float(spec.get("weights", {}).get(name, 0.0)) for name in COMPONENT_NAMES}
    scales_raw = spec.get("scales") or {}
    scales = {name: float(scales_raw.get(name, 1.0)) for name in COMPONENT_NAMES}
    cid = str(spec.get("candidate_id", "unknown"))
    fam = str(spec.get("family_id", "unknown"))

    # Embed weights/scales as literals.
    w_lit = ", ".join(f"'{k}': {weights[k]!r}" for k in COMPONENT_NAMES)
    s_lit = ", ".join(f"'{k}': {scales[k]!r}" for k in COMPONENT_NAMES)

    return f'''# Auto-generated from RewardSpec {cid} / family {fam}
import numpy as np

WEIGHTS = {{{w_lit}}}
SCALES = {{{s_lit}}}


def _clearance(transition):
    c = transition.get("min_obstacle_clearance", transition.get("clearance"))
    if c is not None:
        return float(c)
    pos = np.asarray(transition.get("position", [0.0, 0.0]), dtype=float)
    obstacles = transition.get("obstacles") or []
    robot_r = float(transition.get("robot_radius", 0.15))
    best = 1e9
    for obs in obstacles:
        if isinstance(obs, (list, tuple)) and len(obs) >= 3:
            ox, oy, orad = float(obs[0]), float(obs[1]), float(obs[2])
        else:
            continue
        dist = float(np.linalg.norm(pos - np.array([ox, oy]))) - orad - robot_r
        best = min(best, dist)
    return float(best if best < 1e8 else 0.0)


def _components(transition):
    prev_g = float(transition.get("prev_goal_distance", 0.0))
    goal = float(transition.get("goal_distance", 0.0))
    progress = prev_g - goal
    success = 1.0 if transition.get("success") else 0.0
    collision = -1.0 if transition.get("collision") else 0.0
    clearance = _clearance(transition)
    obstacle = float(np.tanh(clearance))
    action = np.asarray(transition.get("action", [0.0, 0.0]), dtype=float)
    prev_action = np.asarray(transition.get("prev_action", np.zeros_like(action)), dtype=float)
    smoothness = -float(np.linalg.norm(action - prev_action))
    time_pen = -1.0
    return {{
        "progress": float(progress),
        "goal": float(success),
        "collision": float(collision),
        "obstacle": float(obstacle),
        "smoothness": float(smoothness),
        "time": float(time_pen),
    }}


def compute_reward(transition):
    comps = _components(transition)
    weighted = {{}}
    total = 0.0
    for name in {list(COMPONENT_NAMES)!r}:
        term = WEIGHTS[name] * SCALES[name] * comps[name]
        weighted[name] = float(term)
        total += term
    return float(total), comps
'''


def specs_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build RewardSpec-like dicts from a candidates CSV row mapping."""
    out: list[dict[str, Any]] = []
    for row in rows:
        weights = {}
        scales = {}
        for name in COMPONENT_NAMES:
            if f"w_{name}" in row:
                weights[name] = float(row[f"w_{name}"])
            elif name in row:
                weights[name] = float(row[name])
            if f"s_{name}" in row:
                scales[name] = float(row[f"s_{name}"])
        out.append(
            {
                "candidate_id": str(row.get("candidate_id", row.get("id", "unknown"))),
                "family_id": str(row.get("family_id", row.get("family", "unknown"))),
                "weights": weights,
                "scales": scales,
            }
        )
    return out
