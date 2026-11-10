"""Convert parametric RewardSpec dicts into sandboxed candidate source.

Generated ``compute_reward`` must be numerically identical to
``rsr.rewards.reward_function.WeightedRewardFunction`` for every name in
``COMPONENT_NAMES`` (same formulas and sign conventions as
``rsr.rewards.components``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Must match rsr.rewards.components.COMPONENT_NAMES exactly.
COMPONENT_NAMES: tuple[str, ...] = (
    "progress",
    "goal",
    "collision",
    "obstacle",
    "smoothness",
    "time",
    "energy",
    "risk",
    "potential",
    "nonlinear_progress",
    "distance",
    "human_proximity",
    "human_intrusion",
    "crowded_progress",
)


def specs_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build RewardSpec-like dicts from candidates CSV rows.

    Reads ``weight_<name>`` / ``scale_<name>`` (exported schema). Falls back to
    ``w_<name>`` / ``s_<name>`` for older tables.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        weights: dict[str, float] = {}
        scales: dict[str, float] = {}
        for name in COMPONENT_NAMES:
            w = 0.0
            for key in (f"weight_{name}", f"w_{name}", name):
                if key in row and row[key] is not None and str(row[key]) != "nan":
                    try:
                        w = float(row[key])
                        break
                    except (TypeError, ValueError):
                        continue
            weights[name] = w
            s = 1.0
            for key in (f"scale_{name}", f"s_{name}"):
                if key in row and row[key] is not None and str(row[key]) != "nan":
                    try:
                        s = float(row[key])
                        break
                    except (TypeError, ValueError):
                        continue
            scales[name] = s
        out.append(
            {
                "candidate_id": str(row.get("candidate_id", row.get("id", "unknown"))),
                "family_id": str(row.get("family_id", row.get("family", "unknown"))),
                "weights": weights,
                "scales": scales,
            }
        )
    return out


def spec_to_code(spec: Mapping[str, Any]) -> str:
    """Emit sandbox source numerically identical to WeightedRewardFunction."""
    weights = {name: float(spec.get("weights", {}).get(name, 0.0)) for name in COMPONENT_NAMES}
    scales_raw = spec.get("scales") or {}
    scales = {name: float(scales_raw.get(name, 1.0)) for name in COMPONENT_NAMES}
    cid = str(spec.get("candidate_id", "unknown"))
    fam = str(spec.get("family_id", "unknown"))

    w_lit = ", ".join(f"'{k}': {weights[k]!r}" for k in COMPONENT_NAMES)
    s_lit = ", ".join(f"'{k}': {scales[k]!r}" for k in COMPONENT_NAMES)
    names_lit = repr(list(COMPONENT_NAMES))

    # Inline exact component formulas from rsr.rewards.components.
    # Uses only whitelisted numpy helpers (see executor._NumpyProxy).
    return f'''# Auto-generated from RewardSpec {cid} / family {fam}
# Numerically identical to WeightedRewardFunction + default components.
WEIGHTS = {{{w_lit}}}
SCALES = {{{s_lit}}}
SAFE_DISTANCE = 1.0
RISK_RADIUS = 0.75
DEFAULT_COMFORT = 1.0


def _as_vec2(x):
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size < 2:
        a = np.zeros(2)
    return a[:2]


def _min_clearance(transition):
    raw = transition.get("min_obstacle_clearance", transition.get("clearance"))
    # Prefer recomputation from geometry when obstacles are present (matches components.py).
    obstacles = transition.get("obstacles") or []
    if obstacles:
        pos = _as_vec2(transition.get("position", [0.0, 0.0]))
        robot_r = float(transition.get("robot_radius", 0.0))
        best = None
        for obs in obstacles:
            if isinstance(obs, (list, tuple)) and len(obs) >= 3:
                ox, oy, orad = float(obs[0]), float(obs[1]), float(obs[2])
            else:
                continue
            dist = float(np.sqrt((pos[0] - ox) ** 2 + (pos[1] - oy) ** 2)) - orad - robot_r
            best = dist if best is None else min(best, dist)
        return best
    if raw is not None:
        return float(raw)
    return None


def _comfort_radius(transition):
    raw = transition.get("comfort_radius")
    if raw is None:
        return DEFAULT_COMFORT
    value = float(raw)
    if value <= 0:
        raise ValueError("comfort_radius must be positive")
    return value


def _components(transition):
    prev_d = float(transition["prev_goal_distance"])
    curr_d = float(transition["goal_distance"])
    progress = prev_d - curr_d
    goal = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    clearance = _min_clearance(transition)
    if clearance is None:
        obstacle = 0.0
        risk = 0.0
        human_proximity = 0.0
        human_intrusion = 0.0
        crowded_gate = 1.0
    else:
        obstacle = float(np.clip(1.0 - float(clearance) / SAFE_DISTANCE, 0.0, 1.0))
        intrusion_r = max(0.0, RISK_RADIUS - float(clearance))
        risk = float((intrusion_r / RISK_RADIUS) ** 2)
        radius = _comfort_radius(transition)
        human_proximity = float(np.clip(1.0 - float(clearance) / radius, 0.0, 1.0))
        hintr = max(0.0, radius - float(clearance))
        human_intrusion = float(np.clip((hintr / radius) ** 2, 0.0, 1.0))
        crowded_gate = float(np.clip(float(clearance) / radius, 0.0, 1.0))
    action = _as_vec2(transition.get("action", [0.0, 0.0]))
    prev_action = _as_vec2(transition.get("prev_action", [0.0, 0.0]))
    delta = action - prev_action
    smoothness = float(np.sqrt(float(delta[0] ** 2 + delta[1] ** 2)) / np.sqrt(2.0))
    time_c = 1.0
    energy = float((action[0] ** 2 + action[1] ** 2) / 2.0)
    potential = float(np.exp(-prev_d) - np.exp(-curr_d))
    nonlinear_progress = float(np.sign(progress) * (progress ** 2))
    distance = float(curr_d)
    crowded_progress = float(progress * crowded_gate)
    return {{
        "progress": float(progress),
        "goal": float(goal),
        "collision": float(collision),
        "obstacle": float(obstacle),
        "smoothness": float(smoothness),
        "time": float(time_c),
        "energy": float(energy),
        "risk": float(risk),
        "potential": float(potential),
        "nonlinear_progress": float(nonlinear_progress),
        "distance": float(distance),
        "human_proximity": float(human_proximity),
        "human_intrusion": float(human_intrusion),
        "crowded_progress": float(crowded_progress),
    }}


def compute_reward(transition):
    comps = _components(transition)
    weighted = {{}}
    total = 0.0
    for name in {names_lit}:
        term = WEIGHTS[name] * SCALES[name] * comps[name]
        weighted[name] = float(term)
        total += term
    return float(total), comps
'''
