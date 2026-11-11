"""Hand-written 2D reward corpus for Phase 4 without an LLM API key.

Mirrors live-LLM intent: every template tries to be a *good* reward for the
task (no deliberately weak / near-zero families). Diversity is in shaping
style (safety, potential, smooth, …), like temperature/prompt variation.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

# Placeholders __A__..__F__ are replaced with floats (avoids brace conflicts).
TEMPLATES: list[tuple[str, str]] = [
    (
        "balanced",
        """
def compute_reward(transition):
    progress = float(transition.get("progress", 0.0))
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    clearance = float(transition.get("clearance", transition.get("min_obstacle_clearance", 1.0)) or 0.0)
    obstacle = max(0.0, 1.0 - clearance / 0.5)
    ad = transition.get("action_delta", 0.0)
    try:
        smooth = abs(float(ad))
    except (TypeError, ValueError):
        try:
            smooth = float(sum(abs(float(x)) for x in ad))
        except Exception:
            smooth = 0.0
    total = (__A__ * progress) + (__B__ * success) - (__C__ * collision) - (__D__ * obstacle) - (__E__ * smooth) - __F__
    return total, {"progress": progress, "goal": success, "collision": collision, "obstacle": obstacle, "smoothness": smooth, "time": 1.0}
""".strip(),
    ),
    (
        "safety_heavy",
        """
def compute_reward(transition):
    progress = float(transition.get("progress", 0.0))
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    clearance = float(transition.get("clearance", transition.get("min_obstacle_clearance", 1.0)) or 0.0)
    near = 1.0 if clearance < __A__ else 0.0
    total = (__B__ * progress) + (__C__ * success) - (__D__ * collision) - (__E__ * near) - __F__
    return total, {"progress": progress, "goal": success, "collision": collision, "obstacle": near, "time": 1.0}
""".strip(),
    ),
    (
        "sparse_goal",
        """
def compute_reward(transition):
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    total = (__A__ * success) - (__B__ * collision) - __C__
    return total, {"goal": success, "collision": collision, "time": 1.0}
""".strip(),
    ),
    (
        "progress_only",
        """
def compute_reward(transition):
    progress = float(transition.get("progress", 0.0))
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    total = (__A__ * progress) - (__B__ * collision) - __C__
    return total, {"progress": progress, "collision": collision, "time": 1.0}
""".strip(),
    ),
    (
        "potential",
        """
def compute_reward(transition):
    dist = float(transition.get("goal_distance", 0.0) or 0.0)
    prev = float(transition.get("prev_goal_distance", dist) or dist)
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    shaping = __A__ * (prev - dist)
    total = shaping + (__B__ * success) - (__C__ * collision) - __D__
    return total, {"progress": prev - dist, "goal": success, "collision": collision, "potential": shaping, "time": 1.0}
""".strip(),
    ),
    (
        "smooth_focus",
        """
def compute_reward(transition):
    progress = float(transition.get("progress", 0.0))
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    ad = transition.get("action_delta", 0.0)
    try:
        smooth = float(ad)
    except (TypeError, ValueError):
        smooth = 0.0
    total = (__A__ * progress) + (__B__ * success) - (__C__ * collision) - (__D__ * (smooth * smooth)) - __E__
    return total, {"progress": progress, "goal": success, "collision": collision, "smoothness": smooth * smooth, "time": 1.0}
""".strip(),
    ),
    (
        "time_heavy",
        """
def compute_reward(transition):
    progress = float(transition.get("progress", 0.0))
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    total = (__A__ * progress) + (__B__ * success) - (__C__ * collision) - __D__
    return total, {"progress": progress, "goal": success, "collision": collision, "time": 1.0}
""".strip(),
    ),
    (
        "goal_progress_safe",
        """
def compute_reward(transition):
    progress = float(transition.get("progress", 0.0))
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    clearance = float(transition.get("clearance", transition.get("min_obstacle_clearance", 1.0)) or 0.0)
    obstacle = max(0.0, 1.0 - clearance / 0.5)
    total = (__A__ * progress) + (__B__ * success) - (__C__ * collision) - (__D__ * obstacle) - __E__
    return total, {"progress": progress, "goal": success, "collision": collision, "obstacle": obstacle, "time": 1.0}
""".strip(),
    ),
    (
        "fast_goal",
        """
def compute_reward(transition):
    progress = float(transition.get("progress", 0.0))
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    total = (__A__ * progress) + (__B__ * success) - (__C__ * collision) - __D__
    return total, {"progress": progress, "goal": success, "collision": collision, "time": 1.0}
""".strip(),
    ),
    (
        "nonlinear_progress",
        """
def compute_reward(transition):
    progress = float(transition.get("progress", 0.0))
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    shaped = progress if progress <= 0.0 else (progress ** __A__)
    total = (__B__ * shaped) + (__C__ * success) - (__D__ * collision) - __E__
    return total, {"progress": progress, "nonlinear_progress": shaped, "goal": success, "collision": collision, "time": 1.0}
""".strip(),
    ),
    (
        "distance_penalty",
        """
def compute_reward(transition):
    dist = float(transition.get("goal_distance", 0.0) or 0.0)
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    total = (-__A__ * dist) + (__B__ * success) - (__C__ * collision) - __D__
    return total, {"distance": dist, "goal": success, "collision": collision, "time": 1.0}
""".strip(),
    ),
    (
        "risk_clearance",
        """
def compute_reward(transition):
    progress = float(transition.get("progress", 0.0))
    clearance = float(transition.get("clearance", transition.get("min_obstacle_clearance", 1.0)) or 0.0)
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    risk = 0.0 if clearance >= __A__ else ((__A__ - clearance) / __A__)
    total = (__B__ * progress) + (__C__ * success) - (__D__ * collision) - (__E__ * risk) - __F__
    return total, {"progress": progress, "goal": success, "collision": collision, "risk": risk, "time": 1.0}
""".strip(),
    ),
]


def _slots_for(family: str, digest: str) -> dict[str, float]:
    """Map digest bytes to plausible 'trying-to-be-good' weights."""
    vals = [((int(digest[i : i + 2], 16) / 255.0)) for i in range(0, 12, 2)]
    while len(vals) < 6:
        vals.append(0.5)
    a, b, c, d, e, f = vals
    if family == "balanced":
        return {"A": 0.5 + a, "B": 5.0 + 10.0 * b, "C": 5.0 + 10.0 * c, "D": 0.2 + d, "E": 0.05 + 0.2 * e, "F": 0.005 + 0.02 * f}
    if family == "safety_heavy":
        return {"A": 0.3 + 0.5 * a, "B": 0.3 + 0.7 * b, "C": 5.0 + 10.0 * c, "D": 10.0 + 20.0 * d, "E": 2.0 + 4.0 * e, "F": 0.01}
    if family == "sparse_goal":
        # Legitimate sparse design attempt (still includes collision cost).
        return {"A": 8.0 + 12.0 * a, "B": 8.0 + 12.0 * b, "C": 0.01}
    if family == "progress_only":
        return {"A": 0.8 + 1.5 * a, "B": 2.0 + 8.0 * b, "C": 0.01}
    if family == "potential":
        return {"A": 0.5 + a, "B": 5.0 + 10.0 * b, "C": 5.0 + 10.0 * c, "D": 0.01}
    if family == "smooth_focus":
        return {"A": 0.5 + a, "B": 5.0 + 8.0 * b, "C": 5.0 + 8.0 * c, "D": 0.5 + 2.0 * d, "E": 0.01}
    if family == "time_heavy":
        return {"A": 0.5 + a, "B": 5.0 + 8.0 * b, "C": 5.0 + 8.0 * c, "D": 0.05 + 0.15 * d}
    if family == "goal_progress_safe":
        return {"A": 0.8 + a, "B": 8.0 + 12.0 * b, "C": 8.0 + 12.0 * c, "D": 0.5 + d, "E": 0.01}
    if family == "fast_goal":
        return {"A": 1.0 + 1.5 * a, "B": 10.0 + 10.0 * b, "C": 5.0 + 10.0 * c, "D": 0.02 + 0.05 * d}
    if family == "nonlinear_progress":
        return {"A": 1.2 + 1.5 * a, "B": 0.5 + a, "C": 5.0 + 8.0 * c, "D": 5.0 + 8.0 * d, "E": 0.01}
    if family == "distance_penalty":
        return {"A": 0.05 + 0.2 * a, "B": 5.0 + 10.0 * b, "C": 5.0 + 10.0 * c, "D": 0.01}
    if family == "risk_clearance":
        return {"A": 0.4 + 0.6 * a, "B": 0.5 + b, "C": 5.0 + 8.0 * c, "D": 5.0 + 8.0 * d, "E": 1.0 + 3.0 * e, "F": 0.01}
    return {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 0.01}


def render_reward(family: str, template: str, digest: str) -> str:
    slots = _slots_for(family, digest)
    code = template
    for key, val in slots.items():
        code = code.replace(f"__{key}__", f"{val:.6f}")
    return code


def mutate_parent_code(parent_code: str, digest: str, stats: dict[str, Any] | None = None) -> str:
    """Scale numeric literals in parent code slightly; bias by mean_success if present."""
    mean_s = None if not stats else stats.get("mean_success")
    scale = 1.05
    if isinstance(mean_s, (int, float)):
        scale = 1.15 if float(mean_s) < 0.3 else (0.95 if float(mean_s) > 0.7 else 1.05)
    bump = 0.9 + 0.3 * (int(digest[:2], 16) / 255.0)

    def _repl(match: re.Match[str]) -> str:
        val = float(match.group(0))
        if abs(val) < 1e-12:
            return match.group(0)
        return f"{val * scale * bump:.6f}"

    return re.sub(r"(?<![A-Za-z_])\d+\.\d+", _repl, parent_code, count=8)


def pick_family(index: int, digest: str) -> tuple[str, str]:
    idx = (index + int(digest[:2], 16)) % len(TEMPLATES)
    return TEMPLATES[idx]


def sample_code(prompt: str, *, index: int = 0) -> tuple[str, dict[str, Any]]:
    digest = hashlib.sha256(f"{prompt}|{index}".encode()).hexdigest()
    stats: dict[str, Any] | None = None
    stats_match = re.search(r"Parent training stats:\s*(\{.*?\})", prompt, re.S)
    if stats_match:
        try:
            stats = json.loads(stats_match.group(1))
        except json.JSONDecodeError:
            stats = None

    parent_code = ""
    parent_match = re.search(
        r"Parent code(?: \(none for generation 0\))?:\n(.*?)(?:\n\nParent training stats:|\Z)",
        prompt,
        re.S,
    )
    if parent_match:
        parent_code = parent_match.group(1).strip()
    if parent_code.startswith("# none"):
        parent_code = ""

    if parent_code and "def compute_reward" in parent_code:
        code = mutate_parent_code(parent_code, digest, stats)
        return code, {"family": "mutate", "digest": digest[:8], "local": True}

    family, template = pick_family(index, digest)
    code = render_reward(family, template, digest)
    return code, {"family": family, "digest": digest[:8], "local": True}
