"""Frozen probe trajectory sets for black-box reward feature extraction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROBE_KEYS = (
    "state",
    "action",
    "next_state",
    "done",
    "progress",
    "collision",
    "success",
    "clearance",
    "action_delta",
    "policy",
    "reward_placeholder",
)


@dataclass(frozen=True)
class ProbeBatch:
    env_id: str
    transitions: list[dict[str, Any]]
    content_hash: str
    meta: dict[str, Any]


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def transition_gt_signals(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a raw env transition into the frozen probe schema."""
    prev_g = float(raw.get("prev_goal_distance", 0.0))
    goal = float(raw.get("goal_distance", 0.0))
    progress = float(raw.get("progress", prev_g - goal))
    action = np.asarray(raw.get("action", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    prev_action = np.asarray(raw.get("prev_action", np.zeros_like(action)), dtype=np.float64)
    clearance = raw.get("clearance", raw.get("min_obstacle_clearance"))
    if clearance is None:
        clearance = 0.0
    state = np.asarray(raw.get("prev_position", raw.get("state", [0.0, 0.0])), dtype=np.float64)
    next_state = np.asarray(raw.get("position", raw.get("next_state", state)), dtype=np.float64)
    done = bool(
        raw.get("done")
        or raw.get("success")
        or raw.get("collision")
        or raw.get("timeout")
        or raw.get("out_of_bounds")
    )
    return {
        "state": state.tolist(),
        "action": action.tolist(),
        "next_state": next_state.tolist(),
        "done": done,
        "progress": progress,
        "collision": bool(raw.get("collision", False)),
        "success": bool(raw.get("success", False)),
        "clearance": float(clearance),
        "action_delta": (action - prev_action).tolist(),
        "policy": str(raw.get("policy", "unknown")),
        "reward_placeholder": 0.0,
        # Keep extras for adapters that need them.
        "prev_goal_distance": prev_g,
        "goal_distance": goal,
        "prev_position": state.tolist(),
        "position": next_state.tolist(),
        "prev_action": prev_action.tolist(),
        "min_obstacle_clearance": float(clearance),
    }


def make_synthetic_probe_batch(
    env_id: str,
    *,
    n_per_policy: int = 4,
    seed: int = 0,
) -> ProbeBatch:
    """Deterministic synthetic probes for unit tests (no env deps)."""
    rng = np.random.default_rng(seed)
    policies = [
        "random",
        "goal_seeking",
        "obstacle_avoiding",
        "partial_expert",
        "edge_states",
    ]
    transitions: list[dict[str, Any]] = []
    for policy in policies:
        for i in range(n_per_policy):
            pos = rng.uniform(-2, 2, size=2)
            nxt = pos + rng.normal(0, 0.1, size=2)
            action = rng.uniform(-1, 1, size=2)
            prev_action = rng.uniform(-1, 1, size=2)
            prev_g = float(np.linalg.norm(pos - np.array([3.0, 3.0])))
            goal = float(np.linalg.norm(nxt - np.array([3.0, 3.0])))
            near_goal = policy == "edge_states" and i % 3 == 0
            near_collision = policy == "edge_states" and i % 3 == 1
            raw = {
                "prev_position": pos,
                "position": nxt,
                "action": action,
                "prev_action": prev_action,
                "prev_goal_distance": 0.05 if near_goal else prev_g,
                "goal_distance": 0.01 if near_goal else goal,
                "collision": near_collision,
                "success": near_goal and not near_collision,
                "timeout": False,
                "out_of_bounds": policy == "edge_states" and i % 3 == 2,
                "min_obstacle_clearance": 0.01 if near_collision else float(rng.uniform(0.2, 2.0)),
                "policy": policy,
            }
            transitions.append(transition_gt_signals(raw))
    payload = json.dumps(transitions, sort_keys=True).encode("utf-8")
    return ProbeBatch(
        env_id=env_id,
        transitions=transitions,
        content_hash=_hash_bytes(payload),
        meta={
            "synthetic": True,
            "n_per_policy": n_per_policy,
            "seed": seed,
            "policies": policies,
        },
    )


def save_probe_npz(batch: ProbeBatch, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store as object arrays of JSON strings for ragged lists.
    encoded = [json.dumps(t, sort_keys=True) for t in batch.transitions]
    np.savez_compressed(
        path,
        env_id=np.asarray(batch.env_id),
        content_hash=np.asarray(batch.content_hash),
        meta_json=np.asarray(json.dumps(batch.meta, sort_keys=True)),
        transitions_json=np.asarray(encoded, dtype=object),
    )
    digest = _hash_bytes(path.read_bytes())
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def load_probe_npz(path: str | Path) -> ProbeBatch:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    encoded = list(data["transitions_json"])
    transitions = [json.loads(str(s)) for s in encoded]
    meta = json.loads(str(data["meta_json"]))
    return ProbeBatch(
        env_id=str(data["env_id"]),
        transitions=transitions,
        content_hash=str(data["content_hash"]),
        meta=meta,
    )


def verify_probe_hash(path: str | Path) -> bool:
    path = Path(path)
    side = path.with_suffix(path.suffix + ".sha256")
    if not side.exists():
        return False
    expected = side.read_text(encoding="utf-8").strip().split()[0]
    actual = _hash_bytes(path.read_bytes())
    return actual == expected


def default_probe_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "probes"


def ensure_default_synthetic_probes(out_dir: str | Path | None = None) -> dict[str, Path]:
    """Write synthetic NPZ for 2d / crowdnav / irsim if missing."""
    out = Path(out_dir) if out_dir else default_probe_dir()
    out.mkdir(parents=True, exist_ok=True)
    mapping = {
        "navigation_2d": "navigation_2d.npz",
        "crowdnav": "crowdnav.npz",
        "irsim": "irsim.npz",
    }
    paths: dict[str, Path] = {}
    for env_id, name in mapping.items():
        path = out / name
        if not path.exists():
            batch = make_synthetic_probe_batch(env_id, n_per_policy=4, seed=hash(env_id) % 10_000)
            save_probe_npz(batch, path)
        paths[env_id] = path
    return paths
