"""Frozen probe trajectory sets for black-box reward feature extraction."""

from __future__ import annotations

import hashlib
import json
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_POLICIES = (
    "random",
    "goal_seeking",
    "obstacle_avoiding",
    "partial_expert",
    "edge_states",
)


@dataclass(frozen=True)
class ProbeBatch:
    env_id: str
    transitions: list[dict[str, Any]]
    content_hash: str
    meta: dict[str, Any]


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def stable_env_seed(env_id: str, base_seed: int = 0) -> int:
    """Process-stable seed derived from env_id (not Python's salted hash())."""
    return int(base_seed + (zlib.crc32(env_id.encode("utf-8")) & 0xFFFFFFFF) % 10_000)


def transition_gt_signals(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a raw env transition into the frozen probe schema."""
    prev_g = float(raw.get("prev_goal_distance", 0.0))
    goal = float(raw.get("goal_distance", 0.0))
    # Prefer env-provided progress when present.
    if "progress" in raw and raw["progress"] is not None:
        progress = float(raw["progress"])
    else:
        progress = prev_g - goal
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
    obstacles = raw.get("obstacles")
    if obstacles is not None:
        ser_obs = []
        for o in obstacles:
            if hasattr(o, "x"):
                ser_obs.append([float(o.x), float(o.y), float(o.radius)])
            else:
                ser_obs.append([float(o[0]), float(o[1]), float(o[2])])
    else:
        ser_obs = None
    target = raw.get("target")
    if target is not None and hasattr(target, "x"):
        ser_target = [float(target.x), float(target.y), float(target.radius)]
    elif target is not None:
        ser_target = [float(target[0]), float(target[1]), float(target[2])]
    else:
        ser_target = None

    return {
        "state": state.tolist(),
        "action": action.tolist(),
        "next_state": next_state.tolist(),
        "done": done,
        "progress": progress,
        "collision": bool(raw.get("collision", False)),
        "success": bool(raw.get("success", False)),
        "out_of_bounds": bool(raw.get("out_of_bounds", False)),
        "timeout": bool(raw.get("timeout", False)),
        "clearance": float(clearance),
        "action_delta": (action - prev_action).tolist(),
        "policy": str(raw.get("policy", "unknown")),
        "reward_placeholder": 0.0,
        "prev_goal_distance": prev_g,
        "goal_distance": goal,
        "prev_position": state.tolist(),
        "position": next_state.tolist(),
        "prev_action": prev_action.tolist(),
        "min_obstacle_clearance": float(clearance),
        "robot_radius": float(raw.get("robot_radius", 0.15)),
        "obstacles": ser_obs,
        "target": ser_target,
        "comfort_radius": raw.get("comfort_radius", 1.0),
    }


def episodes_to_batch(
    env_id: str,
    episodes: Sequence[Any],
    *,
    meta: Mapping[str, Any] | None = None,
) -> ProbeBatch:
    transitions: list[dict[str, Any]] = []
    for ep in episodes:
        policy = str(getattr(ep, "policy_name", "unknown"))
        for tr in getattr(ep, "transitions", []):
            raw = dict(tr)
            raw["policy"] = policy
            if "progress" not in raw:
                raw["progress"] = float(raw.get("prev_goal_distance", 0.0)) - float(
                    raw.get("goal_distance", 0.0)
                )
            if "clearance" not in raw and "min_obstacle_clearance" in raw:
                raw["clearance"] = raw["min_obstacle_clearance"]
            transitions.append(transition_gt_signals(raw))
    payload = json.dumps(transitions, sort_keys=True).encode("utf-8")
    meta_out = dict(meta or {})
    meta_out.setdefault("synthetic", False)
    policies = sorted({str(t.get("policy")) for t in transitions})
    meta_out["policies"] = policies
    return ProbeBatch(
        env_id=env_id,
        transitions=transitions,
        content_hash=_hash_bytes(payload),
        meta=meta_out,
    )


def make_synthetic_probe_batch(
    env_id: str,
    *,
    n_per_policy: int = 4,
    seed: int | None = None,
) -> ProbeBatch:
    """Deterministic synthetic probes for unit-test fixtures only."""
    if seed is None:
        seed = stable_env_seed(env_id, 0)
    rng = np.random.default_rng(seed)
    transitions: list[dict[str, Any]] = []
    for policy in REQUIRED_POLICIES:
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
                "obstacles": [[0.0, 0.0, 0.3]],
                "target": [3.0, 3.0, 0.2],
                "robot_radius": 0.15,
            }
            transitions.append(transition_gt_signals(raw))
    payload = json.dumps(transitions, sort_keys=True).encode("utf-8")
    return ProbeBatch(
        env_id=env_id,
        transitions=transitions,
        content_hash=_hash_bytes(payload),
        meta={
            "synthetic": True,
            "fixture_only": True,
            "n_per_policy": n_per_policy,
            "seed": seed,
            "policies": list(REQUIRED_POLICIES),
        },
    )


def save_probe_npz(batch: ProbeBatch, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def assert_required_policies(batch: ProbeBatch) -> None:
    policies = set(batch.meta.get("policies") or {t.get("policy") for t in batch.transitions})
    missing = [p for p in REQUIRED_POLICIES if p not in policies]
    if missing:
        raise AssertionError(f"probe batch missing policies: {missing}")


def default_probe_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "probes"


def fixture_probe_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "probes"
