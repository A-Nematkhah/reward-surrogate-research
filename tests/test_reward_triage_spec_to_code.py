"""spec_to_code must match WeightedRewardFunction numerically (blocking)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reward_triage.executor import run_candidate_on_probes
from reward_triage.spec_to_code import COMPONENT_NAMES, spec_to_code, specs_from_rows
from rsr.features.probe_trajectories import load_probe_set
from rsr.rewards.reward_function import RewardSpec, WeightedRewardFunction

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "navigation_2d" / "phase1_108" / "candidates_seed_agg.csv"
PHASE2_PROBES = ROOT / "experiments" / "phase2" / "probe_trajectories.npz"


def _serialize_for_sandbox(transition: dict) -> dict:
    """Pickle-safe transition matching executor cleaning rules."""
    out = {}
    for k, v in transition.items():
        if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "radius"):
            out[k] = [float(v.x), float(v.y), float(v.radius)]
            continue
        if isinstance(v, list) and v and hasattr(v[0], "x"):
            out[k] = [[float(o.x), float(o.y), float(o.radius)] for o in v]
            continue
        if hasattr(v, "tolist") and not isinstance(v, (list, tuple)):
            out[k] = v.tolist()
            continue
        out[k] = v
    return out


def test_specs_from_rows_reads_weight_columns():
    df = pd.read_csv(CANDIDATES)
    specs = specs_from_rows(df.to_dict(orient="records"))
    assert len(specs) == 108
    # At least one non-zero weight somewhere (CSV uses weight_*)
    assert any(any(abs(w) > 0 for w in s["weights"].values()) for s in specs)
    # Families that use extra components must have those weights populated
    by_id = {s["candidate_id"]: s for s in specs}
    assert abs(by_id["balanced_00"]["weights"]["progress"]) > 0
    assert "energy" in by_id["balanced_00"]["weights"]
    assert set(COMPONENT_NAMES) <= set(by_id["balanced_00"]["weights"])


def test_spec_to_code_matches_weighted_reward_all_108():
    """Generated sandbox code vs WeightedRewardFunction: max |diff| < 1e-9."""
    if not CANDIDATES.exists() or not PHASE2_PROBES.exists():
        pytest.skip("local phase1_108 candidates / phase2 probes required")

    df = pd.read_csv(CANDIDATES)
    specs = specs_from_rows(df.to_dict(orient="records"))
    assert len(specs) == 108

    episodes = load_probe_set(PHASE2_PROBES)
    transitions = [tr for ep in episodes for tr in ep.transitions]
    sandbox_transitions = [_serialize_for_sandbox(t) for t in transitions]

    max_abs = 0.0
    worst = None
    for spec in specs:
        code = spec_to_code(spec)
        rs = RewardSpec(
            candidate_id=spec["candidate_id"],
            family_id=spec["family_id"],
            weights=spec["weights"],
            scales=spec.get("scales") or {},
        )
        fn = WeightedRewardFunction.from_spec(rs)
        real = np.asarray([fn(None, t) for t in transitions], dtype=np.float64)
        # Same sandbox loader as the child process (in_process for speed on 108×N).
        result = run_candidate_on_probes(
            code, sandbox_transitions, timeout_s=30.0, in_process=True
        )
        assert result.report.status == "ok", (
            f"{spec['candidate_id']}: {result.report.status} {result.report.reasons}"
        )
        gen = np.asarray(result.rewards, dtype=np.float64)
        assert gen.shape == real.shape
        diff = float(np.max(np.abs(gen - real)))
        if diff > max_abs:
            max_abs = diff
            worst = spec["candidate_id"]
        assert diff < 1e-9, (
            f"{spec['candidate_id']}: max|diff|={diff} (worst so far {worst}={max_abs})"
        )
