"""Black-box features, Tier 0, and determinism."""

from __future__ import annotations

from pathlib import Path

from reward_triage.executor import run_candidate_on_probes
from reward_triage.features import extract_blackbox_features, feature_hash
from reward_triage.pipeline import triage_candidate
from reward_triage.probes import (
    REQUIRED_POLICIES,
    assert_required_policies,
    fixture_probe_dir,
    load_probe_npz,
    verify_probe_hash,
)
from reward_triage.spec_to_code import spec_to_code
from reward_triage.tier0 import apply_tier0, is_hard_reject

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "probes"


VARYING_CODE = """
def compute_reward(transition):
    return float(transition.get("progress", 0.0)) * 2.0 + float(transition.get("clearance", 0.0))
"""

CONSTANT_NONEZERO_CODE = """
def compute_reward(transition):
    return 1.0
"""

CONSTANT_ZERO_CODE = """
def compute_reward(transition):
    return 0.0
"""

NAN_CODE = """
def compute_reward(transition):
    return float("nan")
"""


def _fixture_2d():
    return load_probe_npz(fixture_probe_dir() / "navigation_2d.npz")


def test_fixture_and_frozen_probe_hashes():
    for path in fixture_probe_dir().glob("*.npz"):
        assert verify_probe_hash(path)
        batch = load_probe_npz(path)
        assert_required_policies(batch)
        assert batch.meta.get("synthetic") is True
    for name in ("navigation_2d.npz", "crowdnav.npz", "irsim.npz"):
        path = FROZEN / name
        assert path.exists(), f"missing frozen real probe set {path}"
        assert verify_probe_hash(path)
        batch = load_probe_npz(path)
        assert_required_policies(batch)
        assert batch.meta.get("synthetic") is False
        for p in REQUIRED_POLICIES:
            assert p in set(batch.meta.get("policies") or [])


def test_tier0_constant_and_non_finite():
    batch = _fixture_2d()
    tr = batch.transitions[:8]
    # Non-zero constant (e.g. pure time cost) is a warning, not a hard reject.
    nz_exec = run_candidate_on_probes(CONSTANT_NONEZERO_CODE, tr, timeout_s=10.0)
    nz_rep = apply_tier0(nz_exec)
    assert nz_rep.status == "warn"
    assert not is_hard_reject(nz_rep)

    z_exec = run_candidate_on_probes(CONSTANT_ZERO_CODE, tr, timeout_s=10.0)
    z_rep = apply_tier0(z_exec)
    assert z_rep.status == "constant"
    assert is_hard_reject(z_rep)

    nan_exec = run_candidate_on_probes(NAN_CODE, tr, timeout_s=10.0)
    nan_rep = apply_tier0(nan_exec)
    assert nan_rep.status == "non_finite"
    assert is_hard_reject(nan_rep)


def test_feature_determinism():
    batch = _fixture_2d()
    tr = batch.transitions
    groups = ("bb_stats", "bb_corr", "bb_policy", "bb_shape")
    r1, f1, h1, _ = triage_candidate(VARYING_CODE, tr, feature_groups=groups)
    _r2, f2, h2, _ = triage_candidate(VARYING_CODE, tr, feature_groups=groups)
    assert r1.status in ("ok", "warn")
    assert f1 is not None and f2 is not None
    assert h1 == h2 == feature_hash(f1)
    assert f1 == f2
    assert "bb_stats_mean" in f1
    assert "bb_corr_pearson_progress" in f1
    assert "bb_policy_return_random" in f1
    assert "bb_shape_return_g09" in f1


def test_spec_to_code_runs():
    weights = {
        "progress": 1.0,
        "goal": 1.0,
        "collision": 1.0,
        "obstacle": 0.5,
        "smoothness": 0.1,
        "time": 0.01,
    }
    code = spec_to_code(
        {
            "candidate_id": "demo",
            "family_id": "balanced",
            "weights": weights,
            "scales": {},
        }
    )
    batch = _fixture_2d()
    report, feats, _, _ = triage_candidate(
        code, batch.transitions[:10], weights=weights, feature_groups=("mb",)
    )
    assert report.status in ("ok", "warn")
    assert feats is not None
    assert "math_variance" in feats
    assert "total_mean" in feats
    feats2 = extract_blackbox_features(
        [0.1, 0.2, 0.3],
        batch.transitions[:3],
        code=code,
        groups=("bb_stats", "code"),
    )
    assert "code_ast_nodes" in feats2
