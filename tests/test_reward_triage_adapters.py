"""Adapter contract tests."""

from __future__ import annotations

from pathlib import Path

from reward_triage.adapters import (
    CrowdNavAdapter,
    EurekaStyleAdapter,
    IRSimAdapter,
    TwoDAdapter,
)
from reward_triage.probes import make_synthetic_probe_batch

CODE = """
def compute_reward(transition):
    p = float(transition.get("progress", 0.0))
    return p, {"progress": p}
"""

ROOT = Path(__file__).resolve().parents[1]


def test_twod_adapter_contract():
    ad = TwoDAdapter()
    ad.load(CODE)
    tr = make_synthetic_probe_batch("navigation_2d", n_per_policy=1, seed=0).transitions[0]
    total, comps = ad.call(tr)
    assert isinstance(total, float)
    assert comps is not None and "progress" in comps


def test_crowdnav_and_irsim_adapters():
    tr = make_synthetic_probe_batch("crowdnav", n_per_policy=1, seed=2).transitions[0]
    for cls in (CrowdNavAdapter, IRSimAdapter):
        ad = cls()
        ad.load(CODE)
        total, _ = ad.call(tr)
        assert isinstance(total, float)


def test_eureka_adapter_from_yaml():
    path = ROOT / "configs" / "adapters" / "eureka.yaml"
    ad = EurekaStyleAdapter.from_yaml(path)
    ad.load(CODE)
    tr = make_synthetic_probe_batch("navigation_2d", n_per_policy=1, seed=3).transitions[0]
    total, comps = ad.call(tr)
    assert isinstance(total, float)
    assert comps is not None
