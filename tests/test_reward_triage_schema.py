"""Schema smoke tests — no torch required."""

from __future__ import annotations

from reward_triage import CandidateRecord, Prediction, ValidationReport, __version__


def test_package_version():
    assert __version__


def test_candidate_roundtrip():
    rec = CandidateRecord(
        candidate_id="c1",
        code="def compute_reward(t): return 0.0",
        code_hash="abc",
        host="mini",
        env_id="navigation_2d",
        lineage_id="L1",
    )
    assert CandidateRecord.from_dict(rec.to_dict()).candidate_id == "c1"


def test_validation_and_prediction_roundtrip():
    v = ValidationReport(status="ok", reasons=[], timings={"wall_s": 0.1})
    assert ValidationReport.from_dict(v.to_dict()).status == "ok"
    p = Prediction(mean=0.5, std=0.1, interval=(0.2, 0.8), in_distribution=True, feature_hash="x")
    assert Prediction.from_dict(p.to_dict()).feature_hash == "x"
