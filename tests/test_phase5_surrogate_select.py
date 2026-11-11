"""Phase 5 surrogate + selector unit tests."""

from __future__ import annotations

import numpy as np

from reward_triage.model import SurrogateModel
from reward_triage.schema import Prediction
from reward_triage.select import select_candidates


def test_surrogate_fit_predict_save(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = X[:, 0] * 0.2 + 0.3 + rng.normal(scale=0.05, size=40)
    cols = [f"f{i}" for i in range(5)]
    model = SurrogateModel(cols, feature_set="math+behavioral")
    model.fit(X, y, dataset_hash="toy")
    preds = model.predict_rows(X[:3])
    assert len(preds) == 3
    assert all(isinstance(p, Prediction) for p in preds)
    assert preds[0].interval[0] <= preds[0].mean <= preds[0].interval[1]
    out = model.save(tmp_path / "art")
    loaded = SurrogateModel.load(out)
    p2 = loaded.predict_rows(X[:1])[0]
    assert abs(p2.mean - preds[0].mean) < 1e-9


def test_select_topk_and_ood():
    preds = [
        Prediction(mean=0.9, std=0.1, interval=(0.8, 1.0), in_distribution=True, feature_hash="a"),
        Prediction(mean=0.2, std=0.1, interval=(0.1, 0.3), in_distribution=True, feature_hash="b"),
        Prediction(mean=0.1, std=0.1, interval=(0.0, 0.2), in_distribution=False, feature_hash="c"),
        Prediction(mean=0.5, std=0.1, interval=(0.4, 0.6), in_distribution=True, feature_hash="d"),
    ]
    hard = [False, False, False, False]
    items = select_candidates(preds, hard_reject=hard, k_frac=0.5, audit_frac=0.0)
    # top half of 3 ID → k=2 among ranked ID; plus OOD always train
    train_idx = {s.index for s in items if s.decision == "train"}
    assert 2 in train_idx  # OOD
    assert 0 in train_idx  # highest ID score
    assert items[1].decision in ("skip", "audit", "train")


def test_select_hard_reject():
    preds = [None, None]
    items = select_candidates(
        preds, hard_reject=[True, False], k_frac=0.5, audit_frac=0.0
    )
    assert items[0].decision == "skip"
    assert items[1].decision == "train"


def test_adapt_with_prior_and_save(tmp_path):
    from reward_triage.adapt import adapt_after_generation
    from reward_triage.schema import CandidateRecord, ValidationReport
    from reward_triage.store import SeedLabel, SelectionLog, StoreRecord

    rng = np.random.default_rng(1)
    cols = [f"f{i}" for i in range(4)]
    model = SurrogateModel(cols, feature_set="math+behavioral")
    X0 = rng.normal(size=(20, 4))
    y0 = rng.uniform(0, 1, size=20)
    model.fit(X0, y0)

    rec = StoreRecord(
        candidate=CandidateRecord(
            candidate_id="c1",
            code="def compute_reward(t): return 0.0, {}",
            code_hash="abcd",
            host="test",
            env_id="2d",
            lineage_id="lin1",
        ),
        tier0=ValidationReport(status="ok"),
        features={c: float(rng.normal()) for c in cols},
        feature_hash="h",
        selection=SelectionLog(
            score=0.5, probability=1.0, decision="train", mode="off"
        ),
        labels=[
            SeedLabel(
                seed=0,
                success=0.7,
                collision=0.1,
                timeout=0.2,
                return_mean=1.0,
                eval_episodes=10,
                train_steps=100,
                wall_time_s=1.0,
                status="completed",
            )
        ],
    )
    info = adapt_after_generation(
        model,
        [rec],
        out_dir=tmp_path / "surr",
        generation=0,
        use_prior=False,
        min_new_labels=1,
    )
    assert info["adapted"] is True
    assert (tmp_path / "surr" / "gen_0" / "model.joblib").exists()
    loaded = SurrogateModel.load(tmp_path / "surr" / "gen_0")
    assert loaded.meta is not None
    assert loaded.meta.n_train >= 1
