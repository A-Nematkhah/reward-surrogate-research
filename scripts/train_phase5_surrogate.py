#!/usr/bin/env python3
"""Train Phase-5 ExtraTrees on 2D early+math+behavioral; optional cold-start curve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_triage.feature_matrix import load_phase1_feature_table  # noqa: E402
from reward_triage.model import SurrogateModel  # noqa: E402
from rsr.evaluation.triage_metrics import recall_at_k, regret_at_k  # noqa: E402
from rsr.surrogate.model_comparison import _within_family_spearman  # noqa: E402


def cold_start_curve(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    pool_size: int = 16,
    k: int = 5,
    n_pools: int = 200,
    train_sizes: list[int] | None = None,
    seed: int = 0,
) -> list[dict]:
    """Simulate growing train set; score random pools from the held-out remainder."""
    rng = np.random.default_rng(seed)
    n = len(y)
    train_sizes = train_sizes or [12, 24, 36, 48, 72, 96]
    rows = []
    order = rng.permutation(n)
    for m in train_sizes:
        if m >= n - pool_size:
            continue
        train_idx = order[:m]
        rest = order[m:]
        cols = [f"f{i}" for i in range(X.shape[1])]
        model = SurrogateModel(cols, feature_set="early+math+behavioral")
        model.fit(X[train_idx], y[train_idx])
        recalls = []
        regrets = []
        for _ in range(n_pools):
            pick = rng.choice(rest, size=min(pool_size, len(rest)), replace=False)
            preds = np.array([p.mean for p in model.predict_rows(X[pick])])
            true = y[pick]
            recalls.append(recall_at_k(true, preds, k=k))
            regrets.append(regret_at_k(true, preds, k=k))
        rows.append(
            {
                "n_train": m,
                "recall_at_k_mean": float(np.mean(recalls)),
                "regret_at_k_mean": float(np.mean(regrets)),
                "k": k,
                "pool_size": pool_size,
                "n_pools": n_pools,
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--feature-set",
        default="early+math+behavioral",
        choices=["early+math+behavioral", "math+behavioral", "early"],
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "phase5_surrogate_2d",
    )
    p.add_argument("--cold-start", action="store_true")
    args = p.parse_args()

    feats, feature_cols = load_phase1_feature_table(args.feature_set, root=ROOT / "data" / "navigation_2d" / "phase1_108")
    X = feats[feature_cols].to_numpy(dtype=float)
    y = feats["success_rate"].to_numpy(dtype=float)
    groups = feats["family_id"].astype(str).to_numpy()

    # LOFO sanity
    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = SurrogateModel(feature_cols, feature_set=args.feature_set)
        m.fit(X[tr], y[tr])
        oof[te] = np.array([p.mean for p in m.predict_rows(X[te])])
    within = _within_family_spearman(y, oof, groups, min_size=2)["within_family_spearman_mean"]

    model = SurrogateModel(feature_cols, feature_set=args.feature_set)
    model.fit(X, y, dataset_hash=f"phase1_108:{args.feature_set}")
    out = args.out
    model.save(out)

    summary = {
        "feature_set": args.feature_set,
        "n_train": int(len(y)),
        "n_features": len(feature_cols),
        "lofo_within_family_spearman": float(within),
        "artifact": str(out),
        "meta": model.meta.to_dict() if model.meta else {},
    }
    if args.cold_start:
        curve = cold_start_curve(X, y, groups)
        summary["cold_start"] = curve
        # Suggested m_min: first size with recall>=0.7 or last
        m_min = curve[-1]["n_train"] if curve else 24
        for row in curve:
            if row["recall_at_k_mean"] >= 0.70:
                m_min = row["n_train"]
                break
        summary["suggested_m_min"] = int(m_min)
        summary["suggested_k_frac"] = 0.5

    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
