#!/usr/bin/env python3
"""Score local-corpus rewards with ExtraTrees on chosen 2D feature sets.

Feature sets: math+behavioral | early | early+math+behavioral

Local candidates only have zero-cost mb features (no PPO). For sets that
include early_*, those columns are left missing and median-imputed from the
108 training rows — so early-only ranks among locals are nearly flat;
early+math+behavioral still ranks via the mb part.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_triage.llm_client import LocalCorpusLLMClient  # noqa: E402
from reward_triage.local_rewards import TEMPLATES  # noqa: E402
from reward_triage.pipeline import triage_candidate  # noqa: E402
from reward_triage.probes import load_probe_npz  # noqa: E402
from rsr.surrogate.model_comparison import _within_family_spearman  # noqa: E402

DATA = ROOT / "data" / "navigation_2d" / "phase1_108"
OUT = ROOT / "results" / "phase4_local_surrogate_probe"

FEATURE_SETS = ("math+behavioral", "early", "early+math+behavioral")


def _et() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=250,
                    random_state=0,
                    n_jobs=1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def _train_table(feature_set: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    meta = pd.read_csv(DATA / "features" / "features_meta.csv")
    math_df = pd.read_csv(DATA / "features" / "features_math.csv")
    beh_df = pd.read_csv(DATA / "features" / "features_behavioral.csv")
    early_df = pd.read_csv(DATA / "features" / "features_early.csv")
    targets = pd.read_csv(DATA / "features" / "features_targets.csv")
    labels = pd.read_csv(DATA / "candidates_seed_agg.csv")[
        ["candidate_id", "success_rate"]
    ].rename(columns={"success_rate": "success_rate_agg"})

    parts: list[pd.DataFrame] = [meta]
    cols: list[str] = []
    if feature_set == "math+behavioral":
        parts.extend([math_df, beh_df])
        cols = list(math_df.columns) + list(beh_df.columns)
    elif feature_set == "early":
        parts.append(early_df)
        cols = list(early_df.columns)
    elif feature_set == "early+math+behavioral":
        parts.extend([early_df, math_df, beh_df])
        cols = list(early_df.columns) + list(math_df.columns) + list(beh_df.columns)
    else:
        raise ValueError(feature_set)

    feats = pd.concat(parts + [targets[["success_rate"]]], axis=1)
    feats = feats.merge(labels, on="candidate_id", how="left")
    feats["success_rate"] = feats["success_rate_agg"].fillna(feats["success_rate"])
    feature_cols = [c for c in cols if pd.api.types.is_numeric_dtype(feats[c])]
    X = feats[feature_cols].to_numpy(dtype=float)
    y = feats["success_rate"].to_numpy(dtype=float)
    return feats[["candidate_id", "family_id", "success_rate"]], X, y, feature_cols


def _lofo(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float, np.ndarray]:
    preds = np.zeros(len(y), dtype=float)
    model = _et()
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    within = float(
        _within_family_spearman(y, preds, groups, min_size=2)["within_family_spearman_mean"]
    )
    pooled = float(spearmanr(y, preds).correlation)
    return within, pooled, preds


def _mb_dict_to_row(feats: dict[str, float], feature_cols: list[str]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in feats.items():
        name = k[3:] if k.startswith("mb_") else k
        flat[name] = float(v)
    # early_* absent → NaN (imputed at predict time)
    return {c: flat.get(c, np.nan) for c in feature_cols}


def _score_locals(
    model: Pipeline,
    feature_cols: list[str],
    *,
    n: int = 16,
) -> pd.DataFrame:
    probes = load_probe_npz(ROOT / "data" / "probes" / "navigation_2d.npz")
    client = LocalCorpusLLMClient()
    prompt = (
        "Task: 2D navigation to goal with obstacles.\n"
        "Parent code (none for generation 0):\n# none\n\n"
        "Parent training stats:\n{}"
    )
    responses = client.complete(prompt, temperature=0.7, n=n)
    rows = []
    for i, resp in enumerate(responses):
        report, feats, fhash, _ = triage_candidate(
            resp.text, probes.transitions, in_process=True, feature_groups=("mb",)
        )
        if feats is None:
            rows.append(
                {
                    "index": i,
                    "family": resp.meta.get("family"),
                    "tier0": report.status,
                    "pred_success": np.nan,
                }
            )
            continue
        row = _mb_dict_to_row(feats, feature_cols)
        x = np.array([[row[c] for c in feature_cols]], dtype=float)
        n_early_missing = int(
            sum(1 for c in feature_cols if c.startswith("early_") and np.isnan(row[c]))
        )
        pred = float(model.predict(x)[0])
        rows.append(
            {
                "index": i,
                "family": resp.meta.get("family"),
                "tier0": report.status,
                "pred_success": pred,
                "n_early_missing": n_early_missing,
                "feature_hash": fhash,
            }
        )
    return pd.DataFrame(rows).sort_values("pred_success", ascending=False)


def run_one(feature_set: str) -> dict:
    meta, X, y, feature_cols = _train_table(feature_set)
    within, pooled, _ = _lofo(X, y, meta["family_id"].to_numpy())
    model = _et()
    model.fit(X, y)
    scored = _score_locals(model, feature_cols)
    tag = feature_set.replace("+", "_")
    scored.to_csv(OUT / f"local_scored_{tag}.csv", index=False)
    pred = scored["pred_success"]
    return {
        "feature_set": feature_set,
        "n_features": len(feature_cols),
        "lofo_within_family_spearman": within,
        "lofo_pooled_spearman": pooled,
        "local_pred_mean": float(pred.mean(skipna=True)),
        "local_pred_std": float(pred.std(skipna=True)),
        "local_pred_min": float(pred.min(skipna=True)),
        "local_pred_max": float(pred.max(skipna=True)),
        "local_pred_range": float(pred.max(skipna=True) - pred.min(skipna=True)),
        "top3": scored.head(3)[["family", "pred_success"]].to_dict(orient="records"),
        "bottom3": scored.dropna(subset=["pred_success"]).tail(3)[["family", "pred_success"]].to_dict(
            orient="records"
        ),
        "early_imputed_for_local": feature_set != "math+behavioral",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--feature-set",
        choices=[*FEATURE_SETS, "all"],
        default="all",
    )
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sets = list(FEATURE_SETS) if args.feature_set == "all" else [args.feature_set]
    results = [run_one(fs) for fs in sets]
    summary = {
        "train_n": 108,
        "local_n": 16,
        "local_n_families_in_corpus": len(TEMPLATES),
        "results": results,
        "note": (
            "Local candidates have mb only. For early* columns, values are missing and "
            "median-imputed from the 108 — early-only local ranks are nearly constant; "
            "early+math+behavioral still separates via mb. Not live PPO labels."
        ),
    }
    (OUT / "summary_feature_sets.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\n=== LOFO within-family (108 train) ===")
    for r in results:
        print(
            f"{r['feature_set']:28s}  within={r['lofo_within_family_spearman']:.3f}  "
            f"pooled={r['lofo_pooled_spearman']:.3f}  local_range={r['local_pred_range']:.4f}"
        )
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
