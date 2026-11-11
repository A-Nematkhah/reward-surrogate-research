"""Phase 6c — fix Gate-2 harness (dedupe/code_hash folds) + novelty-skip offline.

Does NOT change the Gate 2 FAIL verdict or frozen thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_triage.gate2_harness import (  # noqa: E402
    RUN_IDS,
    assert_no_code_hash_leak,
    dedupe_by_code_hash,
    feature_columns,
    load_all_records,
    lolo_oof,
    mb_feature_matrix,
    mean_success,
    records_to_frame,
    seed_se,
)
from rsr.evaluation.pool_simulator import simulate_pools  # noqa: E402
from rsr.evaluation.triage_metrics import (  # noqa: E402
    recall_at_k,
    regret_at_k,
    spearman_safe,
)

N_POOL = 16
K = 5
FEATURE_STEPS = 12_000
FULL_STEPS = 50_000


def pool_metrics_block(oof: pd.DataFrame, *, n_pools: int, n_boot: int, seed: int) -> dict:
    if len(oof) < N_POOL:
        return {"error": "too_few_rows", "n": int(len(oof))}
    per, summary = simulate_pools(
        oof,
        pool_sizes=(N_POOL,),
        n_groups_list=(1,),
        ks=(K,),
        n_pools=n_pools,
        selectors=("surrogate", "random"),
        seed=seed,
        feature_steps_per_candidate=FEATURE_STEPS,
        full_run_steps=FULL_STEPS,
        n_boot=n_boot,
        expand_groups=True,
        ci_mode="family",
    )
    sur = summary[summary["selector"] == "surrogate"].iloc[0]
    return {
        "recall_at_k_mean": float(sur["recall_at_k_mean"]),
        "regret_at_k_median": float(sur["regret_at_k_median"]),
        "spearman_oof": float(spearman_safe(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())),
        "n": int(len(oof)),
        "n_folds": int(oof["family_id"].nunique()),
    }


def novelty_cv(records, *, n_folds: int = 5, seed: int = 0) -> dict:
    """Choose NN distance threshold by CV; skip if NN within thr; pred = neighbor label."""
    labeled = [r for r in records if mean_success(r) is not None]
    X, _, hashes = mb_feature_matrix(labeled)
    y = np.asarray([mean_success(r) for r in labeled], dtype=float)
    se = np.asarray(
        [seed_se(r) if seed_se(r) is not None else 0.02 for r in labeled], dtype=float
    )
    if len(X) < n_folds * 2:
        return {"EXPLORATORY": True, "error": "too_few_for_cv"}

    # Candidate thresholds: quantiles of NN distances on full set
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    nn_all = D.min(axis=1)
    cand_thr = np.unique(np.quantile(nn_all, [0.05, 0.1, 0.2, 0.3, 0.5]))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_thr = float(cand_thr[0])
    best_score = float("inf")
    cv_rows = []
    for thr in cand_thr:
        errs = []
        skips = []
        for tr, te in kf.split(X):
            # For each test point, NN among train
            Dt = pairwise_distances(X[te], X[tr])
            nn_i = Dt.argmin(axis=1)
            nn_d = Dt.min(axis=1)
            skip = nn_d <= thr
            skips.append(float(np.mean(skip)))
            if not np.any(skip):
                continue
            pred = y[tr][nn_i]
            err = np.abs(pred[skip] - y[te][skip]) / np.maximum(se[te][skip], 1e-6)
            errs.append(float(np.mean(err)))
        mean_err = float(np.mean(errs)) if errs else float("inf")
        mean_skip = float(np.mean(skips)) if skips else 0.0
        cv_rows.append({"thr": float(thr), "mean_abs_err_se": mean_err, "skip_rate": mean_skip})
        # Prefer thresholds with skip_rate in [0.05, 0.6] and low error
        if 0.05 <= mean_skip <= 0.6 and mean_err < best_score:
            best_score = mean_err
            best_thr = float(thr)
    if not np.isfinite(best_score):
        # fallback: minimize error among any
        for row in cv_rows:
            if row["mean_abs_err_se"] < best_score:
                best_score = row["mean_abs_err_se"]
                best_thr = row["thr"]

    # Final fit: leave-one-out style skip stats at best_thr
    Dt = pairwise_distances(X)
    np.fill_diagonal(Dt, np.inf)
    nn_i = Dt.argmin(axis=1)
    nn_d = Dt.min(axis=1)
    skip = nn_d <= best_thr
    pred = y[nn_i]
    err_se = np.abs(pred[skip] - y[skip]) / np.maximum(se[skip], 1e-6) if np.any(skip) else np.asarray([])
    return {
        "EXPLORATORY": True,
        "threshold_cv": best_thr,
        "cv_grid": cv_rows,
        "skip_rate": float(np.mean(skip)),
        "n_skipped": int(np.sum(skip)),
        "label_error_mean_se_units": float(np.mean(err_se)) if err_se.size else float("nan"),
        "label_error_median_se_units": float(np.median(err_se)) if err_se.size else float("nan"),
    }


def eps_recall_block(df: pd.DataFrame, oof: pd.DataFrame, *, n_draws: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    m = oof.copy()
    if "seed_se" not in m.columns:
        se_map = df.set_index("candidate_id")["seed_se"].to_dict()
        m["seed_se"] = m["candidate_id"].map(se_map)
    y = m["y_true"].to_numpy(dtype=float)
    se = np.maximum(
        pd.to_numeric(m["seed_se"], errors="coerce").fillna(0.02).to_numpy(dtype=float),
        1e-6,
    )
    pred = m["y_pred"].to_numpy(dtype=float)
    if len(y) < N_POOL:
        return {"EXPLORATORY": True, "error": "too_few"}
    eps_s = []
    perfect = []
    for _ in range(n_draws):
        idx = rng.choice(len(y), size=N_POOL, replace=False)
        yp = y[idx]
        sp = se[idx]
        pp = pred[idx]
        best = int(np.argmax(yp))
        kept = np.argsort(-pp)[:K]
        thr = max(1.5 * float(sp[best]), 0.03)
        eps_s.append(float(any(abs(yp[i] - yp[best]) <= thr for i in kept)))
        # perfect scorer on noisy resample
        noisy = rng.normal(yp, sp)
        kept_p = np.argsort(-noisy)[:K]
        perfect.append(float(best in set(kept_p.tolist())))
    return {
        "EXPLORATORY": True,
        "eps_recall_at_5_surrogate": float(np.mean(eps_s)),
        "perfect_scorer_recall_under_noise": float(np.mean(perfect)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pools", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=400)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out = ROOT / "experiments" / "gate2c_harness"
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"Refusing non-empty {out}; pass --force")
    out.mkdir(parents=True, exist_ok=True)

    records = load_all_records(RUN_IDS)
    cols = feature_columns("early+math+behavioral")
    df = records_to_frame(records, cols, labeled_only=True)
    df.to_csv(out / "labeled_raw.csv", index=False)

    # Leakage census
    hash_counts = df.groupby("code_hash").size()
    census = {
        "n_labeled_rows": int(len(df)),
        "n_unique_code_hash": int(df["code_hash"].nunique()),
        "n_hashes_with_multi_rows": int((hash_counts > 1).sum()),
        "EXPLORATORY_NOTE": "Gate2 lineage-only folds allowed same code_hash in train and test across runs",
    }
    (out / "leakage_census.json").write_text(json.dumps(census, indent=2), encoding="utf-8")

    deduped = dedupe_by_code_hash(df)
    deduped.to_csv(out / "labeled_deduped.csv", index=False)

    # Sanity: LOLO by code_hash has no leak
    for h in deduped["code_hash"].unique():
        tr = set(deduped.loc[deduped["code_hash"] != h, "code_hash"])
        te = {h}
        assert_no_code_hash_leak(tr, te)

    print("LOLO on deduped code_hash folds…")
    oof = lolo_oof(
        deduped, cols, fold_col="code_hash", use_prior=False, enforce_no_hash_leak=True
    )
    oof.to_csv(out / "oof_deduped_codehash.csv", index=False)

    report: dict = {
        "EXPLORATORY": True,
        "gate2_verdict_unchanged": "FAIL",
        "leakage_census": census,
        "pooled_deduped": pool_metrics_block(
            oof, n_pools=args.n_pools, n_boot=args.n_boot, seed=101
        ),
    }

    # gen0 vs plateau (gens 1-2)
    for name, mask in (
        ("gen0_pools", oof["generation"] == 0),
        ("plateau_pools", oof["generation"] >= 1),
    ):
        sub = oof.loc[mask].copy()
        # Need family_id for simulator — already set
        print(f"  {name}: n={len(sub)}")
        report[name] = pool_metrics_block(
            sub, n_pools=args.n_pools, n_boot=args.n_boot, seed=102
        )
        if len(sub) >= 4:
            report[name]["spearman"] = float(
                spearman_safe(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
            )

    print("novelty CV…")
    report["novelty_skip"] = novelty_cv(records, n_folds=5, seed=7)

    print("eps-recall…")
    report["noise_aware_metrics"] = eps_recall_block(deduped, oof, n_draws=1000, seed=9)

    report["seeds_for_run_examples"] = {
        rid: list(__import__("reward_triage.gate2_harness", fromlist=["seeds_for_run"]).seeds_for_run(rid))
        for rid in RUN_IDS
    }

    (out / "harness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
