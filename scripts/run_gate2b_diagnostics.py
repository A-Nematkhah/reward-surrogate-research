"""Phase 6b — EXPLORATORY diagnostics on Gate-2 live stores (no PPO, no verdict change)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_triage.gate2_harness import (  # noqa: E402
    RUN_IDS,
    dedupe_by_code_hash,
    feature_columns,
    load_all_records,
    lolo_oof,
    mb_feature_matrix,
    records_to_frame,
    savings_restart,
    savings_resume,
)
from reward_triage.store import StoreRecord  # noqa: E402
from rsr.evaluation.cost import triage_cost  # noqa: E402
from rsr.evaluation.pool_simulator import simulate_pools  # noqa: E402
from rsr.evaluation.triage_metrics import recall_at_k, regret_at_k  # noqa: E402

N_POOL = 16
K = 5
E_RATIO = 12_000 / 50_000
FEATURE_STEPS = 12_000
FULL_STEPS = 50_000
EXPLORATORY = True


def _tag(obj: dict) -> dict:
    out = dict(obj)
    out["EXPLORATORY"] = True
    return out


def run_pool(
    oof: pd.DataFrame,
    *,
    n_pools: int,
    n_boot: int,
    seed: int,
) -> dict:
    per, summary = simulate_pools(
        oof,
        pool_sizes=(N_POOL,),
        n_groups_list=(1,),
        ks=(K,),
        n_pools=n_pools,
        selectors=("surrogate", "random", "oracle"),
        seed=seed,
        feature_steps_per_candidate=FEATURE_STEPS,
        full_run_steps=FULL_STEPS,
        n_boot=n_boot,
        expand_groups=True,
        ci_mode="family",
    )
    sur = summary[summary["selector"] == "surrogate"].iloc[0].to_dict()
    return _tag(
        {
            "recall_at_k_mean": float(sur["recall_at_k_mean"]),
            "regret_at_k_median": float(sur["regret_at_k_median"]),
            "savings_frac_formula_restart": float(sur["savings_frac"]),
            "savings_frac_resume": savings_resume(E_RATIO, K, N_POOL),
            "n_oof": int(len(oof)),
            "n_folds": int(oof["family_id"].nunique()),
            "n_pools_drawn": int(sur.get("n_pools", n_pools)),
        }
    )


def label_noise_ceiling(df: pd.DataFrame, *, n_draws: int = 1000, seed: int = 0) -> dict:
    """Resample labels within SE; perfect-scorer recall and eps-recall."""
    rng = np.random.default_rng(seed)
    # Need rows with SE
    sub = df.dropna(subset=["y_true", "seed_se"]).copy()
    if len(sub) < N_POOL:
        return _tag({"error": "insufficient_rows_with_se", "n": int(len(sub))})

    y = sub["y_true"].to_numpy(dtype=float)
    se = np.maximum(sub["seed_se"].to_numpy(dtype=float), 1e-6)
    obs_best_is_true = []
    perfect_recall = []
    eps_recall = []
    eps = np.maximum(1.5 * se, 0.03)

    for _ in range(n_draws):
        # latent true ~ N(observed, SE^2) then observe again — ceiling for ranking noise
        latent = rng.normal(y, se)
        # observed-with-noise draw for scoring oracle on noisy labels
        noisy = rng.normal(latent, se)
        idx = rng.choice(len(y), size=N_POOL, replace=False)
        lat_p = latent[idx]
        noi_p = noisy[idx]
        eps_p = eps[idx]
        true_best = int(np.argmax(lat_p))
        obs_best = int(np.argmax(noi_p))
        obs_best_is_true.append(float(true_best == obs_best))
        # perfect scorer on noisy labels keeps top-k by noisy
        order = np.argsort(-noi_p)[:K]
        perfect_recall.append(float(true_best in set(order.tolist())))
        # eps-recall: kept contains someone within eps of latent best
        best_lat = float(lat_p[true_best])
        kept_ok = any(abs(lat_p[i] - best_lat) <= float(eps_p[true_best]) for i in order)
        # also: kept contains a candidate within 1.5 SE or 0.03 of pool best (latent)
        eps_recall.append(float(kept_ok))

    return _tag(
        {
            "n_draws": n_draws,
            "frac_observed_best_is_true_best": float(np.mean(obs_best_is_true)),
            "perfect_scorer_recall_at_5": float(np.mean(perfect_recall)),
            "eps_recall_at_5": float(np.mean(eps_recall)),
            "eps_rule": "within max(1.5*SE, 0.03) of latent pool best",
            "median_seed_se": float(np.median(se)),
        }
    )


def corpus_diversity(records: list[StoreRecord], labeled: pd.DataFrame) -> dict:
    hashes = [r.candidate.code_hash for r in records]
    uniq = len(set(hashes))
    # near-duplicates in mb space
    X, _, hs = mb_feature_matrix([r for r in records if r.labels])
    near = 0
    thr = 1e-6
    if len(X) >= 2:
        # pairwise min distance excluding self
        from sklearn.metrics import pairwise_distances

        D = pairwise_distances(X)
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
        # scale-free: near-dup if NN distance < 1e-3 of feature-scale median
        med = float(np.median(D[np.isfinite(D)])) if np.isfinite(D).any() else 1.0
        thr = max(1e-6, 0.01 * med)
        near = int((nn < thr).sum())
    # templates from llm_meta family if present
    families = []
    for r in records:
        fam = (r.candidate.llm_meta or {}).get("family")
        if fam:
            families.append(str(fam))
    spread = {}
    for (run_id, gen), g in labeled.groupby(["run_id", "generation"]):
        ys = g["y_true"].to_numpy(dtype=float)
        spread[f"{run_id}_g{int(gen)}"] = {
            "n": int(len(ys)),
            "std": float(np.std(ys)),
            "range": float(np.max(ys) - np.min(ys)) if len(ys) else float("nan"),
            "mean": float(np.mean(ys)),
        }
    ys_all = labeled["y_true"].to_numpy(dtype=float)
    ceiling = float(np.quantile(ys_all, 0.95)) if len(ys_all) else float("nan")
    near_ceil = float(np.mean(ys_all >= ceiling - 0.02)) if len(ys_all) else float("nan")
    return _tag(
        {
            "n_records": len(records),
            "n_unique_code_hash": uniq,
            "duplicate_code_hash_rate": 1.0 - uniq / max(len(records), 1),
            "distinct_llm_families": sorted(set(families)),
            "n_distinct_llm_families": len(set(families)),
            "mb_near_duplicate_count": near,
            "mb_near_duplicate_rate": near / max(len(X), 1),
            "mb_nn_threshold": thr,
            "label_spread_by_run_gen": spread,
            "share_labels_near_ceiling": near_ceil,
            "ceiling_ref_q95": ceiling,
        }
    )


def selection_bias(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_pools: int,
    n_boot: int,
) -> dict:
    gen0 = df[df["generation"] == 0].copy()
    g12 = df[df["generation"] >= 1].copy()
    out: dict = {"EXPLORATORY": True}
    if len(gen0) >= N_POOL:
        oof0 = lolo_oof(gen0, feature_cols, fold_col="lineage_id", use_prior=False)
        # family_id from lineage — may still leak code_hash across runs; noted
        out["gen0_only"] = run_pool(oof0, n_pools=n_pools, n_boot=n_boot, seed=11)
        out["gen0_only"]["note"] = "all labeled warm-up; fold=lineage (may duplicate codes across runs)"
    if len(g12) >= 8:
        p = np.clip(g12["probability"].to_numpy(dtype=float), 1e-3, 1.0)
        w = 1.0 / p
        oof12 = lolo_oof(
            g12,
            feature_cols,
            fold_col="lineage_id",
            use_prior=False,
            sample_weight=w,
        )
        out["gens_1_2_ipw"] = run_pool(oof12, n_pools=n_pools, n_boot=n_boot, seed=12)
        out["gens_1_2_ipw"]["note"] = "IPW sample_weight=1/selection_probability"
        out["gens_1_2_ipw"]["n"] = int(len(g12))
    return out


def realized_savings(records: list[StoreRecord]) -> dict:
    by_run: dict[str, list[StoreRecord]] = {}
    for r in records:
        by_run.setdefault(r.run_id or "unknown", []).append(r)
    e = E_RATIO
    rows = []
    for run_id, recs in sorted(by_run.items()):
        n = len(recs)
        labeled = [r for r in recs if r.labels]
        k_eff = len(labeled)
        # seeds per labeled
        seed_jobs = sum(
            len([lab for lab in r.labels if lab.status == "completed"]) for r in labeled
        )
        baseline_jobs = n * 3  # config n_seeds
        full_only = 1.0 - seed_jobs / max(baseline_jobs, 1)
        rows.append(
            {
                "run_id": run_id,
                "n_candidates": n,
                "n_labeled": k_eff,
                "seed_jobs": seed_jobs,
                "baseline_seed_jobs": baseline_jobs,
                "realized_savings_full_runs_only": full_only,
                "formula_restart_k_over_n": savings_restart(e, k_eff, n),
                "formula_resume_k_over_n": savings_resume(e, k_eff, n),
            }
        )
    return _tag(
        {
            "host_loop_behavior": "RESTART — each kept candidate trains full PPO from scratch; early features are collected during that same run (no resume-from-early-checkpoint for final policy).",
            "formulas": {
                "restart": "1 - (e + k/N)",
                "resume": "(1 - k/N) * (1 - e)",
                "e": e,
                "threshold_eval_N16_k5_restart": savings_restart(e, 5, 16),
                "threshold_eval_N16_k5_resume": savings_resume(e, 5, 16),
            },
            "per_run": rows,
            "G": 3,
        }
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pools", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=400)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_dir = ROOT / "experiments" / "gate2b_diagnostics"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"Refusing non-empty {out_dir}; pass --force")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_records(RUN_IDS)
    report: dict = {"EXPLORATORY": True, "run_ids": list(RUN_IDS)}

    # --- 1) prior OFF ---
    print("1) prior OFF (early+math+behavioral)…")
    cols_emb = feature_columns("early+math+behavioral")
    df_emb = records_to_frame(records, cols_emb, labeled_only=True)
    oof_noprior = lolo_oof(
        df_emb, cols_emb, fold_col="lineage_id", use_prior=False
    )
    report["prior_off"] = run_pool(
        oof_noprior, n_pools=args.n_pools, n_boot=args.n_boot, seed=21
    )
    oof_noprior.to_csv(out_dir / "oof_prior_off.csv", index=False)

    # --- 2) ablations ---
    print("2) ablations…")
    ablations = {}
    for fs in ("math+behavioral", "early"):
        print(f"  {fs}")
        cols = feature_columns(fs)  # type: ignore[arg-type]
        df = records_to_frame(records, cols, labeled_only=True)
        oof = lolo_oof(df, cols, fold_col="lineage_id", use_prior=False, feature_set=fs)  # type: ignore[arg-type]
        metrics = run_pool(oof, n_pools=args.n_pools, n_boot=args.n_boot, seed=30)
        # zero-cost for mb: e=0
        if fs == "math+behavioral":
            metrics["savings_frac_restart_e0"] = savings_restart(0.0, K, N_POOL)
            metrics["savings_frac_resume_e0"] = savings_resume(0.0, K, N_POOL)
        else:
            metrics["savings_frac_restart"] = savings_restart(E_RATIO, K, N_POOL)
            metrics["savings_frac_resume"] = savings_resume(E_RATIO, K, N_POOL)
        ablations[fs] = metrics
        oof.to_csv(out_dir / f"oof_{fs.replace('+', '_')}.csv", index=False)
    report["ablations"] = ablations

    # --- 3) label noise ---
    print("3) label-noise ceiling…")
    report["label_noise_ceiling"] = label_noise_ceiling(df_emb, n_draws=1000, seed=42)

    # --- 4) diversity ---
    print("4) corpus diversity…")
    report["corpus_diversity"] = corpus_diversity(records, df_emb)

    # --- 5) selection bias ---
    print("5) selection-bias…")
    report["selection_bias"] = selection_bias(
        df_emb, cols_emb, n_pools=args.n_pools, n_boot=args.n_boot
    )

    # --- 6) cost ---
    print("6) cost accounting…")
    report["cost_accounting"] = realized_savings(records)

    (out_dir / "diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in report if k != "selection_bias"}, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
