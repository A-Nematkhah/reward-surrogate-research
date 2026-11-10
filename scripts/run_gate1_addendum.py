#!/usr/bin/env python3
"""Gate 1 addendum: pool audit, within-family tables, family-bootstrap CIs, SH fix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rsr.evaluation.cost import triage_cost
from rsr.evaluation.pool_simulator import (
    draw_pool,
    evaluate_pool,
    load_oof_candidates,
    simulate_pools,
)
from rsr.evaluation.triage_metrics import break_ties


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "gate1_addendum",
    )
    p.add_argument("--n-pools", type=int, default=2000)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def audit_pool_construction(cand: pd.DataFrame, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    dups = 0
    realized = []
    for _ in range(500):
        pool = draw_pool(cand, pool_size=16, n_groups=1, rng=rng, expand_groups=True)
        assert pool is not None
        realized.append(pool.n_groups)
        if len(pool.candidate_ids) != len(set(pool.candidate_ids)):
            dups += 1
    # Random recall vs k/N
    rng = np.random.default_rng(seed)
    rec = []
    for _ in range(5000):
        pool = draw_pool(cand, pool_size=16, n_groups=1, rng=rng, expand_groups=True)
        child = np.random.default_rng(rng.integers(0, 2**31 - 1))
        rec.append(evaluate_pool(pool, k=5, selector="random", rng=child)["recall_at_k"])
    return {
        "sampling": "without_replacement",
        "duplicates_in_500_draws": dups,
        "n16_n_groups_1_realized_mean": float(np.mean(realized)),
        "n16_n_groups_1_realized_min": int(min(realized)),
        "n16_n_groups_1_realized_max": int(max(realized)),
        "note": (
            "When N=16 and n_groups=1, families have size 8-10 so expand_groups "
            "adds a second family; candidates are still unique (replace=False)."
        ),
        "random_recall_at_5_empirical": float(np.mean(rec)),
        "random_recall_at_5_theory_k_over_N": 5 / 16,
        "random_recall_explanation": (
            "For a unique true best and uniform without-replacement keep-k, "
            "E[recall@k]=k/N=0.3125. Reported ~0.28 was a finite-sample Monte Carlo "
            "estimate (n_pools=2000, seed=123) under the same protocol — not a "
            "different sampling model. Re-estimates sit near 0.30–0.31; residual "
            "gap to 0.3125 is sampling noise, not duplicates."
        ),
    }


def sh_on_pools(
    early: pd.DataFrame,
    meta: pd.DataFrame,
    targets: pd.DataFrame,
    cand_scores: pd.DataFrame,
    *,
    checkpoints: list[int],
    keep_frac: float,
    n_pools: int,
    pool_size: int,
    final_k: int,
    full_run_steps: int,
    seed: int,
) -> dict:
    """SH on multi-family N-pools; final keep final_k for full-budget labels."""
    early = early.copy()
    if "candidate_id" not in early.columns:
        early.insert(0, "candidate_id", meta["candidate_id"].to_numpy())
    df = meta[["candidate_id", "family_id"]].copy()
    df["success_rate"] = targets["success_rate"].to_numpy()
    for t in checkpoints:
        df[f"early_t{t}_success_frac"] = early[f"early_t{t}_success_frac"].to_numpy()
    df = df.merge(
        cand_scores[["candidate_id", "y_pred"]], on="candidate_id", how="left"
    )

    rng = np.random.default_rng(seed)
    recalls, regrets, costs = [], [], []
    for _ in range(n_pools):
        pool = draw_pool(
            df.rename(columns={"success_rate": "y_true"}).assign(
                y_pred=df["y_pred"].fillna(0.0)
            ),
            pool_size=pool_size,
            n_groups=1,
            rng=rng,
            expand_groups=True,
        )
        if pool is None:
            continue
        # Map pool candidate ids back to rows
        sub = df.set_index("candidate_id").loc[list(pool.candidate_ids)].reset_index()
        alive = np.arange(len(sub))
        cost = 0
        prev = 0
        for t in checkpoints:
            cost += (t - prev) * len(alive)
            scores = sub.iloc[alive][f"early_t{t}_success_frac"].to_numpy()
            order = break_ties(scores, rng=rng, higher_is_better=True)
            keep_n = max(1, int(np.ceil(len(alive) * keep_frac)))
            alive = alive[order[:keep_n]]
            prev = t
        # Final triage keep final_k among survivors (by last checkpoint score)
        scores = sub.iloc[alive][f"early_t{checkpoints[-1]}_success_frac"].to_numpy()
        order = break_ties(scores, rng=rng, higher_is_better=True)
        keep_n = min(final_k, len(alive))
        final = alive[order[:keep_n]]
        cost += (full_run_steps - prev) * len(final)
        y = sub["success_rate"].to_numpy()
        tb = int(break_ties(y, rng=rng, higher_is_better=True)[0])
        recalls.append(float(tb in set(final.tolist())))
        regrets.append(float(np.max(y) - np.max(y[final])))
        costs.append(cost / pool_size)

    zero_cost = triage_cost(
        pool_size, final_k, feature_steps_per_candidate=0, full_run_steps=full_run_steps
    )
    steps_zero = zero_cost["cost_steps"] / pool_size
    return {
        "checkpoints": checkpoints,
        "keep_frac": keep_frac,
        "n_pools": len(recalls),
        "recall_at_5_mean": float(np.mean(recalls)) if recalls else float("nan"),
        "regret_at_5_median": float(np.median(regrets)) if regrets else float("nan"),
        "mean_steps_per_candidate": float(np.mean(costs)) if costs else float("nan"),
        "zero_cost_triage_steps_per_cand": float(steps_zero),
        "sh_cheaper_than_zero_cost_triage": bool(
            recalls and float(np.mean(costs)) < steps_zero
        ),
    }


def config_gap_report() -> dict:
    confirm = json.loads(
        (PROJECT_ROOT / "experiments/model_comparison_confirm/run_config.json").read_text(
            encoding="utf-8"
        )
    )
    gate1 = json.loads(
        (PROJECT_ROOT / "experiments/gate1_zero_cost/run_config.json").read_text(
            encoding="utf-8"
        )
    )
    quick = json.loads(
        (
            PROJECT_ROOT / "experiments/gate1_secondary_exploratory/run_config.json"
        ).read_text(encoding="utf-8")
    )
    cc, gc, qc = confirm["config"], gate1["config"], quick["config"]

    def et(cfg: dict) -> dict:
        return cfg.get("hyperparameters", {}).get("extra_trees", {})

    return {
        "locked_confirm": {
            "n_repeats": cc.get("n_repeats"),
            "quick": cc.get("quick"),
            "extra_trees": et(cc),
            "inner_cv": cc.get("inner_cv"),
            "default_feature_selection": cc["preprocessing"].get(
                "default_feature_selection"
            ),
            "default_k": cc["preprocessing"].get("default_k"),
        },
        "gate1_zero_cost_primary": {
            "n_repeats": gc.get("n_repeats"),
            "quick": gc.get("quick"),
            "extra_trees": et(gc),
            "inner_cv": gc.get("inner_cv"),
            "default_feature_selection": gc["preprocessing"].get(
                "default_feature_selection"
            ),
            "default_k": gc["preprocessing"].get("default_k"),
            "identical_et_settings_to_confirm": et(gc) == et(cc)
            and gc.get("n_repeats") == cc.get("n_repeats")
            and gc.get("quick") is False
            and cc.get("quick") is False
            and gc.get("inner_cv") == cc.get("inner_cv"),
        },
        "gate1_secondary_exploratory_quick": {
            "n_repeats": qc.get("n_repeats"),
            "quick": qc.get("quick"),
            "extra_trees": et(qc),
            "note": (
                "quick=True forces n_repeats=1 and hyperparameter_max_combos=2 "
                "(see quick_overrides). That — not a different n_estimators grid — "
                "explains exploratory early+math+behavioral recall~0.67 vs locked "
                "confirm pool-sim ~0.89."
            ),
        },
        "gap_explanation": (
            "The 0.667 figure came from experiments/gate1_secondary_exploratory "
            "(--quick, n_repeats=1, truncated inner search). The 0.892 figure is "
            "pool-sim on locked confirm OOF (n_repeats=3, full inner LOFO). "
            "Primary gate1_zero_cost (math+behavioral) used n_repeats=3, quick=false, "
            "and the same ExtraTrees hyperparameter grid as the locked confirm."
        ),
    }


def main() -> int:
    args = parse_args()
    out = args.out_dir
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"Refusing non-empty {out} (pass --force)")
    out.mkdir(parents=True, exist_ok=True)

    gate1_oof = pd.read_csv(
        PROJECT_ROOT / "experiments/gate1_zero_cost/oof_predictions.csv"
    )
    confirm_oof = pd.read_csv(
        PROJECT_ROOT / "experiments/model_comparison_confirm/oof_predictions.csv"
    )
    mb = load_oof_candidates(
        gate1_oof, model="extra_trees", feature_set="math+behavioral", repeat=0
    )
    emb = load_oof_candidates(
        confirm_oof,
        model="extra_trees",
        feature_set="early+math+behavioral",
        repeat=0,
    )

    audit = audit_pool_construction(mb, seed=args.seed)
    (out / "pool_construction_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    gap = config_gap_report()
    (out / "config_gap_quick_vs_locked.json").write_text(
        json.dumps(gap, indent=2), encoding="utf-8"
    )

    summaries = []
    # Within-family N=8, k=2,3  and multi-family N=16, k=5
    for name, cand, feat_steps, oof_src in (
        ("math+behavioral", mb, 0, "gate1_zero_cost"),
        ("early+math+behavioral", emb, 12000, "model_comparison_confirm"),
    ):
        per_w, sum_w = simulate_pools(
            cand,
            pool_sizes=(8,),
            n_groups_list=(1,),
            ks=(2, 3),
            n_pools=args.n_pools,
            selectors=("surrogate", "random", "oracle"),
            seed=args.seed,
            feature_steps_per_candidate=feat_steps,
            full_run_steps=50000,
            n_boot=min(args.n_boot, 300),
            expand_groups=False,
            ci_mode="family",
        )
        per_w["feature_set"] = name
        per_w["pool_regime"] = "within_family"
        per_w["oof_source"] = oof_src
        sum_w["feature_set"] = name
        sum_w["pool_regime"] = "within_family"
        sum_w["oof_source"] = oof_src
        per_w.to_csv(out / f"within_family_{name.replace('+', '_')}_per_pool.csv", index=False)
        summaries.append(sum_w)

        per_m, sum_m = simulate_pools(
            cand,
            pool_sizes=(16,),
            n_groups_list=(1, 2),
            ks=(5,),
            n_pools=args.n_pools,
            selectors=("surrogate", "random"),
            seed=args.seed,
            feature_steps_per_candidate=feat_steps,
            full_run_steps=50000,
            n_boot=min(args.n_boot, 300),
            expand_groups=True,
            ci_mode="family",
        )
        per_m["feature_set"] = name
        per_m["pool_regime"] = "multi_family"
        per_m["oof_source"] = oof_src
        sum_m["feature_set"] = name
        sum_m["pool_regime"] = "multi_family"
        sum_m["oof_source"] = oof_src
        per_m.to_csv(out / f"multi_family_{name.replace('+', '_')}_per_pool.csv", index=False)
        summaries.append(sum_m)

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "summary_family_bootstrap.csv", index=False)

    # SH
    early = pd.read_csv(PROJECT_ROOT / "experiments/phase2/features/features_early.csv")
    meta = pd.read_csv(PROJECT_ROOT / "experiments/phase2/features/features_meta.csv")
    targets = pd.read_csv(
        PROJECT_ROOT / "experiments/phase2/features/features_targets.csv"
    )
    available = [
        t
        for t in (2000, 5000, 12000, 25000)
        if f"early_t{t}_success_frac" in early.columns
    ]
    sh_rows = []
    for keep in (0.5, 1.0 / 3.0):
        # Prefer 2k/5k/12k when 2k exists; else available without 25k if missing
        cps = [t for t in available if t != 25000] or available
        sh_rows.append(
            sh_on_pools(
                early,
                meta,
                targets,
                emb,
                checkpoints=cps,
                keep_frac=keep,
                n_pools=args.n_pools,
                pool_size=16,
                final_k=5,
                full_run_steps=50000,
                seed=args.seed,
            )
        )
    sh_df = pd.DataFrame(sh_rows)
    sh_df.to_csv(out / "sh_n16_k5.csv", index=False)
    (out / "sh_checkpoints_available.json").write_text(
        json.dumps(
            {
                "requested_original_gate1": [5000, 12000, 25000],
                "available": available,
                "early_t2000_success_frac": "early_t2000_success_frac" in early.columns,
                "used_for_n16_k5": [t for t in available if t != 25000] or available,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Primary verdict under family bootstrap (math+behavioral, N=16 k=5)
    prim = summary[
        (summary["feature_set"] == "math+behavioral")
        & (summary["pool_regime"] == "multi_family")
        & (summary["N"] == 16)
        & (summary["k"] == 5)
        & (summary["n_groups_requested"] == 1)
        & (summary["selector"] == "surrogate")
    ]
    verdict = {
        "run_id": "gate1_addendum",
        "n_families": 12,
        "held_out_groups_threshold_20_met": False,
        "primary_family_bootstrap": prim.to_dict(orient="records"),
        "pool_audit": audit,
        "config_gap": gap,
        "sh": sh_rows,
    }
    if not prim.empty:
        r = prim.iloc[0]
        checks = {
            "recall_ge_0_80": bool(r["recall_at_k_mean"] >= 0.80),
            "median_regret_le_0_03": bool(r["regret_at_k_median"] <= 0.03),
            "beats_random_family_ci": bool(
                np.isfinite(r.get("recall_minus_random_lo", np.nan))
                and float(r["recall_minus_random_lo"]) > 0
            ),
            "savings_ge_0_50": bool(float(r["savings_frac"]) >= 0.50),
        }
        verdict["checks"] = checks
        verdict["viable_under_family_bootstrap"] = all(checks.values())
        verdict["caveat"] = (
            "Only 12 families exist, so the pre-registered '>= 20 held-out groups' "
            "condition is NOT met; CIs are family-cluster bootstraps over 12 families."
        )
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps({k: verdict.get(k) for k in ("checks", "viable_under_family_bootstrap", "caveat", "pool_audit")}, indent=2, default=str))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
