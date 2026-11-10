#!/usr/bin/env python3
"""Gate 1 amendments: within-family tables, paired feature-set diffs, fair SH."""

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
)
from rsr.evaluation.triage_metrics import break_ties


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "gate1_amendments",
    )
    p.add_argument("--n-pools", type=int, default=800)
    p.add_argument("--n-boot", type=int, default=250)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _family_boot_summary(
    cand: pd.DataFrame,
    *,
    pool_size: int,
    k: int,
    expand_groups: bool,
    n_groups: int,
    n_pools: int,
    n_boot: int,
    seed: int,
    feature_steps: int,
    full_run_steps: int = 50000,
) -> dict:
    """Point estimates + family-bootstrap CIs for surrogate and random."""
    rng = np.random.default_rng(seed)
    families = np.asarray(sorted(cand["family_id"].unique()))

    def draw_batch(frame: pd.DataFrame, n: int, rng_local: np.random.Generator):
        rows = []
        for _ in range(n):
            pool = draw_pool(
                frame,
                pool_size=pool_size,
                n_groups=n_groups,
                rng=rng_local,
                expand_groups=expand_groups,
            )
            if pool is None:
                continue
            child = np.random.default_rng(rng_local.integers(0, 2**31 - 1))
            ms = evaluate_pool(pool, k=k, selector="surrogate", rng=child)
            mr = evaluate_pool(
                pool,
                k=k,
                selector="random",
                rng=np.random.default_rng(rng_local.integers(0, 2**31 - 1)),
            )
            rows.append(
                {
                    "recall_s": ms["recall_at_k"],
                    "regret_s": ms["regret_at_k"],
                    "recall_r": mr["recall_at_k"],
                    "regret_r": mr["regret_at_k"],
                    "false_reject_s": 1.0 - ms["recall_at_k"],
                    "n_groups_realized": ms["n_groups_realized"],
                }
            )
        return pd.DataFrame(rows)

    point = draw_batch(cand, n_pools, rng)
    cost = triage_cost(
        pool_size,
        k,
        feature_steps_per_candidate=feature_steps,
        full_run_steps=full_run_steps,
    )
    out = {
        "N": pool_size,
        "k": k,
        "expand_groups": expand_groups,
        "n_groups_requested": n_groups,
        "pool_label": (
            f"{int(point['n_groups_realized'].mode().iloc[0])}-family pool "
            f"(n_groups={n_groups} expanded)"
            if expand_groups and pool_size > 8
            else "within-family"
        ),
        "n_pools": int(len(point)),
        "n_families": int(len(families)),
        "held_out_groups_ge_20": False,
        "feature_steps": feature_steps,
        "savings_frac": cost["savings_frac"],
        "recall_surrogate_mean": float(point["recall_s"].mean()),
        "recall_random_mean": float(point["recall_r"].mean()),
        "recall_random_theory": float(k / pool_size),
        "regret_surrogate_mean": float(point["regret_s"].mean()),
        "regret_surrogate_median": float(point["regret_s"].median()),
        "false_reject_rate_mean": float(point["false_reject_s"].mean()),
    }

    boots = []
    pools_per_boot = 40
    for _ in range(n_boot):
        chosen = rng.choice(families, size=len(families), replace=True)
        uniq = list(dict.fromkeys(chosen.tolist()))
        sub = cand[cand["family_id"].isin(uniq)].reset_index(drop=True)
        b = draw_batch(sub, pools_per_boot, rng)
        if b.empty:
            continue
        boots.append(
            {
                "recall_s": float(b["recall_s"].mean()),
                "recall_r": float(b["recall_r"].mean()),
                "regret_mean": float(b["regret_s"].mean()),
                "regret_median": float(b["regret_s"].median()),
                "false_reject": float(b["false_reject_s"].mean()),
                "delta_recall": float(b["recall_s"].mean() - b["recall_r"].mean()),
            }
        )
    boot_df = pd.DataFrame(boots)
    if not boot_df.empty:
        for key, col in (
            ("recall_surrogate", "recall_s"),
            ("recall_random", "recall_r"),
            ("regret_mean", "regret_mean"),
            ("regret_median", "regret_median"),
            ("false_reject_rate", "false_reject"),
            ("delta_recall_vs_random", "delta_recall"),
        ):
            arr = boot_df[col].to_numpy()
            lo, hi = np.quantile(arr, [0.025, 0.975])
            out[f"{key}_lo"] = float(lo)
            out[f"{key}_hi"] = float(hi)
    return out


def paired_feature_diff(
    cand_a: pd.DataFrame,
    cand_b: pd.DataFrame,
    *,
    name_a: str,
    name_b: str,
    pool_size: int,
    k: int,
    expand_groups: bool,
    n_groups: int,
    n_pools: int,
    n_boot: int,
    seed: int,
    steps_a: int,
    steps_b: int,
) -> dict:
    """Paired family-bootstrap of recall difference (A - B)."""
    # Align on candidate_id
    merged = cand_a.merge(
        cand_b[["candidate_id", "y_pred", "y_true"]],
        on="candidate_id",
        suffixes=("_a", "_b"),
    )
    # y_true should match; use a
    base = merged.rename(
        columns={
            "y_true_a": "y_true",
            "family_id": "family_id",
        }
    )[["candidate_id", "family_id", "y_true", "y_pred_a", "y_pred_b"]]

    rng = np.random.default_rng(seed)
    families = np.asarray(sorted(base["family_id"].unique()))

    def one_pool(frame: pd.DataFrame, rng_local: np.random.Generator):
        # Build a temporary cand frame for draw_pool using y_pred_a as placeholder
        tmp = frame.rename(columns={"y_pred_a": "y_pred"})
        pool = draw_pool(
            tmp,
            pool_size=pool_size,
            n_groups=n_groups,
            rng=rng_local,
            expand_groups=expand_groups,
        )
        if pool is None:
            return None
        ids = list(pool.candidate_ids)
        sub = frame.set_index("candidate_id").loc[ids]
        y = sub["y_true"].to_numpy()
        sa = sub["y_pred_a"].to_numpy()
        sb = sub["y_pred_b"].to_numpy()
        child = np.random.default_rng(rng_local.integers(0, 2**31 - 1))
        from rsr.evaluation.pool_simulator import PoolDraw

        pa = PoolDraw(
            candidate_ids=tuple(ids),
            group_ids=tuple(sub["family_id"].astype(str)),
            y_true=y,
            y_pred=sa,
            n_groups=int(sub["family_id"].nunique()),
        )
        pb = PoolDraw(
            candidate_ids=tuple(ids),
            group_ids=tuple(sub["family_id"].astype(str)),
            y_true=y,
            y_pred=sb,
            n_groups=int(sub["family_id"].nunique()),
        )
        ra = evaluate_pool(pa, k=k, selector="surrogate", rng=child)["recall_at_k"]
        rb = evaluate_pool(
            pb,
            k=k,
            selector="surrogate",
            rng=np.random.default_rng(rng_local.integers(0, 2**31 - 1)),
        )["recall_at_k"]
        return ra - rb, ra, rb

    diffs, ra_s, rb_s = [], [], []
    for _ in range(n_pools):
        r = one_pool(base, rng)
        if r is None:
            continue
        d, ra, rb = r
        diffs.append(d)
        ra_s.append(ra)
        rb_s.append(rb)

    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(families, size=len(families), replace=True)
        uniq = list(dict.fromkeys(chosen.tolist()))
        sub = base[base["family_id"].isin(uniq)].reset_index(drop=True)
        bd = []
        for _pi in range(40):
            r = one_pool(sub, rng)
            if r is not None:
                bd.append(r[0])
        if bd:
            boots.append(float(np.mean(bd)))

    lo, hi = (float("nan"), float("nan"))
    if boots:
        lo, hi = [float(x) for x in np.quantile(boots, [0.025, 0.975])]
    return {
        "feature_set_a": name_a,
        "feature_set_b": name_b,
        "N": pool_size,
        "k": k,
        "pool_regime": (
            "2-family pool (n_groups=1 expanded to 2 families)"
            if expand_groups
            else "within-family"
        ),
        "recall_a_mean": float(np.mean(ra_s)) if ra_s else float("nan"),
        "recall_b_mean": float(np.mean(rb_s)) if rb_s else float("nan"),
        "delta_recall_a_minus_b_mean": float(np.mean(diffs)) if diffs else float("nan"),
        "delta_recall_lo": lo,
        "delta_recall_hi": hi,
        "feature_steps_a": steps_a,
        "feature_steps_b": steps_b,
        "ci_excludes_0": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
        "n_pools": len(diffs),
        "n_families": 12,
    }


def sh_fairness(
    early: pd.DataFrame,
    meta: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    checkpoints: list[int],
    keep_frac: float,
    n_pools: int,
    pool_size: int,
    full_run_steps: int,
    seed: int,
) -> dict:
    early = early.copy()
    if "candidate_id" not in early.columns:
        early.insert(0, "candidate_id", meta["candidate_id"].to_numpy())
    df = meta[["candidate_id", "family_id"]].copy()
    df["y_true"] = targets["success_rate"].to_numpy()
    df["y_pred"] = 0.0
    for t in checkpoints:
        df[f"early_t{t}_success_frac"] = early[f"early_t{t}_success_frac"].to_numpy()

    rng = np.random.default_rng(seed)
    rec_sh, rec_rand_count, rec_rand_cost = [], [], []
    regrets_sh = []
    survivor_counts = []
    costs = []

    for _ in range(n_pools):
        pool = draw_pool(
            df, pool_size=pool_size, n_groups=1, rng=rng, expand_groups=True
        )
        if pool is None:
            continue
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
        # Survivors train to full budget (no extra k=5 cut)
        final = alive
        cost += (full_run_steps - prev) * len(final)
        survivor_counts.append(int(len(final)))
        costs.append(cost)

        y = sub["y_true"].to_numpy()
        tb = int(break_ties(y, rng=rng, higher_is_better=True)[0])
        rec_sh.append(float(tb in set(final.tolist())))
        regrets_sh.append(float(np.max(y) - np.max(y[final])))

        # Random with same final count
        m = len(final)
        rand_idx = rng.choice(len(sub), size=m, replace=False)
        rec_rand_count.append(float(tb in set(rand_idx.tolist())))

        # Random equal total PPO cost: spend `cost` as feature_steps=0 full runs
        # => n_full = cost / full_run_steps, compare recall of that many random
        n_full = max(1, int(round(cost / full_run_steps)))
        n_full = min(n_full, len(sub))
        rand_cost_idx = rng.choice(len(sub), size=n_full, replace=False)
        rec_rand_cost.append(float(tb in set(rand_cost_idx.tolist())))

    return {
        "checkpoints": checkpoints,
        "keep_frac": keep_frac,
        "n_pools": len(rec_sh),
        "final_survivor_count_mean": float(np.mean(survivor_counts)),
        "final_survivor_count_median": float(np.median(survivor_counts)),
        "final_survivor_count_min": int(np.min(survivor_counts)),
        "final_survivor_count_max": int(np.max(survivor_counts)),
        "recall_sh_mean": float(np.mean(rec_sh)),
        "regret_sh_median": float(np.median(regrets_sh)),
        "recall_random_same_count_mean": float(np.mean(rec_rand_count)),
        "recall_random_equal_cost_mean": float(np.mean(rec_rand_cost)),
        "mean_total_steps": float(np.mean(costs)),
        "mean_steps_per_candidate": float(np.mean(costs) / pool_size),
        "note": (
            "SH survivors are the post-halving set (not forced to k=5). "
            "Random baselines use the same survivor count, or equal total PPO cost."
        ),
    }


def main() -> int:
    args = parse_args()
    out = args.out_dir
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"Refusing non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)

    gate1 = pd.read_csv(PROJECT_ROOT / "experiments/gate1_zero_cost/oof_predictions.csv")
    confirm = pd.read_csv(
        PROJECT_ROOT / "experiments/model_comparison_confirm/oof_predictions.csv"
    )
    mb = load_oof_candidates(
        gate1, model="extra_trees", feature_set="math+behavioral", repeat=0
    )
    emb = load_oof_candidates(
        confirm, model="extra_trees", feature_set="early+math+behavioral", repeat=0
    )

    within_rows = []
    for name, cand, steps in (
        ("math+behavioral", mb, 0),
        ("early+math+behavioral", emb, 12000),
    ):
        for k in (2, 3):
            row = _family_boot_summary(
                cand,
                pool_size=8,
                k=k,
                expand_groups=False,
                n_groups=1,
                n_pools=args.n_pools,
                n_boot=args.n_boot,
                seed=args.seed,
                feature_steps=steps,
            )
            row["feature_set"] = name
            row["pool_regime"] = "within-family"
            within_rows.append(row)
    within_df = pd.DataFrame(within_rows)
    within_df.to_csv(out / "within_family_N8.csv", index=False)

    # Also N=16 2-family for paired context
    multi_rows = []
    for name, cand, steps in (
        ("math+behavioral", mb, 0),
        ("early+math+behavioral", emb, 12000),
    ):
        row = _family_boot_summary(
            cand,
            pool_size=16,
            k=5,
            expand_groups=True,
            n_groups=1,
            n_pools=args.n_pools,
            n_boot=args.n_boot,
            seed=args.seed,
            feature_steps=steps,
        )
        row["feature_set"] = name
        row["pool_regime"] = "2-family pool (n_groups=1 expanded to 2 families)"
        multi_rows.append(row)
    multi_df = pd.DataFrame(multi_rows)
    multi_df.to_csv(out / "two_family_N16_k5.csv", index=False)

    paired = [
        paired_feature_diff(
            emb,
            mb,
            name_a="early+math+behavioral",
            name_b="math+behavioral",
            pool_size=16,
            k=5,
            expand_groups=True,
            n_groups=1,
            n_pools=args.n_pools,
            n_boot=args.n_boot,
            seed=args.seed,
            steps_a=12000,
            steps_b=0,
        ),
        paired_feature_diff(
            emb,
            mb,
            name_a="early+math+behavioral",
            name_b="math+behavioral",
            pool_size=8,
            k=3,
            expand_groups=False,
            n_groups=1,
            n_pools=args.n_pools,
            n_boot=args.n_boot,
            seed=args.seed + 1,
            steps_a=12000,
            steps_b=0,
        ),
    ]
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(out / "paired_feature_set_delta.csv", index=False)

    early = pd.read_csv(PROJECT_ROOT / "experiments/phase2/features/features_early.csv")
    meta = pd.read_csv(PROJECT_ROOT / "experiments/phase2/features/features_meta.csv")
    targets = pd.read_csv(
        PROJECT_ROOT / "experiments/phase2/features/features_targets.csv"
    )
    cps = [
        t
        for t in (2000, 5000, 12000)
        if f"early_t{t}_success_frac" in early.columns
    ]
    sh_rows = [
        sh_fairness(
            early,
            meta,
            targets,
            checkpoints=cps,
            keep_frac=frac,
            n_pools=args.n_pools,
            pool_size=16,
            full_run_steps=50000,
            seed=args.seed,
        )
        for frac in (0.5, 1.0 / 3.0)
    ]
    sh_df = pd.DataFrame(sh_rows)
    sh_df.to_csv(out / "sh_fairness_N16.csv", index=False)

    meta_out = {
        "run_id": "gate1_amendments",
        "thresholds_frozen": "docs/prereg/gate1_thresholds_frozen.md",
        "n_families": 12,
        "held_out_groups_ge_20": False,
        "pool_label_N16": "2-family pool (n_groups=1 expanded to 2 families)",
        "within_family": within_rows,
        "two_family_N16": multi_rows,
        "paired": paired,
        "sh_fairness": sh_rows,
    }
    (out / "summary.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(within_df.to_string(index=False))
    print(paired_df.to_string(index=False))
    print(sh_df.to_string(index=False))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
