#!/usr/bin/env python3
"""Gate 1 post-processing: SH offline, pool sims per feature set, frontier plot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rsr.evaluation.cost import triage_cost
from rsr.evaluation.pool_simulator import load_oof_candidates, simulate_pools
from rsr.evaluation.triage_metrics import break_ties


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--oof",
        type=Path,
        default=PROJECT_ROOT / "experiments/gate1_zero_cost/oof_predictions.csv",
    )
    p.add_argument(
        "--early-csv",
        type=Path,
        default=PROJECT_ROOT / "experiments/phase2/features/features_early.csv",
    )
    p.add_argument(
        "--meta-csv",
        type=Path,
        default=PROJECT_ROOT / "experiments/phase2/features/features_meta.csv",
    )
    p.add_argument(
        "--targets-csv",
        type=Path,
        default=PROJECT_ROOT / "experiments/phase2/features/features_targets.csv",
    )
    p.add_argument("--n-pools", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--full-run-steps", type=int, default=50000)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results/gate1_zero_cost",
    )
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def successive_halving(
    early: pd.DataFrame,
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    checkpoints: list[int],
    keep_fracs: list[float],
    full_run_steps: int,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Offline SH using early_t{steps}_success_frac as noisy ranks.
    Final label = full-run success_rate. Cost counts every step spent.
    """
    if len(meta) != len(targets) or len(meta) != len(early):
        raise ValueError(
            f"Row mismatch meta={len(meta)} targets={len(targets)} early={len(early)}"
        )
    df = meta[["candidate_id", "family_id"]].copy()
    df["success_rate"] = targets["success_rate"].to_numpy()
    for t in checkpoints:
        col = f"early_t{t}_success_frac"
        if col not in early.columns:
            raise KeyError(col)
        df[col] = early[col].to_numpy()

    rng = np.random.default_rng(seed)
    rows = []
    for keep_frac in keep_fracs:
        # Simulate many random generation pools of size 16 within single families
        # when possible; else from all candidates (documented).
        families = sorted(df["family_id"].unique())
        recalls = []
        regrets = []
        costs = []
        for fam in families:
            sub = df[df["family_id"] == fam].reset_index(drop=True)
            if len(sub) < 4:
                continue
            # Use the whole family as a mini-pool (2D families have ~8–10)
            n = len(sub)
            alive = np.arange(n)
            cost = 0
            prev_t = 0
            for t in checkpoints:
                step_cost = (t - prev_t) * len(alive)
                cost += step_cost
                scores = sub.iloc[alive][f"early_t{t}_success_frac"].to_numpy()
                order = break_ties(scores, rng=rng, higher_is_better=True)
                keep_n = max(1, int(np.ceil(len(alive) * keep_frac)))
                alive = alive[order[:keep_n]]
                prev_t = t
            # Train survivors to full budget
            cost += (full_run_steps - prev_t) * len(alive)
            y = sub["success_rate"].to_numpy()
            true_best = int(break_ties(y, rng=rng, higher_is_better=True)[0])
            recall = float(true_best in set(alive.tolist()))
            regret = float(np.max(y) - np.max(y[alive]))
            recalls.append(recall)
            regrets.append(regret)
            costs.append(cost / n)  # per original candidate in family

        rows.append(
            {
                "method": f"SH_keep_{keep_frac:g}",
                "keep_frac": keep_frac,
                "checkpoints": ",".join(str(t) for t in checkpoints),
                "n_families": len(recalls),
                "recall_mean": float(np.mean(recalls)) if recalls else float("nan"),
                "regret_median": float(np.median(regrets)) if regrets else float("nan"),
                "mean_steps_per_candidate": float(np.mean(costs)) if costs else float("nan"),
                "note": "family-as-pool; early success_frac ranks",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(args.oof)
    feature_sets = sorted(oof["feature_set"].unique())
    models = ["extra_trees"]

    # Cost map
    cost_map = {
        "static": 0,
        "math": 0,
        "behavioral": 0,
        "math+behavioral": 0,
        "early": 12000,
        "early+math": 12000,
        "early+math+behavioral": 12000,
    }

    sim_rows = []
    for fs in feature_sets:
        for model in models:
            try:
                cand = load_oof_candidates(
                    oof, model=model, feature_set=fs, target="success_rate", repeat=0
                )
            except ValueError:
                continue
            per, summary = simulate_pools(
                cand,
                pool_sizes=(16,),
                n_groups_list=(1,),
                ks=(5,),
                n_pools=args.n_pools,
                selectors=("surrogate", "random", "oracle"),
                seed=args.seed,
                feature_steps_per_candidate=int(cost_map.get(fs, 12000)),
                full_run_steps=args.full_run_steps,
                n_boot=args.n_boot,
            )
            sub = summary[summary["selector"] == "surrogate"].copy()
            sub["feature_set"] = fs
            sub["model"] = model
            sub["feature_steps"] = cost_map.get(fs, 12000)
            sim_rows.append(sub)

    sim = pd.concat(sim_rows, ignore_index=True) if sim_rows else pd.DataFrame()
    sim_path = out / "pool_sim_by_feature_set.csv"
    sim.to_csv(sim_path, index=False)

    # Successive halving VERIFY checkpoints
    early = pd.read_csv(args.early_csv)
    meta = pd.read_csv(args.meta_csv)
    targets = pd.read_csv(args.targets_csv)
    available = []
    for t in (5000, 12000, 25000):
        if f"early_t{t}_success_frac" in early.columns:
            available.append(t)
    sh_info = {"requested": [5000, 12000, 25000], "available": available}
    if available:
        # Feature CSVs are row-aligned with features_meta.csv (no candidate_id column).
        if "candidate_id" not in early.columns:
            if len(early) != len(meta):
                raise SystemExit(
                    f"early rows ({len(early)}) != meta rows ({len(meta)}); cannot align"
                )
            early_m = early.copy()
            early_m.insert(0, "candidate_id", meta["candidate_id"].to_numpy())
        else:
            early_m = meta[["candidate_id"]].merge(early, on="candidate_id", how="left")
        sh = successive_halving(
            early_m,
            targets,
            meta,
            checkpoints=available,
            keep_fracs=[0.5, 1.0 / 3.0],
            full_run_steps=args.full_run_steps,
            seed=args.seed,
        )
        sh.to_csv(out / "successive_halving.csv", index=False)
        # Add SH points to frontier using mean_steps_per_candidate as x
        for _, r in sh.iterrows():
            sim = pd.concat(
                [
                    sim,
                    pd.DataFrame(
                        [
                            {
                                "N": 16,
                                "k": 5,
                                "n_groups_requested": 1,
                                "selector": "SH",
                                "feature_set": r["method"],
                                "model": "SH",
                                "feature_steps": r["mean_steps_per_candidate"],
                                "recall_at_k_mean": r["recall_mean"],
                                "regret_at_k_median": r["regret_median"],
                                "savings_frac": float("nan"),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    else:
        sh = pd.DataFrame()
    (out / "sh_checkpoint_verify.json").write_text(
        json.dumps(sh_info, indent=2), encoding="utf-8"
    )

    # Frontier plot
    fig, ax = plt.subplots(figsize=(8, 5))
    if not sim.empty:
        plot_df = sim[sim["selector"].isin(["surrogate", "SH"])].copy()
        for _, r in plot_df.iterrows():
            x = float(r["feature_steps"])
            # For surrogate, total steps per cand for N=16 k=5 = feature + full*(k/N)
            if r["selector"] == "surrogate":
                cost = triage_cost(
                    16,
                    5,
                    feature_steps_per_candidate=int(r["feature_steps"]),
                    full_run_steps=args.full_run_steps,
                )
                x = cost["cost_steps"] / 16.0
            y = float(r["recall_at_k_mean"])
            ax.scatter(x, y, s=60)
            ax.annotate(
                str(r["feature_set"]),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    ax.set_xlabel("PPO steps per candidate (features + kept full runs, N=16 k=5)")
    ax.set_ylabel("recall@5 (single-group pools)")
    ax.set_title("Gate 1 cost–accuracy frontier")
    ax.axhline(0.80, color="gray", ls="--", lw=1, label="recall threshold 0.80")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig_path = out / "figures" / "cost_accuracy_frontier.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    # Verdict helpers for primary
    primary = sim[
        (sim["feature_set"] == "math+behavioral")
        & (sim["selector"] == "surrogate")
        & (sim["model"] == "extra_trees")
    ]
    verdict = {"primary_rows": primary.to_dict(orient="records"), "sh_checkpoints": sh_info}
    if not primary.empty:
        row = primary.iloc[0]
        cost = triage_cost(
            16, 5, feature_steps_per_candidate=0, full_run_steps=args.full_run_steps
        )
        delta_lo = row.get("recall_minus_random_lo", float("nan"))
        verdict["checks"] = {
            "recall_ge_0_80": bool(row["recall_at_k_mean"] >= 0.80),
            "median_regret_le_0_03": bool(row["regret_at_k_median"] <= 0.03),
            "beats_random_ci": bool(
                np.isfinite(delta_lo) and float(delta_lo) > 0
            ),
            "savings_ge_0_50": bool(cost["savings_frac"] >= 0.50),
            "recall_at_k_mean": float(row["recall_at_k_mean"]),
            "regret_at_k_median": float(row["regret_at_k_median"]),
            "recall_minus_random_lo": float(delta_lo) if np.isfinite(delta_lo) else None,
            "savings_frac": float(cost["savings_frac"]),
        }
        verdict["viable"] = all(
            [
                verdict["checks"]["recall_ge_0_80"],
                verdict["checks"]["median_regret_le_0_03"],
                verdict["checks"]["beats_random_ci"],
                verdict["checks"]["savings_ge_0_50"],
            ]
        )
    (out / "verdict_primary.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict.get("checks", verdict), indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
