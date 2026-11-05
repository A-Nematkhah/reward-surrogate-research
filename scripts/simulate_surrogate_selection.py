#!/usr/bin/env python3
"""Phase 5: offline surrogate-assisted selection regret vs PPO budget.

Uses locked confirm OOF predictions (no new training). Compares policies:
  - random
  - surrogate top-k (ExtraTrees scores)
  - oracle (upper bound)

Example::

    python scripts/simulate_surrogate_selection.py
"""

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--oof",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "model_comparison_confirm"
        / "oof_predictions.csv",
    )
    p.add_argument("--model", default="extra_trees")
    p.add_argument("--feature-set", default="early+math+behavioral")
    p.add_argument("--target", default="success_rate")
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--budgets", default="1,3,5,10,20,30,50,108")
    p.add_argument("--n-random", type=int, default=500)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "selection_sim_phase5",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    oof = pd.read_csv(args.oof)
    m = oof[
        (oof["model"] == args.model)
        & (oof["feature_set"] == args.feature_set)
        & (oof["target"] == args.target)
        & (oof["repeat"] == args.repeat)
    ].copy()
    if m.empty:
        print("No OOF rows", file=sys.stderr)
        return 1
    # One row per candidate
    if m["candidate_id"].duplicated().any():
        m = m.groupby("candidate_id", as_index=False).agg(
            y_true=("y_true", "mean"), y_pred=("y_pred", "mean"), family_id=("family_id", "first")
        )
    y = m["y_true"].to_numpy(dtype=float)
    s = m["y_pred"].to_numpy(dtype=float)
    n = len(y)
    budgets = [min(int(b), n) for b in args.budgets.split(",") if b.strip()]
    budgets = sorted(set(budgets))

    oracle_order = np.argsort(-y)
    sur_order = np.argsort(-s)
    rng = np.random.default_rng(args.seed)

    rows = []
    for k in budgets:
        oracle_best = float(np.max(y[oracle_order[:k]]))
        oracle_mean = float(np.mean(y[oracle_order[:k]]))
        sur_best = float(np.max(y[sur_order[:k]]))
        sur_mean = float(np.mean(y[sur_order[:k]]))
        rand_bests = []
        rand_means = []
        for _ in range(args.n_random):
            idx = rng.choice(n, size=k, replace=False)
            rand_bests.append(float(np.max(y[idx])))
            rand_means.append(float(np.mean(y[idx])))
        rows.append(
            {
                "budget_k": k,
                "oracle_best": oracle_best,
                "oracle_mean_topk": oracle_mean,
                "surrogate_best": sur_best,
                "surrogate_mean_topk": sur_mean,
                "random_best_mean": float(np.mean(rand_bests)),
                "random_best_std": float(np.std(rand_bests)),
                "random_mean_topk": float(np.mean(rand_means)),
                "regret_best_vs_oracle": oracle_best - sur_best,
                "regret_best_vs_random": float(np.mean(rand_bests)) - sur_best,
                "frac_oracle_top1_in_sur_topk": float(
                    oracle_order[0] in set(sur_order[:k].tolist())
                ),
            }
        )

    out = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "regret_vs_budget.csv"
    out.to_csv(csv_path, index=False)

    # Simple efficiency: smallest k with surrogate best within 0.02 of global oracle
    global_best = float(np.max(y))
    efficient_k = None
    for _, r in out.iterrows():
        if global_best - r.surrogate_best <= 0.02:
            efficient_k = int(r.budget_k)
            break

    summary = {
        "model": args.model,
        "feature_set": args.feature_set,
        "n_candidates": n,
        "global_best_sr": global_best,
        "efficient_budget_k_within_0.02": efficient_k,
        "at_k10": out[out.budget_k == 10].to_dict("records")[0]
        if (out.budget_k == 10).any()
        else None,
        # regret_best_vs_random = random_best - surrogate_best; negative ⇒ surrogate wins
        "recommendation": (
            "surrogate_top_k_beats_random"
            if out.iloc[min(3, len(out) - 1)]["regret_best_vs_random"] < 0
            else "weak_selection_gain"
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Offline selection simulation (Phase 5)",
        "",
        f"- OOF source: confirm `{args.model}` / `{args.feature_set}` / repeat={args.repeat}",
        f"- Candidates: **{n}**",
        f"- Global best SR: **{global_best:.4f}**",
        f"- Smallest k with surrogate best within 0.02 of oracle: **{efficient_k}**",
        "",
        "| k | oracle best | surrogate best | random best (mean) | regret vs oracle | hit oracle#1 |",
        "|--:|------------:|---------------:|-------------------:|-----------------:|-------------:|",
    ]
    for _, r in out.iterrows():
        lines.append(
            f"| {int(r.budget_k)} | {r.oracle_best:.4f} | {r.surrogate_best:.4f} | "
            f"{r.random_best_mean:.4f} | {r.regret_best_vs_oracle:.4f} | {r.frac_oracle_top1_in_sur_topk:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Gate for real AL",
            "",
            "- Proceed to online AL only if surrogate clearly beats random at small k",
            "  and regret vs oracle shrinks monotonically.",
            "- This simulation uses LOFO OOF scores (slightly optimistic vs a single frozen model).",
            "",
        ]
    )
    report = args.out_dir / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
