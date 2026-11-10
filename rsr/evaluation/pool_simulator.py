"""Offline pool simulator over locked OOF predictions (no new PPO)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from rsr.evaluation.cost import triage_cost
from rsr.evaluation.triage_metrics import (
    bootstrap_ci,
    break_ties,
    pairwise_accuracy,
    pool_metrics,
    select_topk_indices,
    spearman_safe,
    top1_hit,
)

Selector = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def selector_surrogate(scores: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return select_topk_indices(scores, k, rng=rng, higher_is_better=True)


def selector_random(scores: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = scores.size
    kk = min(int(k), n)
    if kk <= 0:
        return np.asarray([], dtype=int)
    return rng.choice(n, size=kk, replace=False)


def selector_oracle(y_true: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return select_topk_indices(y_true, k, rng=rng, higher_is_better=True)


SELECTORS: dict[str, str] = {
    "surrogate": "top-k by OOF y_pred",
    "random": "uniform random-k",
    "oracle": "top-k by y_true (upper bound)",
}


@dataclass(frozen=True)
class PoolDraw:
    candidate_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    n_groups: int


def load_oof_candidates(
    oof: pd.DataFrame,
    *,
    model: str = "extra_trees",
    feature_set: str = "early+math+behavioral",
    target: str = "success_rate",
    repeat: int | None = 0,
) -> pd.DataFrame:
    """One row per candidate with y_true / y_pred / family_id."""
    m = oof[
        (oof["model"] == model)
        & (oof["feature_set"] == feature_set)
        & (oof["target"] == target)
    ].copy()
    if repeat is not None:
        m = m[m["repeat"] == int(repeat)]
    if m.empty:
        raise ValueError("No OOF rows for the requested slice")
    if m["candidate_id"].duplicated().any():
        m = m.groupby("candidate_id", as_index=False).agg(
            y_true=("y_true", "mean"),
            y_pred=("y_pred", "mean"),
            family_id=("family_id", "first"),
        )
    else:
        m = m[["candidate_id", "family_id", "y_true", "y_pred"]].copy()
    return m.reset_index(drop=True)


def _choose_groups(
    groups: Sequence[str],
    n_groups: int,
    pool_size: int,
    by_group: Mapping[str, pd.DataFrame],
    rng: np.random.Generator,
) -> list[str]:
    """
    Sample n_groups families; if one family is too small for the pool, add groups
    until the union can supply ``pool_size`` candidates (documented Phase 1 rule).
    """
    available = list(groups)
    rng.shuffle(available)
    chosen: list[str] = []
    for g in available:
        if len(chosen) >= n_groups and sum(len(by_group[x]) for x in chosen) >= pool_size:
            break
        if g in chosen:
            continue
        # Prefer hitting the requested n_groups first.
        if len(chosen) < n_groups or sum(len(by_group[x]) for x in chosen) < pool_size:
            chosen.append(g)
    # Still short? keep adding.
    for g in available:
        if sum(len(by_group[x]) for x in chosen) >= pool_size:
            break
        if g not in chosen:
            chosen.append(g)
    return chosen


def draw_pool(
    candidates: pd.DataFrame,
    *,
    pool_size: int,
    n_groups: int,
    rng: np.random.Generator,
    group_col: str = "family_id",
) -> PoolDraw | None:
    by_group = {g: df for g, df in candidates.groupby(group_col)}
    groups = list(by_group.keys())
    if not groups:
        return None
    chosen = _choose_groups(groups, n_groups, pool_size, by_group, rng)
    pool_df = pd.concat([by_group[g] for g in chosen], ignore_index=True)
    if len(pool_df) < pool_size:
        return None
    idx = rng.choice(len(pool_df), size=int(pool_size), replace=False)
    sample = pool_df.iloc[idx].reset_index(drop=True)
    return PoolDraw(
        candidate_ids=tuple(sample["candidate_id"].astype(str)),
        group_ids=tuple(sample[group_col].astype(str)),
        y_true=sample["y_true"].to_numpy(dtype=float),
        y_pred=sample["y_pred"].to_numpy(dtype=float),
        n_groups=int(sample[group_col].nunique()),
    )


def _metrics_from_kept(
    y_true: np.ndarray,
    kept: np.ndarray,
    *,
    y_score_for_aux: np.ndarray,
    k: int,
    rng: np.random.Generator,
    top1_kept: np.ndarray | None = None,
) -> dict[str, float]:
    """Recall/regret from an explicit kept set; pairwise/spearman from scores."""
    if kept.size == 0:
        recall = float("nan")
        regret = float("nan")
    else:
        true_best = int(break_ties(y_true, rng=rng, higher_is_better=True)[0])
        recall = float(true_best in set(kept.tolist()))
        regret = float(np.max(y_true) - np.max(y_true[kept]))
    if top1_kept is None:
        t1 = top1_hit(y_true, y_score_for_aux, rng=rng)
    else:
        true_best = int(break_ties(y_true, rng=rng, higher_is_better=True)[0])
        t1 = float(true_best in set(top1_kept.tolist()))
    return {
        "n": float(y_true.size),
        "k": float(k),
        "recall_at_k": recall,
        "regret_at_k": regret,
        "top1_hit": t1,
        "pairwise_accuracy": pairwise_accuracy(y_true, y_score_for_aux),
        "spearman": spearman_safe(y_true, y_score_for_aux),
    }


def evaluate_pool(
    pool: PoolDraw,
    *,
    k: int,
    selector: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    if selector == "random":
        kept = selector_random(pool.y_pred, k, rng)
        kept1 = selector_random(pool.y_pred, 1, rng)
        metrics = _metrics_from_kept(
            pool.y_true,
            kept,
            y_score_for_aux=pool.y_pred,
            k=k,
            rng=rng,
            top1_kept=kept1,
        )
    elif selector == "oracle":
        kept = selector_oracle(pool.y_true, k, rng)
        metrics = _metrics_from_kept(
            pool.y_true, kept, y_score_for_aux=pool.y_true, k=k, rng=rng
        )
    else:
        metrics = pool_metrics(pool.y_true, pool.y_pred, k, rng=rng)
    metrics["n_groups_realized"] = float(pool.n_groups)
    return metrics


def simulate_pools(
    candidates: pd.DataFrame,
    *,
    pool_sizes: Sequence[int] = (8, 16, 32),
    n_groups_list: Sequence[int] = (1, 2, 3),
    ks: Sequence[int] = (3, 4, 5, 6),
    n_pools: int = 2000,
    selectors: Sequence[str] = ("surrogate", "random", "oracle"),
    seed: int = 123,
    feature_steps_per_candidate: int = 12000,
    full_run_steps: int = 50000,
    n_boot: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample pools and return (per_pool_rows, summary_rows).

    Caveat: when n_groups > 1, OOF y_pred for different families come from
    different LOFO fold models. Single-group pools avoid that mismatch.
    """
    rng = np.random.default_rng(seed)
    per_rows: list[dict[str, Any]] = []
    for n in pool_sizes:
        for n_groups in n_groups_list:
            for pool_i in range(int(n_pools)):
                pool = draw_pool(
                    candidates, pool_size=int(n), n_groups=int(n_groups), rng=rng
                )
                if pool is None:
                    continue
                for k in ks:
                    if int(k) > int(n):
                        continue
                    for sel in selectors:
                        child = np.random.default_rng(rng.integers(0, 2**31 - 1))
                        metrics = evaluate_pool(pool, k=int(k), selector=sel, rng=child)
                        cost = triage_cost(
                            int(n),
                            int(k),
                            feature_steps_per_candidate=feature_steps_per_candidate,
                            full_run_steps=full_run_steps,
                        )
                        per_rows.append(
                            {
                                "pool_i": pool_i,
                                "N": int(n),
                                "k": int(k),
                                "n_groups_requested": int(n_groups),
                                "n_groups_realized": int(metrics["n_groups_realized"]),
                                "selector": sel,
                                "recall_at_k": metrics["recall_at_k"],
                                "regret_at_k": metrics["regret_at_k"],
                                "top1_hit": metrics["top1_hit"],
                                "pairwise_accuracy": metrics["pairwise_accuracy"],
                                "spearman": metrics["spearman"],
                                "cost_steps": cost["cost_steps"],
                                "savings_frac": cost["savings_frac"],
                            }
                        )

    per_df = pd.DataFrame(per_rows)
    if per_df.empty:
        return per_df, pd.DataFrame()

    summaries: list[dict[str, Any]] = []
    keys = ["N", "k", "n_groups_requested", "selector"]
    for key_vals, g in per_df.groupby(keys, sort=True):
        key_dict = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        row: dict[str, Any] = {**key_dict, "n_pools": int(len(g))}
        for metric in ("recall_at_k", "regret_at_k", "top1_hit", "pairwise_accuracy"):
            ci_mean = bootstrap_ci(
                g[metric].to_numpy(), n_boot=n_boot, rng=rng, statistic="mean"
            )
            ci_med = bootstrap_ci(
                g[metric].to_numpy(), n_boot=n_boot, rng=rng, statistic="median"
            )
            row[f"{metric}_mean"] = ci_mean["estimate"]
            row[f"{metric}_mean_lo"] = ci_mean["lo"]
            row[f"{metric}_mean_hi"] = ci_mean["hi"]
            row[f"{metric}_median"] = ci_med["estimate"]
            row[f"{metric}_median_lo"] = ci_med["lo"]
            row[f"{metric}_median_hi"] = ci_med["hi"]
        # Surrogate minus random on paired pools (same pool_i)
        if "surrogate" in set(per_df["selector"]) and key_dict["selector"] == "surrogate":
            base = per_df[
                (per_df["N"] == key_dict["N"])
                & (per_df["k"] == key_dict["k"])
                & (per_df["n_groups_requested"] == key_dict["n_groups_requested"])
            ]
            sur = base[base["selector"] == "surrogate"].set_index("pool_i")
            rnd = base[base["selector"] == "random"].set_index("pool_i")
            common = sur.index.intersection(rnd.index)
            if len(common) > 0:
                delta = (
                    sur.loc[common, "recall_at_k"].to_numpy()
                    - rnd.loc[common, "recall_at_k"].to_numpy()
                )
                dci = bootstrap_ci(delta, n_boot=n_boot, rng=rng, statistic="mean")
                row["recall_minus_random_mean"] = dci["estimate"]
                row["recall_minus_random_lo"] = dci["lo"]
                row["recall_minus_random_hi"] = dci["hi"]
                dreg = (
                    rnd.loc[common, "regret_at_k"].to_numpy()
                    - sur.loc[common, "regret_at_k"].to_numpy()
                )
                rci = bootstrap_ci(dreg, n_boot=n_boot, rng=rng, statistic="mean")
                row["regret_improvement_vs_random_mean"] = rci["estimate"]
                row["regret_improvement_vs_random_lo"] = rci["lo"]
                row["regret_improvement_vs_random_hi"] = rci["hi"]
        cost = triage_cost(
            int(key_dict["N"]),
            int(key_dict["k"]),
            feature_steps_per_candidate=feature_steps_per_candidate,
            full_run_steps=full_run_steps,
        )
        row["savings_frac"] = cost["savings_frac"]
        summaries.append(row)
    return per_df, pd.DataFrame(summaries)
