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
    expand_groups: bool = True,
) -> PoolDraw | None:
    """
    Draw a pool of ``pool_size`` **distinct** candidates (without replacement).

    When ``expand_groups`` is True (default) and the first ``n_groups`` families
    cannot supply ``pool_size`` rows (2D families are size 8–10, so N=16 with
    n_groups=1 always expands), additional families are added until the union
    is large enough. When ``expand_groups`` is False, exactly ``n_groups``
    families are used and the draw returns None if they are too small — this is
    the within-family mode (e.g. N=8, n_groups=1).
    """
    by_group = {g: df for g, df in candidates.groupby(group_col)}
    groups = list(by_group.keys())
    if not groups:
        return None
    if expand_groups:
        chosen = _choose_groups(groups, n_groups, pool_size, by_group, rng)
    else:
        if len(groups) < int(n_groups):
            return None
        chosen = list(rng.choice(groups, size=int(n_groups), replace=False))
    pool_df = pd.concat([by_group[g] for g in chosen], ignore_index=True)
    if pool_df["candidate_id"].duplicated().any():
        raise RuntimeError("candidate_id duplicates in group union — refuse draw")
    if len(pool_df) < pool_size:
        return None
    idx = rng.choice(len(pool_df), size=int(pool_size), replace=False)
    sample = pool_df.iloc[idx].reset_index(drop=True)
    if sample["candidate_id"].duplicated().any():
        raise RuntimeError("duplicate candidates in pool sample")
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
    # Fix true-best once up front so random top1 / recall share the same label tie-break.
    true_best = int(break_ties(pool.y_true, rng=rng, higher_is_better=True)[0])
    if selector == "random":
        kept = selector_random(pool.y_pred, k, rng)
        kept1 = selector_random(pool.y_pred, 1, rng)
        recall = float(true_best in set(kept.tolist())) if kept.size else float("nan")
        regret = (
            float(np.max(pool.y_true) - np.max(pool.y_true[kept]))
            if kept.size
            else float("nan")
        )
        metrics = {
            "n": float(pool.y_true.size),
            "k": float(k),
            "recall_at_k": recall,
            "regret_at_k": regret,
            "top1_hit": float(true_best in set(kept1.tolist())),
            "pairwise_accuracy": pairwise_accuracy(pool.y_true, pool.y_pred),
            "spearman": spearman_safe(pool.y_true, pool.y_pred),
        }
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
    expand_groups: bool = True,
    ci_mode: str = "pool",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample pools and return (per_pool_rows, summary_rows).

    Caveat: when n_groups > 1 (or expand_groups fills N>family size), OOF y_pred
    for different families come from different LOFO fold models. On 2D, N=16 with
    n_groups_requested=1 yields a 2-family pool (n_groups=1 expanded to 2 families).

    ``ci_mode``:
      - ``pool``: bootstrap over pool draws (legacy Phase 1).
      - ``family``: resample the 12 families, then re-draw pools (Gate 1 addendum).
    """
    rng = np.random.default_rng(seed)
    per_rows: list[dict[str, Any]] = []
    for n in pool_sizes:
        for n_groups in n_groups_list:
            for pool_i in range(int(n_pools)):
                pool = draw_pool(
                    candidates,
                    pool_size=int(n),
                    n_groups=int(n_groups),
                    rng=rng,
                    expand_groups=expand_groups,
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
                                "families": "|".join(sorted(set(pool.group_ids))),
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

    if ci_mode == "family":
        summary = _summarize_family_bootstrap(
            candidates,
            per_df,
            pool_sizes=pool_sizes,
            n_groups_list=n_groups_list,
            ks=ks,
            selectors=selectors,
            n_pools=n_pools,
            n_boot=n_boot,
            seed=seed + 17,
            expand_groups=expand_groups,
            feature_steps_per_candidate=feature_steps_per_candidate,
            full_run_steps=full_run_steps,
        )
        return per_df, summary

    summaries: list[dict[str, Any]] = []
    keys = ["N", "k", "n_groups_requested", "selector"]
    for key_vals, g in per_df.groupby(keys, sort=True):
        key_dict = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        row: dict[str, Any] = {
            **key_dict,
            "n_pools": int(len(g)),
            "ci_mode": "pool",
            "n_families_total": int(candidates["family_id"].nunique()),
        }
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


def _summarize_family_bootstrap(
    candidates: pd.DataFrame,
    per_df: pd.DataFrame,
    *,
    pool_sizes: Sequence[int],
    n_groups_list: Sequence[int],
    ks: Sequence[int],
    selectors: Sequence[str],
    n_pools: int,
    n_boot: int,
    seed: int,
    expand_groups: bool,
    feature_steps_per_candidate: int,
    full_run_steps: int,
) -> pd.DataFrame:
    """Resample families with replacement, then redraw pools for CIs."""
    families = np.asarray(sorted(candidates["family_id"].unique()))
    n_fam = int(families.size)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    configs = [
        (int(n), int(ng), int(k), sel)
        for n in pool_sizes
        for ng in n_groups_list
        for k in ks
        if int(k) <= int(n)
        for sel in selectors
    ]
    # Point estimates from the original per_df draws
    point: dict[tuple, dict[str, float]] = {}
    for (n, ng, k, sel), g in per_df.groupby(
        ["N", "n_groups_requested", "k", "selector"], sort=True
    ):
        point[(int(n), int(ng), int(k), str(sel))] = {
            "recall_at_k_mean": float(g["recall_at_k"].mean()),
            "regret_at_k_mean": float(g["regret_at_k"].mean()),
            "regret_at_k_median": float(g["regret_at_k"].median()),
            "n_pools": float(len(g)),
        }

    boot_store: dict[tuple, list[dict[str, float]]] = {c: [] for c in configs}
    # Keep addendum runtime bounded: few pools per bootstrap replicate.
    pools_per_boot = 40 if expand_groups else 20
    # Only CI the selectors we need for the gate (surrogate + random paired).
    ci_selectors = [s for s in selectors if s in ("surrogate", "random", "oracle")]
    for _ in range(int(n_boot)):
        chosen = rng.choice(families, size=n_fam, replace=True)
        uniq = list(dict.fromkeys(chosen.tolist()))
        sub = candidates[candidates["family_id"].isin(uniq)].reset_index(drop=True)
        for n, ng, k, sel in configs:
            if sel not in ci_selectors:
                continue
            recalls_s: list[float] = []
            recalls_r: list[float] = []
            regrets_s: list[float] = []
            regrets_r: list[float] = []
            for _pi in range(pools_per_boot):
                pool = draw_pool(
                    sub,
                    pool_size=n,
                    n_groups=ng,
                    rng=rng,
                    expand_groups=expand_groups,
                )
                if pool is None:
                    continue
                child = np.random.default_rng(rng.integers(0, 2**31 - 1))
                if sel == "surrogate":
                    ms = evaluate_pool(pool, k=k, selector="surrogate", rng=child)
                    mr = evaluate_pool(
                        pool,
                        k=k,
                        selector="random",
                        rng=np.random.default_rng(rng.integers(0, 2**31 - 1)),
                    )
                    recalls_s.append(ms["recall_at_k"])
                    regrets_s.append(ms["regret_at_k"])
                    recalls_r.append(mr["recall_at_k"])
                    regrets_r.append(mr["regret_at_k"])
                else:
                    m = evaluate_pool(pool, k=k, selector=sel, rng=child)
                    recalls_s.append(m["recall_at_k"])
                    regrets_s.append(m["regret_at_k"])
            if not recalls_s:
                continue
            entry = {
                "recall_at_k_mean": float(np.mean(recalls_s)),
                "regret_at_k_mean": float(np.mean(regrets_s)),
                "regret_at_k_median": float(np.median(regrets_s)),
            }
            if sel == "surrogate" and recalls_r:
                entry["recall_minus_random"] = float(
                    np.mean(recalls_s) - np.mean(recalls_r)
                )
                entry["regret_improvement_vs_random"] = float(
                    np.mean(regrets_r) - np.mean(regrets_s)
                )
            boot_store[(n, ng, k, sel)].append(entry)

    for n, ng, k, sel in configs:
        boots = boot_store[(n, ng, k, sel)]
        pt = point.get((n, ng, k, sel), {})
        cost = triage_cost(
            n,
            k,
            feature_steps_per_candidate=feature_steps_per_candidate,
            full_run_steps=full_run_steps,
        )
        row: dict[str, Any] = {
            "N": n,
            "k": k,
            "n_groups_requested": ng,
            "selector": sel,
            "n_pools": pt.get("n_pools", float("nan")),
            "ci_mode": "family",
            "n_families_total": n_fam,
            "n_boot": int(n_boot),
            "held_out_groups_threshold_met": False,  # only 12 families exist
            "recall_at_k_mean": pt.get("recall_at_k_mean", float("nan")),
            "regret_at_k_mean": pt.get("regret_at_k_mean", float("nan")),
            "regret_at_k_median": pt.get("regret_at_k_median", float("nan")),
            "savings_frac": cost["savings_frac"],
        }
        if boots:
            for key, out_lo, out_hi in (
                ("recall_at_k_mean", "recall_at_k_mean_lo", "recall_at_k_mean_hi"),
                ("regret_at_k_median", "regret_at_k_median_lo", "regret_at_k_median_hi"),
            ):
                arr = np.asarray([b[key] for b in boots if key in b], dtype=float)
                if arr.size:
                    lo, hi = np.quantile(arr, [0.025, 0.975])
                    row[out_lo] = float(lo)
                    row[out_hi] = float(hi)
            if sel == "surrogate":
                d = np.asarray(
                    [b["recall_minus_random"] for b in boots if "recall_minus_random" in b],
                    dtype=float,
                )
                if d.size:
                    lo, hi = np.quantile(d, [0.025, 0.975])
                    row["recall_minus_random_mean"] = float(np.mean(d))
                    row["recall_minus_random_lo"] = float(lo)
                    row["recall_minus_random_hi"] = float(hi)
        rows.append(row)
    return pd.DataFrame(rows)
