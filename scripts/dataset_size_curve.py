#!/usr/bin/env python3
"""Learning curve: ExtraTrees LOFO accuracy vs number of labeled train candidates.

Each LOFO test family stays intact. Only the training side is shrunk,
stratified across the other families.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.model_comparison import (  # noqa: E402
    _fit_predict,
    evaluate_oof_bundle,
    load_feature_matrix,
    load_yaml_config,
)

DEFAULT_TRAIN_SIZES = (12, 18, 24, 36, 48, 72, 99)
N_REPEATS = 5
PARAMS = {"n_estimators": 250, "max_depth": None, "min_samples_leaf": 4}
PREP = {
    "target_transform": "none",
    "feature_selection": "none",
    "k": 8,
    "pca_components": None,
}


def _subsample_train(
    train_idx: np.ndarray,
    groups: np.ndarray,
    train_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(train_idx) <= train_n:
        return train_idx
    families = groups[train_idx]
    order = rng.permutation(np.unique(families))
    base, extra = divmod(train_n, len(order))
    chosen: list[np.ndarray] = []
    for i, family in enumerate(order):
        members = train_idx[families == family]
        take = min(len(members), base + (1 if i < extra else 0))
        if take <= 0:
            continue
        chosen.append(rng.choice(members, size=take, replace=False))
    picked = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    if len(picked) < train_n:
        rest = np.setdiff1d(train_idx, picked, assume_unique=False)
        need = min(train_n - len(picked), len(rest))
        if need > 0:
            picked = np.concatenate([picked, rng.choice(rest, size=need, replace=False)])
    return picked


def _one_curve_point(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    train_n: int,
    seed: int,
    config: dict,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    oof = np.full(len(y), np.nan)
    families = np.unique(groups)
    for fold_i, family in enumerate(families):
        test_idx = np.where(groups == family)[0]
        train_idx = np.where(groups != family)[0]
        chosen = _subsample_train(train_idx, groups, train_n, rng)
        pred, _, _, _ = _fit_predict(
            "extra_trees",
            X[chosen],
            y[chosen],
            groups[chosen],
            X[test_idx],
            params=PARAMS,
            prep_cfg=PREP,
            standardize=False,
            random_state=seed + fold_i,
        )
        oof[test_idx] = pred
    metrics = evaluate_oof_bundle(
        y,
        oof,
        groups,
        target="success_rate",
        model_name="extra_trees",
        config=config,
        seed=seed,
    )
    return {
        "within_spearman": float(metrics["within_family_spearman_mean"]),
        "top3": float(metrics["within_family_top3_hit"]),
        "within_pairwise": float(metrics["within_pairwise_accuracy"]),
        "pooled_spearman": float(metrics["spearman"]),
        "mae": float(metrics["mae"]),
        "n_train_actual_mean": float(
            np.mean(
                [
                    min(train_n, int((groups != fam).sum()))
                    for fam in families
                ]
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase2" / "features",
    )
    p.add_argument(
        "--feature-set",
        default="early+math+behavioral",
    )
    p.add_argument(
        "--train-sizes",
        default=None,
        help="Comma-separated train_n values (default depends on pool size).",
    )
    p.add_argument("--n-repeats", type=int, default=N_REPEATS)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV output path.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model_comparison.yaml",
    )
    return p.parse_args()


def _default_train_sizes(n_candidates: int, n_families: int) -> tuple[int, ...]:
    max_train = n_candidates - max(1, n_candidates // n_families)
    if n_candidates <= 80:
        # CrowdNav-scale pool (~72): denser grid up to LOFO max train.
        grid = [12, 18, 24, 33, 44, 55, max_train]
        return tuple(sorted({min(g, max_train) for g in grid if g >= 12}))
    return tuple(s for s in DEFAULT_TRAIN_SIZES if s <= max_train) or (max_train,)


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg["n_bootstrap"] = 200
    X_df, meta, targets = load_feature_matrix(args.features_dir, args.feature_set)
    X = X_df.to_numpy(dtype=np.float64)
    y = targets["success_rate"].to_numpy(dtype=np.float64)
    groups = meta["family_id"].to_numpy()
    n_cand = len(y)
    n_fam = int(pd.Series(groups).nunique())
    max_train = n_cand - int((groups == groups[0]).sum())

    if args.train_sizes:
        train_sizes = tuple(int(x) for x in args.train_sizes.split(",") if x.strip())
    else:
        train_sizes = _default_train_sizes(n_cand, n_fam)
    train_sizes = tuple(min(t, max_train) for t in train_sizes)

    out = args.out
    if out is None:
        if "crowdnav" in str(args.features_dir).lower():
            out = (
                PROJECT_ROOT
                / "results"
                / "crowdnav_v4_full_150k"
                / "dataset_size_curve.csv"
            )
        else:
            out = PROJECT_ROOT / "results" / "dataset_size_curve.csv"

    print(
        f"features={args.features_dir} set={args.feature_set} "
        f"n={n_cand} families={n_fam} max_train={max_train} "
        f"sizes={train_sizes}",
        flush=True,
    )

    rows: list[dict[str, float]] = []
    for train_n in train_sizes:
        point_rows = []
        for repeat in range(args.n_repeats):
            point = _one_curve_point(
                X,
                y,
                groups,
                train_n=train_n,
                seed=1000 + train_n * 10 + repeat,
                config=cfg,
            )
            point_rows.append(point)
            print(
                f"train_n={train_n} repeat={repeat} "
                f"within={point['within_spearman']:.3f} top3={point['top3']:.3f}",
                flush=True,
            )
        frame = pd.DataFrame(point_rows)
        summary = {
            "train_n": train_n,
            "n_repeats": args.n_repeats,
            "n_candidates": n_cand,
            "n_families": n_fam,
            "max_train_lofo": max_train,
        }
        for column in frame.columns:
            summary[f"{column}_mean"] = float(frame[column].mean())
            summary[f"{column}_std"] = float(frame[column].std(ddof=0))
        rows.append(summary)
        print(
            f"SUMMARY train_n={train_n} "
            f"within={summary['within_spearman_mean']:.3f} "
            f"top3={summary['top3_mean']:.3f} "
            f"pairwise={summary['within_pairwise_mean']:.3f} "
            f"pooled={summary['pooled_spearman_mean']:.3f}",
            flush=True,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
