"""Shared Gate-2 / 6b / 6c harness helpers (fold keys, dedupe, cost formulas)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from reward_triage.adapt import load_prior_xy
from reward_triage.feature_matrix import (
    FeatureSet,
    load_phase1_feature_table,
    vectorize_record,
)
from reward_triage.model import SurrogateModel
from reward_triage.store import LabelStore, StoreRecord

ROOT = Path(__file__).resolve().parents[1]
RUN_IDS = ("phase6_2d_r1b", "phase6_2d_r2", "phase6_2d_r3")
FeatureMode = Literal["early+math+behavioral", "math+behavioral", "early"]


def seeds_for_run(run_id: str, n_seeds: int = 3) -> tuple[int, ...]:
    """Deterministic PPO seeds derived from run_id (independent across runs)."""
    out: list[int] = []
    for i in range(int(n_seeds)):
        digest = hashlib.sha256(f"{run_id}:seed:{i}".encode()).digest()
        out.append(int.from_bytes(digest[:4], "big") % (2**31 - 1))
    return tuple(out)


def savings_restart(e: float, k: float, n: float) -> float:
    """1 - (e + k/N) with e = feature_steps/full_steps."""
    return float(1.0 - (e + k / n))


def savings_resume(e: float, k: float, n: float) -> float:
    """(1 - k/N) * (1 - e) — resume kept candidates from early checkpoint."""
    return float((1.0 - k / n) * (1.0 - e))


def mean_success(rec: StoreRecord) -> float | None:
    done = [lab for lab in rec.labels if lab.status == "completed"]
    if not done:
        return None
    return float(sum(lab.success for lab in done) / len(done))


def seed_se(rec: StoreRecord) -> float | None:
    done = [lab.success for lab in rec.labels if lab.status == "completed"]
    if len(done) < 2:
        return None
    return float(np.std(done, ddof=1) / np.sqrt(len(done)))


def load_all_records(run_ids: tuple[str, ...] = RUN_IDS) -> list[StoreRecord]:
    out: list[StoreRecord] = []
    for run_id in run_ids:
        path = ROOT / "experiments" / run_id / "label_store.jsonl"
        out.extend(LabelStore(path).load_all())
    return out


def feature_columns(feature_set: FeatureMode) -> list[str]:
    if feature_set == "early+math+behavioral":
        cold = SurrogateModel.load(ROOT / "experiments" / "phase5_surrogate_2d")
        return list(cold.feature_cols)
    _, cols = load_phase1_feature_table(feature_set)  # type: ignore[arg-type]
    return cols


def records_to_frame(
    records: list[StoreRecord],
    feature_cols: list[str],
    *,
    labeled_only: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        y = mean_success(rec)
        if labeled_only and y is None:
            continue
        x = vectorize_record(rec, feature_cols)
        se = seed_se(rec)
        rows.append(
            {
                "candidate_id": rec.candidate.candidate_id,
                "code_hash": rec.candidate.code_hash,
                "lineage_id": rec.candidate.lineage_id,
                "run_id": rec.run_id,
                "generation": int(rec.candidate.generation),
                "y_true": y if y is not None else float("nan"),
                "seed_se": se if se is not None else float("nan"),
                "decision": rec.selection.decision,
                "probability": float(rec.selection.probability),
                "live_score": rec.selection.score,
                "fold_key": f"{rec.candidate.code_hash}::{rec.candidate.lineage_id}",
                **{f"f{i}": float(x[i]) for i in range(len(feature_cols))},
            }
        )
    return pd.DataFrame(rows)


def dedupe_by_code_hash(df: pd.DataFrame) -> pd.DataFrame:
    """Average labels / features for identical code_hash before pooling."""
    if df.empty:
        return df
    feat_cols = [
        c for c in df.columns if c.startswith("f") and c[1:].isdigit()
    ]
    rows: list[dict[str, Any]] = []
    for code_hash, g in df.groupby("code_hash", sort=False):
        row: dict[str, Any] = {
            "code_hash": str(code_hash),
            "y_true": float(np.nanmean(g["y_true"].to_numpy(dtype=float))),
            "seed_se": float(np.nanmean(g["seed_se"].to_numpy(dtype=float))),
            "generation": int(g["generation"].min()),
            "lineage_id": str(g["lineage_id"].iloc[0]),
            "run_id": str(g["run_id"].iloc[0]),
            "candidate_id": str(g["candidate_id"].iloc[0]),
            "fold_key": str(code_hash),
            "decision": str(g["decision"].iloc[0]),
            "probability": float(np.nanmean(g["probability"].to_numpy(dtype=float))),
        }
        if "live_score" in g.columns:
            ls = pd.to_numeric(g["live_score"], errors="coerce").to_numpy(dtype=float)
            row["live_score"] = float(np.nanmean(ls)) if np.isfinite(ls).any() else float("nan")
        for c in feat_cols:
            row[c] = float(np.nanmean(g[c].to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows)


def assert_no_code_hash_leak(train_hashes: set[str], test_hashes: set[str]) -> None:
    leak = train_hashes & test_hashes
    if leak:
        raise AssertionError(f"code_hash leakage across fold: {sorted(leak)[:5]}")


def lolo_oof(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    fold_col: str = "fold_key",
    use_prior: bool = False,
    feature_set: FeatureMode = "early+math+behavioral",
    sample_weight: np.ndarray | None = None,
    enforce_no_hash_leak: bool = False,
) -> pd.DataFrame:
    """Leave-one-fold-out predictions; optionally blend phase1 prior."""
    X = df[[f"f{i}" for i in range(len(feature_cols))]].to_numpy(dtype=float)
    y = df["y_true"].to_numpy(dtype=float)
    folds = df[fold_col].astype(str).to_numpy()
    y_pred = np.full(len(y), np.nan, dtype=float)
    X_prior = y_prior = None
    if use_prior:
        X_prior, y_prior = load_prior_xy(
            feature_cols, feature_set=feature_set  # type: ignore[arg-type]
        )

    for fold in sorted(set(folds.tolist())):
        te = folds == fold
        tr = ~te
        if not np.any(tr) or not np.any(te):
            continue
        tr_hashes = set(df.loc[tr, "code_hash"].astype(str))
        te_hashes = set(df.loc[te, "code_hash"].astype(str))
        if enforce_no_hash_leak:
            assert_no_code_hash_leak(tr_hashes, te_hashes)

        model = SurrogateModel(feature_cols, feature_set=feature_set)
        sw = None if sample_weight is None else sample_weight[tr]
        if use_prior and X_prior is not None and y_prior is not None:
            # Prior rows get weight 1.0; campaign rows use sw when provided.
            model.adapt_with_prior(
                X[tr],
                y[tr],
                X_prior=X_prior,
                y_prior=y_prior,
                dataset_hash=f"fold_{fold}",
            )
        else:
            # Fit with optional sample weights via underlying pipeline.
            pipe = model._make_pipeline()
            if sw is not None:
                pipe.fit(X[tr], y[tr], model__sample_weight=sw)
            else:
                pipe.fit(X[tr], y[tr])
            model.pipeline = pipe
        assert model.pipeline is not None
        y_pred[te] = model.pipeline.predict(X[te])

    base_cols = [
        "candidate_id",
        "code_hash",
        "lineage_id",
        "run_id",
        "generation",
        "y_true",
        "seed_se",
    ]
    cols_keep = list(base_cols)
    if fold_col not in cols_keep:
        cols_keep.append(fold_col)
    out = df[cols_keep].copy()
    out["family_id"] = df[fold_col].astype(str).to_numpy()
    out["y_pred"] = y_pred
    return out.dropna(subset=["y_pred"]).reset_index(drop=True)


def mb_feature_matrix(records: list[StoreRecord]) -> tuple[np.ndarray, list[str], list[str]]:
    """math+behavioral matrix aligned to phase1 columns; returns X, cols, code_hashes."""
    cols = feature_columns("math+behavioral")
    xs = []
    hashes = []
    for rec in records:
        xs.append(vectorize_record(rec, cols))
        hashes.append(rec.candidate.code_hash)
    X = np.vstack(xs) if xs else np.zeros((0, len(cols)))
    # Median impute for distance
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X = X.copy()
    X[inds] = np.take(med, inds[1])
    return X, cols, hashes
