"""Per-environment surrogate: fit / predict / adapt + versioned artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from reward_triage.schema import Prediction


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _schema_hash(feature_cols: Sequence[str]) -> str:
    blob = "\n".join(feature_cols).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class SurrogateArtifactMeta:
    feature_cols: list[str]
    feature_set: str
    schema_hash: str
    dataset_hash: str
    git_commit: str
    n_train: int
    conformal_q: float
    ood_threshold: float
    model_name: str = "extra_trees"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_cols": list(self.feature_cols),
            "feature_set": self.feature_set,
            "schema_hash": self.schema_hash,
            "dataset_hash": self.dataset_hash,
            "git_commit": self.git_commit,
            "n_train": int(self.n_train),
            "conformal_q": float(self.conformal_q),
            "ood_threshold": float(self.ood_threshold),
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurrogateArtifactMeta:
        return cls(
            feature_cols=list(data["feature_cols"]),
            feature_set=str(data.get("feature_set", "early+math+behavioral")),
            schema_hash=str(data["schema_hash"]),
            dataset_hash=str(data.get("dataset_hash", "")),
            git_commit=str(data.get("git_commit", "unknown")),
            n_train=int(data.get("n_train", 0)),
            conformal_q=float(data.get("conformal_q", 0.0)),
            ood_threshold=float(data.get("ood_threshold", float("inf"))),
            model_name=str(data.get("model_name", "extra_trees")),
        )


class SurrogateModel:
    """ExtraTrees + residual conformal interval + Mahalanobis OOD flag."""

    def __init__(
        self,
        feature_cols: list[str],
        *,
        feature_set: str = "early+math+behavioral",
        alpha: float = 0.1,
        random_state: int = 0,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.feature_set = feature_set
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.pipeline: Pipeline | None = None
        self.conformal_q = 0.0
        self._train_mean: np.ndarray | None = None
        self._train_cov_inv: np.ndarray | None = None
        self.ood_threshold = float("inf")
        self.meta: SurrogateArtifactMeta | None = None

    def _make_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=250,
                        min_samples_leaf=2,
                        random_state=self.random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        dataset_hash: str = "",
        calib_frac: float = 0.2,
    ) -> SurrogateModel:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n < 4:
            # Fit on all; conformal width = residual std fallback.
            self.pipeline = self._make_pipeline()
            self.pipeline.fit(X, y)
            pred = self.pipeline.predict(X)
            resid = np.abs(y - pred)
            self.conformal_q = float(np.quantile(resid, 1.0 - self.alpha)) if len(resid) else 0.5
        else:
            rng = np.random.default_rng(self.random_state)
            idx = np.arange(n)
            rng.shuffle(idx)
            n_cal = max(1, int(round(n * calib_frac)))
            cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
            if len(fit_idx) < 2:
                fit_idx, cal_idx = idx[n_cal:], idx[:n_cal]
            self.pipeline = self._make_pipeline()
            self.pipeline.fit(X[fit_idx], y[fit_idx])
            cal_pred = self.pipeline.predict(X[cal_idx])
            resid = np.abs(y[cal_idx] - cal_pred)
            self.conformal_q = float(np.quantile(resid, 1.0 - self.alpha))
            # Refit on all for deployment.
            self.pipeline.fit(X, y)

        # OOD: Mahalanobis on imputed features.
        Xt = self.pipeline.named_steps["imputer"].transform(X)
        self._train_mean = Xt.mean(axis=0)
        cov = np.cov(Xt, rowvar=False)
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov) + 1e-6]])
        cov = cov + np.eye(cov.shape[0]) * 1e-6
        try:
            self._train_cov_inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            self._train_cov_inv = np.eye(Xt.shape[1])
        dists = self._mahalanobis(Xt)
        self.ood_threshold = float(np.quantile(dists, 0.99)) if len(dists) else float("inf")

        self.meta = SurrogateArtifactMeta(
            feature_cols=list(self.feature_cols),
            feature_set=self.feature_set,
            schema_hash=_schema_hash(self.feature_cols),
            dataset_hash=dataset_hash or hashlib.sha256(y.tobytes()).hexdigest()[:16],
            git_commit=_git_commit(),
            n_train=int(n),
            conformal_q=self.conformal_q,
            ood_threshold=self.ood_threshold,
        )
        return self

    def adapt(self, X: np.ndarray, y: np.ndarray, *, dataset_hash: str = "") -> SurrogateModel:
        """Retrain on the (updated) environment label matrix."""
        return self.fit(X, y, dataset_hash=dataset_hash)

    def adapt_with_prior(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        *,
        X_prior: np.ndarray | None = None,
        y_prior: np.ndarray | None = None,
        dataset_hash: str = "",
    ) -> SurrogateModel:
        """Retrain on prior rows (e.g. phase1 108) plus new environment labels."""
        Xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        if X_prior is not None and y_prior is not None and len(y_prior):
            Xs.append(np.asarray(X_prior, dtype=float))
            ys.append(np.asarray(y_prior, dtype=float))
        if len(y_new):
            Xs.append(np.asarray(X_new, dtype=float))
            ys.append(np.asarray(y_new, dtype=float))
        if not Xs:
            return self
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        return self.fit(X, y, dataset_hash=dataset_hash or f"adapt_n={len(y)}")

    def _mahalanobis(self, Xt: np.ndarray) -> np.ndarray:
        assert self._train_mean is not None and self._train_cov_inv is not None
        delta = Xt - self._train_mean
        return np.sqrt(np.sum(delta @ self._train_cov_inv * delta, axis=1))

    def predict_rows(self, X: np.ndarray, *, feature_hashes: Sequence[str] | None = None) -> list[Prediction]:
        if self.pipeline is None or self.meta is None:
            raise RuntimeError("SurrogateModel is not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        means = self.pipeline.predict(X)
        Xt = self.pipeline.named_steps["imputer"].transform(X)
        dists = self._mahalanobis(Xt)
        nan_frac = np.mean(np.isnan(X), axis=1)
        out: list[Prediction] = []
        for i, mean in enumerate(means):
            lo = float(mean - self.conformal_q)
            hi = float(mean + self.conformal_q)
            fhash = (
                feature_hashes[i]
                if feature_hashes is not None and i < len(feature_hashes)
                else hashlib.sha256(X[i].tobytes()).hexdigest()[:16]
            )
            # Heavy imputation (e.g. missing early_* before PPO) should not force
            # OOD-bypass keep-all; rank by score instead.
            in_dist = bool(dists[i] <= self.ood_threshold) or bool(nan_frac[i] > 0.25)
            out.append(
                Prediction(
                    mean=float(mean),
                    std=float(self.conformal_q),
                    interval=(lo, hi),
                    in_distribution=in_dist,
                    feature_hash=fhash,
                )
            )
        return out

    def save(self, directory: str | Path) -> Path:
        if self.pipeline is None or self.meta is None:
            raise RuntimeError("nothing to save")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_cols": self.feature_cols,
                "feature_set": self.feature_set,
                "alpha": self.alpha,
                "conformal_q": self.conformal_q,
                "ood_threshold": self.ood_threshold,
                "train_mean": self._train_mean,
                "train_cov_inv": self._train_cov_inv,
                "meta": self.meta.to_dict(),
            },
            path / "model.joblib",
        )
        (path / "meta.json").write_text(
            json.dumps(self.meta.to_dict(), indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, directory: str | Path) -> SurrogateModel:
        path = Path(directory)
        payload = joblib.load(path / "model.joblib")
        obj = cls(
            list(payload["feature_cols"]),
            feature_set=str(payload.get("feature_set", "early+math+behavioral")),
            alpha=float(payload.get("alpha", 0.1)),
        )
        obj.pipeline = payload["pipeline"]
        obj.conformal_q = float(payload["conformal_q"])
        obj.ood_threshold = float(payload["ood_threshold"])
        obj._train_mean = payload["train_mean"]
        obj._train_cov_inv = payload["train_cov_inv"]
        obj.meta = SurrogateArtifactMeta.from_dict(payload["meta"])
        return obj
