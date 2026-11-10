"""Model factory for the model-comparison experiment (optional heavy deps)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from rsr.surrogate.baselines import MeanPredictor, MedianPredictor

logger = logging.getLogger("reward_surrogate.model_zoo")

OPTIONAL_SKIPPED: list[str] = []
_IMPORT_CACHE: dict[str, Any | None] = {}
_IMPORT_LOGGED: set[str] = set()


def _try_import(name: str) -> Any | None:
    if name in _IMPORT_CACHE:
        return _IMPORT_CACHE[name]
    try:
        mod = __import__(name)
    except Exception as exc:  # noqa: BLE001
        if name not in _IMPORT_LOGGED:
            logger.info(
                "Optional package '%s' unavailable (%s); related models skipped",
                name,
                exc,
            )
            _IMPORT_LOGGED.add(name)
            OPTIONAL_SKIPPED.append(name)
        _IMPORT_CACHE[name] = None
        return None
    _IMPORT_CACHE[name] = mod
    return mod


def _tabpfn_ready() -> bool:
    """True only if package is importable AND a license token is already available.

    TabPFN v2+ opens interactive Chrome login windows per fit otherwise.
    We never enable it unless the user explicitly opts in with a token.
    """
    import os

    # Always block browser popups from this process.
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")

    if _try_import("tabpfn") is None:
        return False
    # Require explicit opt-in; do not auto-run even if a stale cache exists unless token set.
    env_token = (os.environ.get("TABPFN_TOKEN") or "").strip()
    if not env_token:
        if "tabpfn_auth" not in _IMPORT_LOGGED:
            logger.info(
                "TabPFN disabled (would open Chrome for license). "
                "To enable later: set TABPFN_TOKEN and pass --models tabpfn."
            )
            _IMPORT_LOGGED.add("tabpfn_auth")
        return False
    return True


def available_models() -> dict[str, bool]:
    """Return which named models can be constructed in this environment."""
    has_lgb = _try_import("lightgbm") is not None
    has_xgb = _try_import("xgboost") is not None
    has_cb = _try_import("catboost") is not None
    has_tabpfn = _tabpfn_ready()
    has_torch = _try_import("torch") is not None
    return {
        "mean": True,
        "median": True,
        "ridge": True,
        "elasticnet": True,
        "kernel_ridge": True,
        "svr": True,
        "knn": True,
        "random_forest": True,
        "extra_trees": True,
        "hist_gradient_boosting": True,
        "gp_ard": True,
        "pairwise_logistic": True,
        "stack_ridge": True,
        "rank_average": True,
        "lightgbm": has_lgb,
        "catboost": has_cb,
        "xgboost": has_xgb,
        "lgbm_lambdarank": has_lgb,
        "xgb_rank_pairwise": has_xgb,
        "tabpfn": has_tabpfn,
        "mlp_mse": has_torch,
        "mlp_rank": has_torch,
        "tabm_or_realmlp": False,
    }


class PairwiseLogisticRanker(BaseEstimator, RegressorMixin):
    """Score candidates via logistic regression on feature differences."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        random_state: int = 0,
        max_pairs: int = 2000,
        within_family_frac: float = 1.0,
        within_family_only: bool = True,
        min_abs_delta: float = 0.0,
    ) -> None:
        self.C = C
        self.random_state = random_state
        self.max_pairs = max_pairs
        self.within_family_frac = within_family_frac
        self.within_family_only = within_family_only
        self.min_abs_delta = min_abs_delta

    def fit(self, X, y, groups=None):
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=False)
        groups = np.asarray(groups) if groups is not None else np.zeros(len(y), dtype=int)
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        min_gap = float(self.min_abs_delta)
        pairs: list[tuple[int, int]] = []
        # Prefer within-family pairs
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = int(idx[a]), int(idx[b])
                    if abs(float(y[i] - y[j])) <= min_gap:
                        continue
                    pairs.append((i, j) if y[i] > y[j] else (j, i))
        within = pairs
        cross: list[tuple[int, int]] = []
        if not bool(self.within_family_only):
            for _ in range(max(0, self.max_pairs)):
                i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
                if i == j or abs(float(y[i] - y[j])) <= min_gap:
                    continue
                cross.append((i, j) if y[i] > y[j] else (j, i))
        n_within = (
            int(self.max_pairs)
            if bool(self.within_family_only)
            else int(self.within_family_frac * self.max_pairs)
        )
        rng.shuffle(within)
        rng.shuffle(cross)
        chosen = within[:n_within] + cross[: max(0, self.max_pairs - n_within)]
        if not chosen:
            # Degenerate: fall back to ridge on raw features
            self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
            self.intercept_ = float(np.mean(y))
            self._fallback = Ridge(alpha=1.0).fit(X, y)
            return self
        self._fallback = None
        X_diff = np.vstack([X[i] - X[j] for i, j in chosen])
        y_bin = np.ones(len(chosen), dtype=np.int64)
        # Add reversed pairs for balance
        X_diff = np.vstack([X_diff, -X_diff])
        y_bin = np.concatenate([y_bin, np.zeros(len(chosen), dtype=np.int64)])
        clf = LogisticRegression(
            C=self.C,
            max_iter=500,
            random_state=self.random_state,
        )
        clf.fit(X_diff, y_bin)
        self.coef_ = np.asarray(clf.coef_, dtype=np.float64).ravel()
        self.intercept_ = float(clf.intercept_[0])
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X, dtype=np.float64)
        if getattr(self, "_fallback", None) is not None:
            return self._fallback.predict(X)
        return X @ self.coef_


class RankerWrapper(BaseEstimator, RegressorMixin):
    """Wrap a ranking booster that needs group sizes / relevance bins."""

    def __init__(
        self,
        backend: str = "lgbm",
        *,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        max_depth: int = 4,
        min_child_samples: int = 10,
        n_relevance_bins: int = 5,
        random_state: int = 0,
    ) -> None:
        self.backend = backend
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.n_relevance_bins = n_relevance_bins
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=False)
        groups = np.asarray(groups) if groups is not None else np.zeros(len(y), dtype=int)
        # Quantile bins on training labels only
        n_bins = max(2, int(self.n_relevance_bins))
        try:
            ranks = np.argsort(np.argsort(y))
            rel = np.minimum((ranks * n_bins) // max(len(y), 1), n_bins - 1).astype(int)
        except Exception:  # noqa: BLE001
            rel = np.zeros(len(y), dtype=int)

        order = np.argsort(groups, kind="mergesort")
        Xo, yo, go, relo = X[order], y[order], groups[order], rel[order]
        _, group_counts = np.unique(go, return_counts=True)

        if self.backend == "lgbm":
            lgb = _try_import("lightgbm")
            if lgb is None:
                raise ImportError("lightgbm")
            train = lgb.Dataset(Xo, label=relo, group=group_counts.tolist())
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "min_child_samples": self.min_child_samples,
                "verbosity": -1,
                "seed": self.random_state,
            }
            self.model_ = lgb.train(params, train, num_boost_round=int(self.n_estimators))
            self._predict_fn = lambda Xt: self.model_.predict(Xt)
        elif self.backend == "xgb":
            xgb = _try_import("xgboost")
            if xgb is None:
                raise ImportError("xgboost")
            dtrain = xgb.DMatrix(Xo, label=relo)
            dtrain.set_group(group_counts.tolist())
            params = {
                "objective": "rank:pairwise",
                "eta": self.learning_rate,
                "max_depth": self.max_depth,
                "seed": self.random_state,
                "verbosity": 0,
            }
            self.model_ = xgb.train(params, dtrain, num_boost_round=int(self.n_estimators))
            self._predict_fn = lambda Xt: self.model_.predict(xgb.DMatrix(Xt))
        else:
            raise ValueError(self.backend)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X, dtype=np.float64)
        return np.asarray(self._predict_fn(X), dtype=np.float64)


class StackRidge(BaseEstimator, RegressorMixin):
    """Ridge on base-model predictions (expects precomputed stacked features)."""

    def __init__(self, *, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y, groups=None):
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=False)
        self.model_ = Ridge(alpha=self.alpha).fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(check_array(X, dtype=np.float64))


class RankAverage(BaseEstimator, RegressorMixin):
    """Average ranks of columns of X (each column = a base prediction)."""

    def fit(self, X, y, groups=None):
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=False)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "n_features_in_")
        X = check_array(X, dtype=np.float64)
        ranks = np.zeros_like(X)
        for j in range(X.shape[1]):
            col = X[:, j]
            ranks[:, j] = np.argsort(np.argsort(col)).astype(np.float64)
        return ranks.mean(axis=1)


class MLPRegressorTorch(BaseEstimator, RegressorMixin):
    """Small MLP with optional pairwise ranking loss."""

    def __init__(
        self,
        *,
        hidden: Sequence[int] = (64, 64),
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        lr: float = 1e-3,
        max_epochs: int = 80,
        patience: int = 10,
        n_ensemble: int = 5,
        loss: str = "mse",
        random_state: int = 0,
    ) -> None:
        self.hidden = list(hidden)
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.n_ensemble = n_ensemble
        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y, groups=None):
        torch = _try_import("torch")
        if torch is None:
            raise ImportError("torch")
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=False)
        groups = np.asarray(groups) if groups is not None else None
        self.models_ = []
        for seed_i in range(int(self.n_ensemble)):
            model = self._fit_one(torch, X, y, groups, seed=self.random_state + seed_i)
            self.models_.append(model)
        return self

    def _build_net(self, torch, n_in: int):
        layers: list[Any] = []
        prev = n_in
        for h in self.hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        return torch.nn.Sequential(*layers)

    def _fit_one(self, torch, X, y, groups, *, seed: int):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        n = len(y)
        # Inner holdout by family if possible
        if groups is not None and len(np.unique(groups)) >= 3:
            held = rng.choice(np.unique(groups))
            va = groups == held
            tr = ~va
        else:
            idx = np.arange(n)
            rng.shuffle(idx)
            cut = max(1, int(0.2 * n))
            va = np.zeros(n, dtype=bool)
            va[idx[:cut]] = True
            tr = ~va
        device = torch.device("cpu")
        net = self._build_net(torch, X.shape[1]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Xt = torch.tensor(X[tr], dtype=torch.float32, device=device)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=device).view(-1, 1)
        Xv = torch.tensor(X[va], dtype=torch.float32, device=device)
        yv = torch.tensor(y[va], dtype=torch.float32, device=device).view(-1, 1)
        best_state = None
        best_val = float("inf")
        bad = 0
        for _ in range(int(self.max_epochs)):
            net.train()
            opt.zero_grad()
            pred = net(Xt)
            if self.loss == "rank":
                loss = self._pairwise_loss(torch, pred, yt)
            else:
                loss = torch.nn.functional.mse_loss(pred, yt)
            loss.backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                pv = net(Xv)
                if self.loss == "rank":
                    vloss = float(self._pairwise_loss(torch, pv, yv).item())
                else:
                    vloss = float(torch.nn.functional.mse_loss(pv, yv).item())
            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= int(self.patience):
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    @staticmethod
    def _pairwise_loss(torch, pred, y, max_pairs: int = 512):
        pred = pred.view(-1)
        y = y.view(-1)
        n = pred.numel()
        if n < 2:
            return (pred * 0).sum()
        # Sample pairs
        i = torch.randint(0, n, (max_pairs,))
        j = torch.randint(0, n, (max_pairs,))
        mask = y[i] != y[j]
        if mask.sum() == 0:
            return (pred * 0).sum()
        i, j = i[mask], j[mask]
        s_ij = pred[i] - pred[j]
        y_ij = torch.sign(y[i] - y[j])
        # RankNet
        return torch.nn.functional.binary_cross_entropy_with_logits(
            s_ij, (y_ij > 0).float()
        )

    def predict(self, X):
        torch = _try_import("torch")
        check_is_fitted(self, "models_")
        X = check_array(X, dtype=np.float64)
        device = torch.device("cpu")
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        preds = []
        with torch.no_grad():
            for net in self.models_:
                preds.append(net(Xt).cpu().numpy().ravel())
        return np.mean(np.stack(preds, axis=0), axis=0)


def make_model(
    name: str,
    *,
    random_state: int = 0,
    params: Mapping[str, Any] | None = None,
) -> Any:
    """Construct a named estimator. Raises ImportError if optional dep missing."""
    params = dict(params or {})
    p = params

    if name == "mean":
        return MeanPredictor()
    if name == "median":
        return MedianPredictor()
    if name == "ridge":
        return Ridge(alpha=float(p.get("alpha", 1.0)), random_state=random_state)
    if name == "elasticnet":
        return ElasticNet(
            alpha=float(p.get("alpha", 1.0)),
            l1_ratio=float(p.get("l1_ratio", 0.5)),
            random_state=random_state,
            max_iter=5000,
        )
    if name == "kernel_ridge":
        return KernelRidge(
            alpha=float(p.get("alpha", 1.0)),
            kernel="rbf",
            gamma=float(p.get("gamma", 0.1)),
        )
    if name == "svr":
        return SVR(
            C=float(p.get("C", 1.0)),
            epsilon=float(p.get("epsilon", 0.1)),
            gamma=float(p.get("gamma", 0.1)),
        )
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=int(p.get("n_neighbors", 5)))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=p.get("max_depth"),
            min_samples_leaf=int(p.get("min_samples_leaf", 2)),
            random_state=random_state,
            # Nested LOFO already multiplies work; keep trees single-process.
            n_jobs=1,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(p.get("n_estimators", 250)),
            max_depth=p.get("max_depth"),
            min_samples_leaf=int(p.get("min_samples_leaf", 2)),
            random_state=random_state,
            n_jobs=1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_depth=p.get("max_depth", 5),
            learning_rate=float(p.get("learning_rate", 0.1)),
            min_samples_leaf=int(p.get("min_samples_leaf", 10)),
            max_iter=int(p.get("max_iter", 150)),
            random_state=random_state,
        )
    if name == "gp_ard":
        use_ard = bool(p.get("ard", True))
        n_features = int(p.get("n_features_hint", 15))
        if use_ard:
            length = np.ones(n_features)
            kernel = ConstantKernel(1.0) * Matern(
                length_scale=length, nu=2.5
            ) + WhiteKernel(1e-3)
        else:
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
                1e-3
            )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=float(p.get("alpha", 1e-6)),
            normalize_y=True,
            random_state=random_state,
            n_restarts_optimizer=1,
        )
    if name == "lightgbm":
        lgb = _try_import("lightgbm")
        if lgb is None:
            raise ImportError("lightgbm")
        return lgb.LGBMRegressor(
            n_estimators=int(p.get("n_estimators", 200)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            num_leaves=int(p.get("num_leaves", 15)),
            min_child_samples=int(p.get("min_child_samples", 10)),
            subsample=float(p.get("subsample", 0.8)),
            colsample_bytree=float(p.get("colsample_bytree", 0.8)),
            random_state=random_state,
            verbosity=-1,
        )
    if name == "catboost":
        cb = _try_import("catboost")
        if cb is None:
            raise ImportError("catboost")
        return cb.CatBoostRegressor(
            iterations=int(p.get("iterations", 200)),
            depth=int(p.get("depth", 4)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            l2_leaf_reg=float(p.get("l2_leaf_reg", 3.0)),
            random_seed=random_state,
            verbose=False,
        )
    if name == "xgboost":
        xgb = _try_import("xgboost")
        if xgb is None:
            raise ImportError("xgboost")
        return xgb.XGBRegressor(
            n_estimators=int(p.get("n_estimators", 200)),
            max_depth=int(p.get("max_depth", 3)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            min_child_weight=float(p.get("min_child_weight", 5)),
            subsample=float(p.get("subsample", 0.8)),
            colsample_bytree=float(p.get("colsample_bytree", 0.8)),
            random_state=random_state,
            verbosity=0,
        )
    if name == "lgbm_lambdarank":
        return RankerWrapper(
            backend="lgbm",
            n_estimators=int(p.get("n_estimators", 150)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            num_leaves=int(p.get("num_leaves", 15)),
            min_child_samples=int(p.get("min_child_samples", 10)),
            n_relevance_bins=int(p.get("n_relevance_bins", 5)),
            random_state=random_state,
        )
    if name == "xgb_rank_pairwise":
        return RankerWrapper(
            backend="xgb",
            n_estimators=int(p.get("n_estimators", 150)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            max_depth=int(p.get("max_depth", 4)),
            n_relevance_bins=int(p.get("n_relevance_bins", 5)),
            random_state=random_state,
        )
    if name == "pairwise_logistic":
        return PairwiseLogisticRanker(
            C=float(p.get("C", 1.0)),
            random_state=random_state,
            max_pairs=int(p.get("max_pairs", 2000)),
            within_family_frac=float(p.get("within_family_frac", 1.0)),
            within_family_only=bool(p.get("within_family_only", True)),
            min_abs_delta=float(p.get("min_abs_delta", 0.0)),
        )
    if name == "tabpfn":
        if not _tabpfn_ready():
            raise ImportError(
                "tabpfn requires TABPFN_TOKEN or cached PriorLabs license; skipped"
            )
        tabpfn = _try_import("tabpfn")
        if tabpfn is None:
            raise ImportError("tabpfn")
        logger.info(
            "TabPFN: pretrained weights may be non-commercial; using cached license token."
        )
        cls = getattr(tabpfn, "TabPFNRegressor", None)
        if cls is None:
            raise ImportError("tabpfn regressor API not found")
        try:
            return cls(device="cpu")
        except TypeError:
            return cls()
    if name == "mlp_mse":
        return MLPRegressorTorch(
            hidden=p.get("hidden", [64, 64]),
            dropout=float(p.get("dropout", 0.1)),
            weight_decay=float(p.get("weight_decay", 1e-4)),
            lr=float(p.get("lr", 1e-3)),
            max_epochs=int(p.get("max_epochs", 80)),
            patience=int(p.get("patience", 10)),
            n_ensemble=int(p.get("n_ensemble", 5)),
            loss="mse",
            random_state=random_state,
        )
    if name == "mlp_rank":
        return MLPRegressorTorch(
            hidden=p.get("hidden", [64, 64]),
            dropout=float(p.get("dropout", 0.1)),
            weight_decay=float(p.get("weight_decay", 1e-4)),
            lr=float(p.get("lr", 1e-3)),
            max_epochs=int(p.get("max_epochs", 80)),
            patience=int(p.get("patience", 10)),
            n_ensemble=int(p.get("n_ensemble", 3)),
            loss="rank",
            random_state=random_state,
        )
    if name == "stack_ridge":
        return StackRidge(alpha=float(p.get("alpha", 1.0)))
    if name == "rank_average":
        return RankAverage()
    if name == "tabm_or_realmlp":
        raise ImportError("tabm_or_realmlp not installed")
    raise ValueError(f"Unknown model '{name}'")


def needs_groups(name: str) -> bool:
    return name in {
        "lgbm_lambdarank",
        "xgb_rank_pairwise",
        "pairwise_logistic",
        "mlp_mse",
        "mlp_rank",
    }


def is_constant_ranker(name: str) -> bool:
    return name in {"mean", "median"}


def is_ranking_model(name: str) -> bool:
    return name in {
        "lgbm_lambdarank",
        "xgb_rank_pairwise",
        "pairwise_logistic",
        "mlp_rank",
        "rank_average",
    }
