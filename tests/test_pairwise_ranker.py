"""Pairwise logistic ranker: within-family-only + min gap."""

from __future__ import annotations

import numpy as np

from rsr.surrogate.model_zoo import PairwiseLogisticRanker


def test_pairwise_logistic_within_family_only_ranks():
    rng = np.random.default_rng(0)
    X = []
    y = []
    g = []
    for fam in range(4):
        for i in range(6):
            z = rng.normal(size=5)
            # Strong linear signal
            score = float(z[0] + 0.2 * fam)
            X.append(z)
            y.append(score)
            g.append(f"f{fam}")
    X = np.asarray(X)
    y = np.asarray(y)
    g = np.asarray(g)
    model = PairwiseLogisticRanker(
        C=1.0, random_state=0, within_family_only=True, min_abs_delta=0.01
    )
    model.fit(X, y, groups=g)
    pred = model.predict(X)
    # Should correlate with y overall
    from scipy.stats import spearmanr

    sp = spearmanr(y, pred).correlation
    assert sp is not None and sp > 0.3
