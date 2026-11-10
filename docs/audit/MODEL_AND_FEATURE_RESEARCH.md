# Model & Feature Research Assessment

Constraints: n≈108 candidates, 12 families, noisy labels, LOFO. Prefer classical tabular methods.

---

## A. Feature engineering

| Idea | LOFO-valid? | Priority | Rationale |
|------|-------------|----------|-----------|
| Reward coefficient **ratios** / normalized weights | Yes | High | Increases within-family contrast without test labels |
| Term contribution stats on fixed probes (mean/var/tails) | Yes | High | Already partially in math/behavioral; extend tails |
| Local reward sensitivity / finite differences on probes | Yes | Medium | Costly but inductive |
| Safety-critical state slices | Yes | Medium | Aligns with collision/timeout modes |
| Early learning-curve descriptors (deltas across 2k/5k/12k) | Yes (budgeted) | High | Early already helps; deltas may add within signal |
| Within-family z-scoring of features using **test family** | **No** (transductive) | — | Only OK if deployment is batch-transductive and declared |
| Train-only family residual targets | Careful | Medium | Cannot use test-family target mean |
| Interaction features | Yes | Low | Historically near-noise; ablation only |
| Drop near-constant-within-family behavioral cols | Yes | High | ~80% behavioral cols ~0 within-std |

**Cost-benefit:** Prefer transforming existing probes/early artifacts over new PPO.

---

## B. Ranking models

| Method | Realistic at this n? | Notes |
|--------|----------------------|-------|
| Pairwise logistic on Δφ within family | Yes | Simple, controllable leakage |
| RankSVM | Marginal | Kernel + small n |
| XGB/LGBM LambdaMART / pairwise | Yes (already wired) | Group = family; hold out all test pairs |
| CatBoost YetiRank | Optional | Extra dep; prior CatBoost regression lost confirm |
| Listwise deep RankNet | No | Overfit risk |
| Neural RankNet/ListNet deep | No | Unless tiny MLP + heavy reg |

**Pair construction rules:**
- Pairs only within family  
- All pairs of held-out family excluded from train  
- Down-weight / drop pairs with \|ΔY\| < seed SE  
- Report metrics at **family** level, not pair level  

**Expectation:** modest lift on identifiable families; little help on flat families.

---

## C. Regression models

| Model | Verdict |
|-------|---------|
| ExtraTrees | **Keep as locked baseline** |
| Ridge | Keep as sanity / linear floor |
| RF / HGB / XGB / CatBoost | Fair benchmark only; CatBoost already weaker on 2D confirm emb |
| GP | Useful for uncertainty if dims reduced (≤15) |
| SVR | Optional |
| GAM | Interpretability secondary |
| Shallow MLP / deep ensembles | **Not justified** as primary; optional stress test with strong reg |
| TabPFN / large nets | Skip unless deps + tiny feature set |

---

## D. Hierarchical / within-family modeling

| Approach | LOFO validity |
|----------|---------------|
| Predict family mean (from features) + within residual model | OK if residual model uses only inductive features |
| Mixed-effects with family random intercept | **Invalid** for unseen family unless random effect marginalized carefully; family ID unavailable at test |
| Two-stage: screen families then rank within | OK as **system** design, not as leaking test stats |
| Pairwise within-family ranker | OK under pair holdout |

---

## E. Multi-task learning

Targets are highly structured (on CrowdNav, SR≈1−TR when CR≈0; 2D has more collision mass).  
Multi-output trees/GPs may regularize representation but with n=108 risk of fitting noise.  
**Priority:** low–medium ablation; do not replace `success_rate` as selection target without task justification.

---

## F. Uncertainty

| Source | Use |
|--------|-----|
| Seed aleatoric \(\hat\sigma_i\) | Adaptive replication; soft pairwise labels |
| Ensemble / tree variance | Acquisition for AL |
| GP posterior | Small-dim feature sets |
| Conformal (family-aware) | Prefer for intervals with LOFO coverage claims |

Distinguish: uncertainty about **mean reward quality** vs **PPO seed noise**.

---

## G. Active learning / selection

Not implemented. Scientifically valid next step **after** Stage 1–2:

1. Simulate offline: given frozen OOF-style surrogate, select top-k / EI / UCB under budget.  
2. Measure true regret vs random / vs full eval.  
3. Only then run online AL with new PPO — with a **held-out family block** never used for model selection.

Avoid fitting AL policy on the same 12 folds used to pick the surrogate.

---

## Cost-benefit ranking of research bets

1. Identifiability + metric redefinition (science clarity)  
2. Feature cleanup / ratios / early deltas (signal)  
3. Pairwise ranking fair test (objective match)  
4. Adaptive seeds (label quality)  
5. Uncertainty-aware selection sim (Objective B)  
6. Heavier models / multi-task / deep nets (low EV)
