# Fair Benchmark Protocol

## Purpose

Compare models under **identical information and folds** for reward performance prediction on the locked 2D 108-candidate archive. Separate exploratory fishing from confirmatory claims.

---

## Dataset & folds

| Item | Specification |
|------|----------------|
| Labels | `experiments/phase1/dataset/reward_performance_dataset.csv` after official outlier policy |
| Aggregation | `seed_aggregation = mean` → 1 row/candidate |
| Features | Declare set(s); primary operating: `early+math+behavioral` |
| Outer CV | Leave-one-**family**-out, 12 folds |
| Repeats | 3 RNG repeats (`seed + 1000*r`) |
| Inner HPO | LOFO or GroupKFold **on training families only**; optimize `within_family_spearman` on **identifiable** families if Stage-1 policy exists |
| Preprocessing | Impute / drop-constant / select / scale / target transform: **fit train only** |
| Family one-hot | Off |
| Output dir | New path e.g. `experiments/benchmark_<date>/` — **do not overwrite** `model_comparison_confirm` |

---

## Mandatory models

1. Mean predictor (train-family mean of y)  
2. Ridge (standardized)  
3. **ExtraTrees** — locked reference hyperparameters from confirm **or** re-tuned only inside train folds (report both if retuned)  
4. Best justified additional tree model (one of: HGB / XGB / CatBoost) — single pre-registered choice  
5. Best justified ranking model (pairwise logistic **or** LambdaMART) — single pre-registered choice  
6. Optional uncertainty model (ExtraTrees ensemble variance **or** GP on ≤15 features)

No method may use features or labels unavailable to others at test time.

---

## Metrics (report all)

### Regression
- MAE, RMSE  
- Pooled Spearman, Kendall  

### Ranking / within-family
- Within-family Spearman mean/median/std (**macro**)  
- Same on **identifiable subset** (Stage-1 definition)  
- Within-family Kendall  
- Within-family Top-3 hit  
- Normalized regret@3 within family (macro)

### Selection (pooled — labeled as pooled)
- Top-10 selected mean, top-10 regret, overlap with oracle  

### Uncertainty (if applicable)
- Spearman(uncertainty, \|error\|)  
- Coverage of family-bootstrap or conformal intervals  

### Efficiency
- Fit/predict time  
- `needs_ppo_steps` feature cost  

---

## Statistical comparison

- Primary contrast: model M vs ExtraTrees.  
- Unit: per-family within-Spearman difference.  
- Inferential: bootstrap over 12 families; report mean Δ and CI.  
- **Do not** claim significance from pair-level n.  
- Exploratory extras must be labeled exploratory; no silent max over large grids as primary.

---

## Pre-registration template (fill before run)

```
Target: success_rate
Feature set: early+math+behavioral
Models: [mean, ridge, extra_trees, <tree>, <ranker>]
Primary metric: within_family_spearman_mean on identifiable families
Secondary: top3_hit, regret@3, pooled spearman
n_repeats: 3
Output: experiments/benchmark_YYYYMMDD/
```

---

## Acceptance of a “winner”

A model beats ExtraTrees only if:
1. Pre-registered primary metric improves, **and**
2. Family-bootstrap 90% CI for Δ excludes 0, **or** clearly motivated practical gain (e.g. same accuracy, 0 PPO feature cost), **and**
3. Result reproduces on ≥2/3 repeats with same sign.
