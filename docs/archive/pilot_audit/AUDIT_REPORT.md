# Phase 2 Statistical Audit

> Audit of the 32-candidate pilot (4-fold GroupKFold). Superseded by the LOFO scale-up; kept for methodology history.

Independent methodology audit of reward-performance surrogate evaluation.
All numerical claims below were recomputed from
`experiments/phase2/predictions/all_predictions.csv` via
`scripts/audit_phase2_statistics.py` (executed).

Artifacts archived under `docs/archive/pilot_audit/` (formerly `results/phase2_audit/`).

---

## 1. Critical findings

### C1. Pooled OOF Spearman ≠ mean of fold Spearmans (and both were conflated in interpretation)

**What the code does**

- Per-fold metrics: `src/surrogate/training.py` `run_grouped_cv` → `evaluate_predictions(y[te], pred)` with `split="grouped_cv"`.
- Pooled OOF metric: same function on concatenated OOF vectors with `split="grouped_cv_oof"` (lines 259–273).
- Report headlines use **pooled** OOF (`phase2_analysis.oof_metrics` filters `grouped_cv_oof`).
- `phase2_fold_sensitivity.csv` is the **nan-aware mean/std of per-fold Spearmans** (`data_sensitivity_from_folds`).

**Recomputation**

- Pooled Spearman recomputed from saved OOF predictions matches `surrogate_metrics.csv` exactly (max |diff| = 0).
- Fold-mean/std matches `phase2_fold_sensitivity.csv` (numerical noise ~1e-16).

**Why they differ**

1. Spearman is nonlinear: mean of fold correlations ≠ correlation of pooled pairs.
2. Folds have unequal sizes (n = 6, 9, 9, 8 for success_rate) and unequal family composition.
3. Within small folds, ranks are coarse; ties and near-ties inflate fold-level variance.
4. For constant predictors, fold Spearman is **undefined (NaN)** while pooled Spearman can be a large negative number (see C2).

**Which should be primary**

- **Primary for “can we rank candidates across the study?”:** pooled grouped OOF Spearman / Kendall / MAE on the full OOF vector, **always accompanied by** fold-level mean±std (or all fold values).
- **Primary for “is this stable under held-out families?”:** fold-level distribution; a high pooled score with fold_std ≳ 0.4–0.5 is **not** reliable.
- Do **not** replace pooled with fold-mean silently; report both.

**Appropriateness**

- Pooled OOF is statistically legitimate for a single overall association under the grouped CV design.
- Averaging fold Spearmans is also legitimate as a stability summary, but with n_test ∈ {6…9} each fold Spearman has huge sampling variance; the mean of four noisy correlations is not a precise estimand.
- Small folds + possible prediction ties make fold Spearman fragile; that is why many “best” configs look strong on one fold and weak/negative on others.

### C2. Mean/Median baseline pooled Spearmans are artifacts, not meaningful ranking skill

**Mechanism (executed on success_rate / static)**

| Fold | Held-out families | Mean baseline pred (constant) | Within-fold Spearman |
|---|---|---|---|
| 0 | balanced | 0.169 | **undefined (constant pred)** |
| 1 | near_zero, safety_heavy | 0.284 | undefined |
| 2 | efficiency, progress_heavy | 0.227 | undefined |
| 3 | sparse_goal, weak_unbalanced | 0.274 | undefined |

Pooled OOF Spearman for mean/median = **−0.518**.

Fold-level correlation between (constant fold prediction) and (held-out family mean label) = **−1.0**.

**Interpretation**

- Within a fold, Spearman with a constant prediction is mathematically undefined (`scipy` `ConstantInputWarning`; audit marks status `constant`).
- Across folds, GroupKFold leaves out families with different mean success. The training-set mean used as the prediction is roughly “global mean without those families,” which is **anti-correlated** with the held-out family’s mean when high-performing families are left out (e.g. leave out `balanced` → train mean low, but test labels high).
- Therefore the reported negative baseline Spearmans are **design artifacts of pooling fold-wise constants**, not evidence that the mean predictor “ranks badly.”

**Proper baseline comparisons**

| Goal | Appropriate baseline metric |
|---|---|
| Point accuracy | MAE / RMSE vs mean and median baselines (these are well-defined) |
| Ranking | Do **not** use Spearman of constant predictors; use chance ranking / random ranking null, or compare only models with non-degenerate score variation; report “ranking N/A” for mean/median |
| Surrogate claim | Require MAE ≤ mean baseline **and** pooled Spearman with fold stability for models with varying predictions |

### C3. Random holdout massively overstates generalization vs grouped CV

Recomputed mean Spearman (non-mean/median models):

| target | grouped OOF | random holdout |
|---|---:|---:|
| success_rate | −0.044 | **0.866** |
| collision_rate | −0.042 | **0.734** |
| timeout_rate | 0.037 | **0.817** |
| mean_path_length | 0.175 | **0.637** |
| mean_goal_time | 0.112 | −0.021 |

Random splitting is **not** an acceptable generalization test here. Same-family candidates are highly similar; leakage across the split inflates ranking metrics. Grouped CV is the correct primary protocol (and was correctly designated as such in config), but secondary random numbers must not be treated as supportive evidence.

### C4. Apparent “best” success-rate model is unstable across folds

Reported best: `gaussian_process` + `behavioral`, pooled Spearman **0.307**.

Fold Spearmans: **[−0.086, 0.709, 0.087, 0.855]**  
fold_mean=0.392, fold_std=**0.461**.

Two folds dominate; two folds are ~0. This is **not** consistent usefulness for selecting unseen reward families.

Several other success_rate configs have fold_std > 0.65 with pooled scores near 0.1 (see `unstable_configs.csv`).

### C5. Family one-hot features are incompatible with the grouped generalization claim

Static/hybrid features include:
`family_balanced`, `family_efficiency`, … (all 7 families)
— `src/features/static_features.py` (`include_family_one_hot=True`, default).

Under GroupKFold, the held-out family’s one-hot column is **all zeros in training** and becomes 1 only at test time. That:

- cannot teach a transferable family effect for unseen families;
- injects zero-variance columns into train scalers/linear models;
- leaks the **family vocabulary** into the feature schema even when labels are held out.

For the scientific question (“unseen reward function / unseen family”), family one-hots should be **removed** from X in grouped evaluation (or replaced with family-agnostic descriptors).

---

## 2. Important findings

### I1. Grouped CV family separation is correctly implemented

`GroupKFold(n_splits=4)` on `family_id` yields disjoint train/test families every fold (verified; overlaps empty). Example fold 0 tests only `balanced` (n=6).

Implementation: `src/surrogate/training.py` `run_grouped_cv`.

### I2. Preprocessing / hyperparameter fitting discipline is mostly correct

- `StandardScaler` / Ridge / GP live in sklearn `Pipeline` and are fit inside `_fit_predict_fold` on **training fold only** — OK.
- Hyperparameter search `_select_params_grouped` uses GroupKFold **within the training families only** — does not peek at held-out outer-fold labels — OK.
- No separate feature-selection step exists (nothing to leak there).

### I3. OOF coverage

- For success/collision/timeout/path length: exactly **32** candidates, one OOF prediction each, no duplicates.
- For `mean_goal_time`: only **26** candidates — 6 rows dropped because labels are NaN (0% success: all `efficiency_*` plus `near_zero_00/02`). This is documented behavior in training (`_finite_mask`), but reports should state that goal-time results are **conditional on candidates with ≥1 success**.

### I4. Feature-set evidence (pooled Spearman, excl. mean/median)

From `feature_set_pooled_summary.csv`:

- **Static** is best on average for `mean_path_length` (mean 0.37, max 0.58) and competitive for collision/timeout.
- **Behavioral** has the only positive mean for `success_rate` (mean 0.12, max 0.31) and the strongest goal-time max (0.55).
- **Hybrid** is not consistently better; for success_rate mean pooled Spearman is **negative** (−0.11). On this N, concatenation mostly adds variance/noise.

Behavioral probes are not clearly “family-independent magic”: they can help some targets, but given fold instability and small N, they may partly reflect family-specific reward-scale patterns rather than causal probe information. The audit cannot prove causality either way from these data alone.

### I5. Timeout_rate / ridge+static looks unusually fold-stable

`ridge/static` timeout_rate: pooled 0.476; fold Spearmans ≈ [0.66, 0.73, 0.80, 0.76]; fold_std **0.061**.

This is the most fold-stable ranking signal in the study. Still based on tiny folds; treat as **suggestive**, not confirmatory. Note: `phase2_analysis.BASELINES` includes `ridge`, so some report “best surrogate” paths exclude this config — inconsistent presentation (minor/process issue).

### I6. Point prediction remains weak even when ranking looks moderate

Example: GP+behavioral success_rate R² ≈ 0.01 with Spearman 0.31. MAE improvements vs mean baseline are small (best MAE 0.158 vs mean 0.218). Ranking ≠ calibrated performance prediction.

---

## 3. Minor findings

### M1. `BASELINES = {mean, median, ridge}` in `phase2_analysis.py`

Ridge is sometimes a competitive model (timeout). Excluding it from “best surrogate” while including it in model tables creates inconsistent “best” narratives.

### M2. Sensitivity CSV shows NaN Spearman for mean/median

Correct consequence of undefined fold Spearmans; but the Phase 2 report still contrasted against mean-baseline Spearman −0.52 as if it were a fair ranking baseline.

### M3. Duplicate Spearman fields

`regression_metrics` and `ranking_agreement` both compute Spearman (the latter after sign flip for lower-is-better targets). For higher-is-better targets they match; for lower-is-better targets, `spearman` vs `rank_spearman` can differ in intent. Reports primarily used `spearman` without clarifying sign convention for collision/timeout/goal time/path length.

### M4. GP kernel convergence warnings

Noise level hitting lower bound — GP fits are numerically fragile in high dimension / small N.

### M5. Report path absolute Windows paths

Cosmetic only in earlier final report drafts.

---

## 4. Exact implementation references

| Issue | Location |
|---|---|
| Pooled OOF metric aggregation | `src/surrogate/training.py` `run_grouped_cv` ~L259–273 |
| Per-fold metrics | same, ~L217–227 |
| Spearman for possibly constant preds | `src/surrogate/evaluation.py` `regression_metrics` L32–39 |
| Fold sensitivity = mean of fold metrics | `src/surrogate/phase2_analysis.py` `data_sensitivity_from_folds` |
| Report uses pooled OOF only for headlines | `src/surrogate/phase2_analysis.py` `oof_metrics`, `build_phase2_report` |
| Mean/Median constant predictors | `src/surrogate/baselines.py` |
| Family one-hot | `src/features/static_features.py` L101–124; enabled in `feature_builder.build_feature_tables` |
| Inner grouped HP search on train only | `src/surrogate/training.py` `_select_params_grouped` |
| Drop NaN targets (goal time) | `src/surrogate/training.py` `_finite_mask` / `run_grouped_cv` L182–187 |
| Random split secondary | `run_random_split` L277+ |

---

## 5. Corrected statistical tables (summary)

Full tables: `results/phase2_audit/corrected_primary_metrics.csv`.

### Reported “best” configs — pooled vs fold stability (recomputed)

| target | config | pooled Spearman | fold mean | fold std | fold values |
|---|---|---:|---:|---:|---|
| success_rate | GP / behavioral | 0.307 | 0.392 | **0.461** | −0.09, 0.71, 0.09, 0.86 |
| collision_rate | RF / static | 0.469 | 0.343 | 0.343 | 0.35, 0.05, 0.82, 0.15 |
| timeout_rate | ridge / static | 0.476 | 0.736 | **0.061** | 0.66–0.80 |
| mean_goal_time | RF / behavioral | 0.547 | 0.378 | 0.250 | 0.26, 0.57, 0.60, 0.08 (n=26) |
| mean_path_length | ET / static | 0.580 | 0.289 | **0.502** | 0.31, 0.35, −0.37, 0.86 |

**Reading rule:** path-length and success-rate “wins” are fold-fragile; timeout ridge/static is the only relatively stable ranking signal.

### Success-rate MAE (fair baseline comparison)

| model | feature_set | MAE | pooled Spearman | fold Spearman std |
|---|---|---:|---:|---:|
| random_forest | hybrid | **0.158** | 0.126 | 0.693 |
| random_forest | behavioral | 0.162 | 0.112 | 0.738 |
| gaussian_process | behavioral | 0.181 | **0.307** | 0.461 |
| median | * | 0.193 | N/A (artifact −0.52) | undefined |
| mean | * | 0.218 | N/A (artifact −0.52) | undefined |

Best MAE configs are **not** the best Spearman config, and the MAE leaders have catastrophic fold Spearman variance.

---

## 6. Research conclusions (direct answers)

1. **Does the study demonstrate that reward performance can be predicted before PPO training?**  
   **Not in a scientifically reliable sense.** There are suggestive associations for some metrics under grouped CV, but success-rate prediction is weak and fold-unstable; random-split numbers are misleadingly optimistic.

2. **For which metrics is there credible evidence?**  
   - **Most credible (still provisional):** `timeout_rate` with ridge+static (stable fold Spearmans).  
   - **Suggestive but unstable:** `collision_rate` RF+static; `mean_goal_time` RF+behavioral (and only on the 26 finite labels); `mean_path_length` ET+static (high fold_std).  
   - **Not credible for selection:** `success_rate` as currently measured.

3. **Is success-rate prediction useful for selecting unseen reward candidates?**  
   **No.** Best pooled Spearman ~0.31 with fold swings from ~−0.09 to ~0.86 is insufficient for triage.

4. **Is Active Learning justified?**  
   **No.** AL would optimize a noisy, fold-unstable surrogate—especially on the metric that matters most for “good rewards” (success). Prioritize methodology fixes and stronger signal first.

5. **Smallest next experiment with most useful new evidence?**  
   Re-run **only** Phase 2 evaluation (no new PPO) with:  
   (a) **drop family one-hots**,  
   (b) report **pooled + per-fold** metrics with undefined ranking for constant baselines,  
   (c) optionally Leave-One-Family-Out (7 folds) instead of 4-fold grouped CV,  
   (d) primary success criterion = MAE vs mean baseline **and** pooled Spearman with fold_std threshold.  
   If after that success-rate pooled Spearman remains ≲ 0.3 with high fold_std, collect more **families** (not just more seeds) before any AL discussion.

---

## 7. Prioritized next-step plan

1. **Fix evaluation reporting** (code): dual reporting of pooled vs fold Spearman; mark constant-baseline ranking as N/A; stop using −0.52 as a ranking baseline.  
2. **Remove family one-hot** from grouped-CV feature matrices; regenerate Phase 2 tables.  
3. **Recompute** surrogates under Leave-One-Family-Out; publish fold table for every target.  
4. **Only if** success-rate ranking becomes stable: increase PPO label budget / family diversity.  
5. **Defer Active Learning** until success/safety ranking clears pre-registered thresholds with stability constraints.

---

## Data availability statement

Used existing files:

- `experiments/phase2/predictions/all_predictions.csv` (OOF + random holdout predictions) — **available**
- `experiments/phase2/surrogate_metrics.csv` — **available**
- `results/phase2_fold_sensitivity.csv` — **available**
- Feature CSVs / meta — **available**

No OOF predictions were invented. Audit script: `scripts/audit_phase2_statistics.py`.
