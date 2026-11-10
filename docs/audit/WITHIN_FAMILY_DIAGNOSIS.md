# Within-Family Ranking Diagnosis

**Primary evidence:** 2D Phase-1 labels + `model_comparison_confirm` OOF for ExtraTrees / `early+math+behavioral` / `success_rate`.  
**Secondary:** CrowdNav v1 noted only where helpful.  
**Claimed brief weakness (0.575 within) is not the 2D confirm mean (0.648); diagnosis uses verified OOF.**

---

## 1. Per-family OOF Spearman (ExtraTrees emb, mean over 3 repeats)

| Family | Mean Spearman | SR range (approx) | Notes |
|--------|--------------:|------------------:|-------|
| distance_dense | **−0.635** | 0.017 | Near-zero SR; metric instability / noise |
| risk_sensitive | 0.394 | 0.217 | Weak ranking |
| nonlinear_progress | 0.430 | 0.210 | Weak ranking |
| progress_heavy | 0.643 | 0.244 | Moderate |
| near_zero | 0.723 | 0.087 | OK but small scale |
| weak_unbalanced | 0.810 | 0.073 | OK but small scale |
| safety_heavy | 0.849 | 0.573 | Strong |
| efficiency | 0.881 | 0.835 | Strong |
| balanced | 0.895 | 0.469 | Strong |
| sparse_goal | 0.905 | 0.113 | Strong |
| energy_aware | 0.935 | 0.831 | Strong |
| potential_shaping | 0.947 | 0.158 | Strong |

**corr(range, Spearman) ≈ 0.45** — range helps but does not fully explain failures.

**Key reinterpretation:** The problem is **not** “the model cannot rank within families.” It ranks well in most families and fails in a **minority** (especially flat `distance_dense` and a couple of mid-range families).

---

## 2. `distance_dense` case (repeat 0)

True SR ∈ [0, 0.017]; predictions ∈ ~[0.02, 0.07].  
Ordering noise among near-failures dominates Spearman. Including this family in an unweighted mean **pulls the headline metric down without reflecting triage utility**.

---

## 3. Variance decomposition (seed-mean SR)

- Between-family SS share ≈ **63.9%**  
- Within-family SS share ≈ **36.1%**  

Supports the pooled-vs-within gap narrative **without** implying uniform within-family incompetence.

---

## 4. Hypothesis assessments

### H1 — Between-family signal dominates
**Partially supported.**  
Evidence: SS between ≈ 64%; pooled Spearman ≫ within mean; family means span ≈ 0.63.  
Experiment: residualize predictions by train-family means (train-only) / report between vs within R².

### H2 — Insufficient within-family feature variation
**Partially supported (feature-set dependent).**  
Evidence: behavioral features — mean within/between std ratio ≈ 0.08; **~80%** features near-zero within-family std. Early features ratio ≈ 0.71 (better). Math/static ≈ 0.4–0.45.  
Experiment: within-family feature std report; engineer ratio/delta features that vary inside families; ablate behavioral-only.

### H3 — Features poorly represent learned PPO behavior
**Partially supported / open.**  
Early features help confirm winner vs static-only historically, but early↔label coupling complicates causal reading. Probe behavioral may miss PPO-specific adaptations.  
Experiment: correlate early learning-curve descriptors with final SR **within** family; compare probe-only vs early-only within-Spearman.

### H4 — Seed noise dominates small differences
**Partially supported.**  
Evidence: ~25% pairs \|Δ\|<SE; ~44% \|Δ\|<2·SE; median seed std ≈ 0.023.  
Does **not** alone explain `risk_sensitive` / `nonlinear_progress` mid-range underperformance.  
Experiment: adaptive extra seeds; seed-bootstrap pairwise agreement rate.

### H5 — Insufficient candidate diversity
**Family-dependent.**  
Some families have tiny ranges (`distance_dense`); others are diverse (`efficiency`, `energy_aware`). Global “108 is not enough” is too coarse.  
Experiment: regenerate **only** flat families with wider weight ranges (new run_id); compare identifiable fraction.

### H6 — Regression objective mismatch
**Untested → plausible, not evidenced as primary.**  
Rankers exist in config (`lgbm_lambdarank`, `xgb_rank_pairwise`, `pairwise_logistic`) but were demoted / not in locked confirm winner set.  
Experiment: fair LOFO benchmark vs ExtraTrees (Stage 2). Expect modest gains at best given label noise.

### H7 — Metric instability
**Supported.**  
Near-constant families yield unstable Spearman; Top-3 ties arbitrary.  
Experiment: range/SE gate; report Winsorized or thresholded within-mean; Kendall with tie handling.

### H8 — Feature redundancy / overfitting
**Plausible.**  
Hybrid/behavioral dims high vs n=108; prune max 40 helps; behavioral near-constant dims waste capacity.  
Experiment: nested FS curves; correlation clustering; prefer early+math with fewer behavioral columns.

---

## 5. Evidence-backed diagnosis (synthesis)

| Factor | Contribution to weak mean within-Spearman |
|--------|-------------------------------------------|
| Flat / near-failure families in the macro average | **High** |
| Genuine hard mid-range families (risk_sensitive, nonlinear_progress) | **Medium** |
| Between-family-dominated representation learning | **Medium** (explains pooled gap) |
| Seed noise on close pairs | **Medium** |
| Wrong loss (MSE vs rank) | **Low–unknown** until fair test |
| Need for deep nets | **Unsupported** at this n |

**Working diagnosis:**  
Improve **estimand and labels** first (identifiability, seeds), then **intra-family feature signal**, then test ranking losses. Do not lead with model complexity.

---

## 6. What would falsify this diagnosis?

- After excluding flat families, within-mean still <0.4 across remaining families → deeper representation failure.  
- Pairwise ranking lifts identifiable-family Spearman by ≥0.1 with family-bootstrap CI excluding 0 → elevate H6.  
- Extra seeds flip >30% of mid-range family orderings → elevate H4 as dominant.
