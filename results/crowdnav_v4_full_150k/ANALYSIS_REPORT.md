# CrowdNav Full Run Analysis — `v4_full_150k`

**Date analyzed:** 2026-11-06  
**Campaign:** `experiments/crowdnav_v4_full_150k/` → `results/crowdnav_v4_full_150k/`  
**Config:** CrowdNav-aware 72-candidate pool, PPO **150k**, early checkpoints 5k/12k/25k, 3 seeds  

---

## 1. Executive verdict

The full CrowdNav campaign **completed successfully** and is scientifically much healthier than the archived 50k pilot (`crowdnav_v1_50k`).

| Layer | Verdict |
|-------|---------|
| Phase 1 PPO | **216/216 completed**, 0 failed; wall ≈ **6.2 h** |
| Label quality | Mean success ≈ **0.43** (was ≈0.10 at 50k); all 12 families identifiable |
| Surrogate (ExtraTrees / early+math+behavioral) | **Within-family Spearman ≈ 0.75**, Top-3 hit **0.92** |
| Selection | k=10 best-regret ≈ **0.044** vs oracle (beats random); oracle #1 by k≈20–30 |
| Main weakness | Family `time_pressure` (within Spearman ≈ 0.12; Top-3 miss) |

**Bottom line:** On CrowdNav@150k, the operating ExtraTrees surrogate is **strong at within-family ranking** (the metric that matters for reward triage) and weaker at pooled ranking than the sparse 50k pilot — which is expected once performance becomes continuous rather than “almost everyone fails.”

---

## 2. Campaign integrity

| Check | Result |
|-------|--------|
| Candidates | 72 (12 families × 6) |
| Seeds | 3 → **216** jobs |
| Timesteps | 150000 for all rows |
| Completed | 216 |
| Failed | 0 |
| Outlier seeds flagged | **none** (empty CSV) |
| Dataset validation | `ok: true` |
| Phase-1 wall clock | **22283 s ≈ 6.19 h** |
| Mean train time / job | ≈ **6.2 min** |
| Surrogate feature cost | 12k early steps accounted |

Manifest description: *CrowdNav-aware 72 pool, PPO 150k, human proximity/intrusion/crowded_progress*.

---

## 3. Task-performance landscape

### Overall (seed-mean per candidate)

| Metric | Value |
|--------|------:|
| Mean success rate | **0.435** |
| Std | 0.207 |
| Min / Max | 0.00 / 0.81 |
| Span | **0.85** |
| Fraction SR = 0 | 2.8% |
| Mean collision rate | ≈ **0.003** (near-dead) |
| Mean timeout rate | **0.562** |
| corr(SR, TR) | **−1.00** |
| Between-family variance share | ≈ **64%** |
| Median within-family SR range | ≈ **0.35** |
| Close pairs (\|Δ\| < SE) | 30% |
| Close pairs (\|Δ\| < 2·SE) | 44% |

**Interpretation**
- Failures are still mostly **timeouts**, not collisions — same structural issue as the 50k pilot, but absolute success is much higher.
- Because CR ≈ 0, reporting CR adds little discriminative value; SR and TR are mirrors.
- Within-family ranges are large enough that **all 12 families pass the identifiability gate** (unlike 2D/`distance_dense` problems).

### Family means (seed-mean SR)

| Family | Mean SR | Within range |
|--------|--------:|-------------:|
| progress_heavy | 0.70 | 0.17 |
| progress_dense | 0.65 | 0.35 |
| time_pressure | 0.61 | 0.22 |
| crowded_progress | 0.57 | 0.12 |
| balanced | 0.50 | 0.37 |
| human_clearance | 0.43 | 0.35 |
| nonlinear_progress | 0.38 | 0.29 |
| safety_moderate | 0.38 | 0.45 |
| efficiency | 0.31 | 0.30 |
| risk_sensitive | 0.30 | 0.41 |
| energy_aware | 0.20 | 0.48 |
| weak_contrast | 0.19 | 0.50 |

**Top candidates:** `progress_dense_03` (0.81), then several `progress_heavy_*` (~0.75–0.76).  
**Worst:** `energy_aware_05` (0.0), `weak_contrast_00` (0.0).

Progress-oriented families dominate; energy/weak-contrast families underperform — coherent with navigation under timeout pressure.

---

## 4. Surrogate results (LOFO, 3 repeats)

**Model:** ExtraTrees  
**Features:** `early+math+behavioral`  
**Target:** `success_rate`  

| Metric | Repeat mean |
|--------|------------:|
| Pooled Spearman | **0.653** |
| Within-family Spearman mean | **0.746** |
| Within-family Spearman std | 0.234 |
| Within-family Top-3 hit | **0.917** |
| Top-10 regret | 0.086 |
| MAE | 0.124 |
| Bootstrap within Spearman 90% CI | **[0.65, 0.85]** |
| Bootstrap pooled Spearman 90% CI | **[0.38, 0.86]** |

Repeats are stable (within mean 0.741 / 0.744 / 0.753).

### Per-family OOF Spearman

| Family | Mean Spearman | Top-3 hit | SR range |
|--------|--------------:|----------:|---------:|
| time_pressure | **0.12** | **0.0** | 0.22 |
| balanced | 0.49 | 1.0 | 0.37 |
| human_clearance | 0.66 | 1.0 | 0.35 |
| safety_moderate | 0.71 | 1.0 | 0.45 |
| risk_sensitive | 0.75 | 1.0 | 0.41 |
| weak_contrast | 0.77 | 1.0 | 0.50 |
| crowded_progress | 0.79 | 1.0 | 0.12 |
| progress_heavy | 0.87 | 1.0 | 0.17 |
| efficiency | 0.90 | 1.0 | 0.30 |
| nonlinear_progress | 0.94 | 1.0 | 0.29 |
| progress_dense | 0.94 | 1.0 | 0.35 |
| energy_aware | **1.00** | 1.0 | 0.48 |

**11/12 families** have Top-3 hit = 1. Only `time_pressure` fails ranking/selection.

---

## 5. Comparison to prior campaigns

| Campaign | Env / budget | Pooled ρ | Within ρ | Top-3 | Mean SR |
|----------|--------------|---------:|---------:|------:|--------:|
| 2D confirm (locked) | Toy 2D / 50k | **0.935** | 0.648 | 0.833 | higher / different task |
| CrowdNav v1 archive | CrowdNav / **50k** | **0.860** | 0.533 | 0.833 | ≈ **0.10** |
| **CrowdNav v4 (this)** | CrowdNav / **150k** | 0.653 | **0.746** | **0.917** | ≈ **0.43** |

### How to read the pooled drop (v1 → v4)

Not a regression of the method in the sense that matters:
- At 50k, labels were sparse/timeout-dominated; family means were almost separable → easy **pooled** ranking, hard **within** ranking.
- At 150k, agents actually succeed more; within-family diversity is real → **within** ranking improves (0.53 → 0.75), while pooled becomes harder (0.86 → 0.65).
- Primary research estimand for triage is **within-family** / Top-3 — v4 is the better scientific setting.

---

## 6. Implications for the thesis claims

**Supported**
1. Full CrowdNav@150k pipeline is runnable and reproducible (versioned root `v4_full_150k`).
2. ExtraTrees + early+math+behavioral transfers as operating surrogate under LOFO.
3. Within-family ranking on CrowdNav is **stronger** than in the sparse 50k pilot and competitive with / better than 2D within-family mean.
4. Surrogate triage is useful: Top-3 hit 0.92; Top-10 regret ~0.09 on a mean-SR≈0.43 pool.

**Caveats**
1. Collision rate remains uninformative; do not over-claim multi-metric safety prediction.
2. Pooled Spearman alone would **mislead** (looks worse than v1 while within got better).
3. `time_pressure` is a **diagnosed** failure mode (transfer + near-ties; see case study) — not an open mystery.
4. Early features still cost 12k PPO steps; not a zero-cost oracle.
5. LOFO n_families=12 → bootstrap CIs still wide for pooled.

**Do not claim**
- Automatic transfer from 2D numbers to CrowdNav without stating both tables.
- That the surrogate replaces full PPO for final certification.
- That ranking models / feature extras are needed here (not re-tested on v4 in this report).

---

## 7. Follow-ups completed / remaining

**Done**
1. **Case study `time_pressure`** → [`time_pressure_case_study.md`](time_pressure_case_study.md): transfer failure on near-tie top band; local early features correlate with SR (ρ≈0.94) but LOFO does not transfer; `weight_time` tiny vs goal.
2. **Selection sim** → [`selection_sim/REPORT.md`](selection_sim/REPORT.md): at **k=10** surrogate best ≈0.765 vs random ≈0.72 (regret vs oracle ≈0.044); oracle #1 enters shortlist by **k≈20–30**.

**Done (cont.)**
3. **Dataset-size curve** → [`dataset_size_curve_REPORT.md`](dataset_size_curve_REPORT.md): plateau by train_n≈24–44; full 66 only +0.03–0.04 within.

**Still optional**
4. Always publish pooled **and** within (+ Top-3) for CrowdNav vs 2D.
5. Optional Phase-3.2 pairwise contrast on v4 (0 new PPO).
6. Adaptive seeds only if separating `_03/_02/_01` scientifically.

---

## 8. Artifact index

| Path | Content |
|------|---------|
| `experiments/crowdnav_v4_full_150k/dataset/` | 216-row performance dataset |
| `experiments/crowdnav_v4_full_150k/model_comparison/` | LOFO metrics, OOF preds |
| `results/crowdnav_v4_full_150k/model_comparison_report.md` | Auto report |
| `results/crowdnav_v4_full_150k/leaderboard.csv` | Summary metrics |
| `results/crowdnav_v4_full_150k/figures/` | CI / top-10 / cost plots |
| `results/crowdnav_v4_full_150k/ANALYSIS_REPORT.md` | This analysis |
| `results/crowdnav_v4_full_150k/time_pressure_case_study.md` | LOFO failure diagnosis |
| `results/crowdnav_v4_full_150k/selection_sim/` | Budget/regret selection sim |
| `results/crowdnav_v4_full_150k/dataset_size_curve.csv` | Train-n vs LOFO accuracy |
| `results/crowdnav_v4_full_150k/dataset_size_curve_REPORT.md` | Plateau / cut recommendation |
