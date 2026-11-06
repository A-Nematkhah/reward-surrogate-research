# CrowdNav Analysis — `v5_invisible_24`

**Date:** 2026-11-06  
**Campaign:** `experiments/crowdnav_v5_invisible_24/` → `results/crowdnav_v5_invisible_24/`  
**Setup:** 12 families × 2 = **24** candidates · PPO **150k** · 3 seeds · **`robot_visible=false`**  
**Surrogate:** ExtraTrees / `early+math+behavioral` / LOFO / 3 repeats

---

## 1. Executive verdict

| Layer | Result |
|-------|--------|
| Phase 1 | **72/72 completed**, 0 failed; phase1 wall ≈ **1.87 h** (pipeline total ≈ **2.16 h**, 4 workers) |
| `robot_visible` | **False** (confirmed in `runs/.../run_config.json`) |
| Labels | Mean SR ≈ **0.158** (vs **0.435** on v4 visible@72) — much harder / collision-dominated |
| Collision / timeout | mean CR ≈ **0.42**, TR ≈ **0.42** (v4 CR was ≈0) |
| Surrogate pooled Spearman | **0.698** |
| Within Spearman | **0.717** (but **n=2/family** → metric is fragile; std **0.689**) |
| Top-3 hit | **1.0** — **degenerate** with family size 2 |
| Better within check | OOF **Top-1** (pick the better of 2) ≈ **0.833** |

**Bottom line:** the run is technically clean, but **`robot_visible=false` changes the task** (humans ignore the robot → collisions explode, SR collapses). Do **not** treat this as a pure “24 vs 72 pool size” ablation versus v4; visibility and pool size are confounded.

---

## 2. Campaign integrity

| Item | Value | Source |
|------|------:|--------|
| candidates | 24 | `candidates_summary.json` |
| families | 12 × 2 | same |
| jobs | 72 (67 new + 5 skipped complete) | `phase1_summary.json` |
| failed | 0 | same |
| outliers | 0 | `outlier_seeds.json` |
| timesteps | 150000 | dataset / status |
| n_eval_episodes | 160 | dataset |
| device | cpu | dataset |
| early checkpoints | 5k / 12k / 25k | `run_manifest.json` |
| phase1 `duration_sec` | 6744.9 ≈ 1.87 h | `phase1_summary.json` |
| mean train+eval / job | ≈ 6.73 min | dataset |
| sum train CPU-h | ≈ 7.49 | dataset |

---

## 3. Label landscape

Seed-mean `success_rate` over 24 candidates:

| Stat | v5 invisible@24 | v4 visible@72 (ref) |
|------|----------------:|--------------------:|
| mean | 0.158 | 0.435 |
| median | 0.168 | 0.472 |
| std | 0.117 | 0.207 |
| min / max | 0.002 / 0.402 | 0.000 / 0.808 |
| mean collision | **0.423** | ≈0.003 |
| mean timeout | 0.419 | 0.562 |
| frac SR < 0.05 | 0.333 | 0.042 |

### Per-family mean SR (v5)

| family | mean | std | min | max |
|--------|-----:|----:|----:|----:|
| progress_heavy | 0.333 | 0.006 | 0.329 | 0.338 |
| progress_dense | 0.293 | 0.155 | 0.183 | 0.402 |
| crowded_progress | 0.240 | 0.044 | 0.208 | 0.271 |
| balanced | 0.205 | 0.058 | 0.165 | 0.246 |
| human_clearance | 0.167 | 0.038 | 0.140 | 0.194 |
| time_pressure | 0.156 | 0.194 | 0.019 | 0.294 |
| safety_moderate | 0.151 | 0.028 | 0.131 | 0.171 |
| efficiency | 0.130 | 0.116 | 0.048 | 0.213 |
| nonlinear_progress | 0.115 | 0.103 | 0.042 | 0.188 |
| risk_sensitive | 0.069 | 0.038 | 0.042 | 0.096 |
| energy_aware | 0.020 | 0.025 | 0.002 | 0.038 |
| weak_contrast | 0.013 | 0.009 | 0.006 | 0.019 |

**Best candidate:** `progress_dense_01` (SR ≈ 0.402).  
**Weakest band:** `energy_aware`, `weak_contrast` (near floor).

**Interpretation of `robot_visible=false`:** ORCA humans do not avoid the robot, so collisions become a first-class failure mode (~42%). Success is rarer; the reward pool still separates progress-heavy families from weak ones, but absolute performance is much lower than v4.

---

## 4. Surrogate results

Source: `results/crowdnav_v5_invisible_24/leaderboard.csv` (mean over 3 repeats).

| Metric | v5 | v4 (ref) | Caution |
|--------|---:|---------:|---------|
| pooled Spearman | 0.698 | 0.653 | OK to compare directionally |
| within Spearman mean | 0.717 | 0.746 | v5: only 2 points/family → often ±1 |
| within Spearman std | **0.689** | 0.234 | v5 unstable across families |
| Top-3 hit | **1.000** | 0.917 | **degenerate** at family size 2 |
| within pairwise | 0.833 | 0.823 | = Top-1 for pairs |
| MAE | 0.062 | 0.124 | different SR scale |
| R² | 0.511 | 0.402 | |
| top10 regret | 0.025 | 0.086 | smaller pool |
| boot within CI | [0.49, 1.00] | [0.65, 0.85] | |

### Repeats

| repeat | pooled | within | Top-3 |
|-------:|-------:|-------:|------:|
| 0 | 0.719 | 0.667 | 1.0 |
| 1 | 0.682 | 0.818 | 1.0 |
| 2 | 0.693 | 0.667 | 1.0 |

### Within by family (mean over repeats)

| family | within Spearman | note |
|--------|----------------:|------|
| 10 families | **+1.0** | perfect order of the 2 members |
| safety_moderate | −0.33 | often wrong order |
| crowded_progress | **−1.0** | systematically inverted |

### OOF Top-1 (meaningful with n=2)

Mean over families×repeats: **0.833**  
Failures concentrated in `crowded_progress` (0.0) and partly `safety_moderate` (0.33).

---

## 5. What this run does / does not show

**Shows**
1. Full pipeline with invisible robot is runnable (~2.2 h wall @ 4 workers for 24×3).
2. ExtraTrees still gets usable **pooled** ranking (≈0.70) under the new dynamics.
3. Collision becomes informative under `robot_visible=false` (unlike v4).

**Does not show**
1. That “24 candidates are enough” in the same sense as the offline size curve on v4 — here visibility also changed.
2. That within-family ranking is as strong as v4’s 0.75 — Top-3=1.0 and many ±1 Spearmans are artifacts of n=2.
3. A fair head-to-head vs v4 for pool size alone.

---

## 6. Recommendations

1. **For thesis claims about CrowdNav triage quality:** keep **v4 (visible, 72)** as the primary operating result.  
2. **For visibility ablation:** either re-run **72 with `robot_visible=false`**, or re-run **24 with `robot_visible=true`**, so one factor moves at a time.  
3. If staying on 24-pool: report **Top-1 / pairwise**, not Top-3; footnote family-size degeneracy.  
4. Optional next metric: train a collision-rate or multi-objective view — CR is no longer ~0.

---

## 7. Artifact index

| Path | Content |
|------|---------|
| `experiments/crowdnav_v5_invisible_24/dataset/` | 72-row performance dataset |
| `experiments/crowdnav_v5_invisible_24/model_comparison/` | LOFO metrics + OOF |
| `results/crowdnav_v5_invisible_24/leaderboard.csv` | Primary metrics |
| `results/crowdnav_v5_invisible_24/model_comparison_report.md` | Auto report |
| `results/crowdnav_v5_invisible_24/figures/` | CI / top-10 / cost plots |
| `results/crowdnav_v5_invisible_24/ANALYSIS_REPORT.md` | This analysis |
