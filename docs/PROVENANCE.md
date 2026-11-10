# Provenance map (headline numbers)

Maps README / report headline numbers to locked artifacts. Confirm run recorded
`git_commit` **2fb024c** in `experiments/model_comparison_confirm/run_config.json`
(not current HEAD). Locked trees must not be rewritten.

| Claim | Value | Source | run_id / path | git commit |
| --- | --- | --- | --- | --- |
| Phase 1 pool size | 108 rewards × 3 seeds, 12 families | `experiments/phase1/dataset/reward_performance_dataset.csv` (322 seed rows → 108 candidates) | phase1 | (dataset local; labels predate confirm) |
| Pre-registered primary (ExtraTrees, `early+math`, success_rate) pooled Spearman | **0.933** | `results/model_comparison_confirm/model_comparison_report.md` § Pre-registered; mean over repeats in `experiments/model_comparison_confirm/metrics_summary.csv` | `model_comparison_confirm` | 2fb024c |
| Pre-registered within-family Spearman | **0.616** (std **0.453**) | same report / `metrics_summary.csv` | `model_comparison_confirm` | 2fb024c |
| Pre-registered within-family top-3 hit | **0.722** | same | `model_comparison_confirm` | 2fb024c |
| Operating feature set (ExtraTrees, `early+math+behavioral`) pooled Spearman | **0.935** | report § Exploratory top rows; `metrics_summary.csv` | `model_comparison_confirm` | 2fb024c |
| Operating within-family Spearman | **0.648** | same | `model_comparison_confirm` | 2fb024c |
| Operating within-family top-3 hit | **0.833** | same | `model_comparison_confirm` | 2fb024c |
| Early feature PPO cost | 12,000 steps / candidate | `configs/model_comparison.yaml` → `ppo_cost_steps`; confirm `run_config.json` | config + confirm | 2fb024c |
| OOF coverage | 4 model × feature_set combos only (catboost/extra_trees × early+math / early+math+behavioral) | `experiments/model_comparison_confirm/oof_predictions.csv` | `model_comparison_confirm` | 2fb024c |

## Pre-registered vs operating

- **Pre-registered** before looking at the confirm grid: ExtraTrees (and CatBoost) on **`early+math`** → success_rate.
- **Operating** choice **`early+math+behavioral`** (0.935 / 0.648 / 0.833) was selected **after** seeing the exploratory full grid. Treat it as post-hoc; always report the pre-registered row alongside it.
- Zero-PPO-cost sets (`math`, `behavioral`, `math+behavioral`) have **no** rows in the locked OOF / metrics artifacts.

## Recompute

```bash
python scripts/reproduce_confirm_table.py
```

Expect ExtraTrees `early+math+behavioral` pooled Spearman ≈ 0.935 and within-family ≈ 0.648 (tolerance 1e-3 vs `metrics_summary.csv` means).
