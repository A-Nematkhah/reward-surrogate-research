# Phase 5 report — Per-env surrogate + filter (MVP)

**Status: MVP complete** (offline train + selector + active wire). Full Phase-4-log evaluation / IPW deferred until more live labels exist.

## Locked choices

- Feature set: **`early+math+behavioral`**
- Model: ExtraTrees + residual conformal + Mahalanobis OOD
- Pre-reg: `docs/prereg/phase5.md`

## Delivered

| Piece | Path |
| --- | --- |
| Surrogate | `reward_triage/model.py` |
| Selector | `reward_triage/select.py` |
| Feature matrix helpers | `reward_triage/feature_matrix.py` |
| Train + cold-start | `scripts/train_phase5_surrogate.py` |
| Artifact | `experiments/phase5_surrogate_2d/` |
| Active wire | `mini_host_loop.py --mode active --surrogate …` |

## Numbers (2D 108)

| Metric | Value |
| --- | ---: |
| LOFO within-family Spearman | **0.641** |
| Cold-start recall@5 (n_train=12) | 0.87 |
| Cold-start recall@5 (n_train=24) | 0.925 |
| Suggested `m_min` | **12** |
| Suggested `k_frac` | 0.5 |

## How to use

```powershell
# Train / refresh artifact
python scripts/train_phase5_surrogate.py --cold-start

# Active filter after warm-up (dry)
python scripts/mini_host_loop.py --provider local --dry-run --mode active `
  --surrogate experiments/phase5_surrogate_2d --m-min 12 --max-jobs 0 --run-id phase5_filter_dry

# Live: warm up with --mode off --real-ppo until ≥ m_min labels, then active + surrogate
```

## Honesty / gaps

- Active scoring on brand-new candidates often lacks early_* until an early PPO pass; missing early columns are median-imputed by the pipeline (same caveat as the Phase-4 probe). A dedicated early-then-select pass can follow.
- **Auto-adapt:** after each generation with new labels, surrogate is refit on phase1_108 prior + this run’s labels and saved under `experiments/<run_id>/surrogate/gen_<g>/`. Disable with `--no-adapt`; skip prior with `--no-adapt-prior`.
- Phase-4 live-log LOFO / Gate-2 replay still need more labeled runs.
