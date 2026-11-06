# Dataset-size curve — CrowdNav v4@150k

**Setup:** ExtraTrees / `early+math+behavioral` / LOFO  
**Protocol:** test family intact; train side stratified-subsampled to `train_n`  
**Repeats:** 5 random subsamples per size  
**CSV:** [`dataset_size_curve.csv`](dataset_size_curve.csv)

Full LOFO train max = **66** (72 − 6).

## Results

| train_n | ≈ fraction of max | within Spearman | Top-3 hit | pairwise | pooled Spearman | MAE |
|--------:|------------------:|----------------:|----------:|---------:|----------------:|----:|
| 12 | 18% | 0.594 ± 0.06 | 0.75 | 0.75 | 0.50 | — |
| 18 | 27% | 0.693 ± 0.04 | 0.90 | 0.79 | 0.63 | — |
| 24 | 36% | 0.734 ± 0.06 | 0.93 | 0.82 | 0.61 | — |
| 33 | 50% | 0.693 ± 0.03 | 0.87 | 0.79 | 0.62 | — |
| 44 | 67% | 0.745 ± 0.02 | 0.93 | 0.82 | 0.66 | — |
| 55 | 83% | 0.756 ± 0.02 | 0.95 | 0.83 | 0.66 | — |
| 66 | 100% | **0.778 ± 0.01** | **0.92** | **0.85** | **0.68** | — |

(Std from 5 subsample repeats; see CSV for MAE columns.)

## Where accuracy stops falling much

| Region | Verdict |
|--------|---------|
| **train_n ≤ 12** | Too thin — within ~0.59, Top-3 ~0.75 |
| **≈ 18** | Big jump; Top-3 already ~0.90 |
| **≈ 24–44** | **Plateau band** — within ~0.73–0.75, Top-3 ~0.87–0.93; close to full |
| **55–66** | Small extra gain (+0.02–0.04 within); variance shrinks |

Δ(full − 24) ≈ **+0.04** within Spearman, Top-3 essentially flat (~0.93 vs 0.92).  
Δ(full − 44) ≈ **+0.03** within.

## Practical recommendation

On this CrowdNav setup you can cut labeled **train** size to about **⅓–⅔ of full** without a large ranking drop:

- **Aggressive / cheap:** aim ~**24** train labels/fold (~⅓) — expect within ≈ 0.73, Top-3 ≈ 0.93.  
- **Safe:** aim ~**44** (~⅔) — within ≈ 0.75, nearly full quality, lower variance.  
- **Do not** go below ~**18** if Top-3 triage matters.

### Mapping to pool design (rough)

With 12 families × 6 candidates, “train_n ≈ 44” ≈ keeping ~**4 / family** on the train side (11×4).  
A redesigned pool of **~48 candidates** (12×4) is the natural next campaign size if PPO cost is the bottleneck — expect mild within drop vs 72, not collapse.

This curve **shrinks train labels under fixed 72-pool LOFO**; it does **not** re-run PPO. A true smaller pool also changes which candidates exist; treat 48 as a hypothesis for the next generation, not a free lunch from this offline plot alone.

## Contrast with 2D curve

2D (`results/dataset_size_curve.csv`) kept climbing slowly to n=99. CrowdNav saturates earlier: most of the ranking signal is already present by ~24–44 train points.
