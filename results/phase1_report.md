# Phase 1 Report: Reward–Performance Dataset

## Summary

- Total rows: **324**; completed: **324**; failed: **0**.
- Candidates with completed runs: **108** across **12** families.

## Which candidates performed well / poorly?

Rankings below are **specific to this environment, PPO budget, and seeds**. They are not universal claims about reward design.

### Highest seed-averaged success rate

| candidate_id | family_id | success | collision | timeout | mean_goal_time |
|---|---|---:|---:|---:|---:|
| energy_aware_02 | energy_aware | 0.842 | 0.156 | 0.002 | 5.08 |
| efficiency_00 | efficiency | 0.838 | 0.140 | 0.023 | 4.65 |
| energy_aware_05 | energy_aware | 0.833 | 0.148 | 0.019 | 5.05 |
| efficiency_06 | efficiency | 0.829 | 0.148 | 0.023 | 4.61 |
| balanced_03 | balanced | 0.827 | 0.156 | 0.017 | 4.60 |

### Lowest seed-averaged success rate

| candidate_id | family_id | success | collision | timeout |
|---|---|---:|---:|---:|
| distance_dense_03 | distance_dense | 0.000 | 0.027 | 0.000 |
| near_zero_04 | near_zero | 0.000 | 0.065 | 0.000 |
| near_zero_03 | near_zero | 0.000 | 0.083 | 0.000 |
| weak_unbalanced_01 | weak_unbalanced | 0.000 | 0.088 | 0.000 |
| efficiency_07 | efficiency | 0.002 | 0.025 | 0.000 |

## Did different rewards produce different behaviors?

- Success rate range: [0.000, 0.842] (span=0.842, std=0.275).
- Collision rate range: [0.004, 0.335] (span=0.331).

### Family-level success (seed-averaged candidates)

| family_id | n | mean success | std |
|---|---:|---:|---:|
| balanced | 10 | 0.635 | 0.148 |
| progress_heavy | 8 | 0.596 | 0.080 |
| energy_aware | 10 | 0.571 | 0.350 |
| risk_sensitive | 10 | 0.429 | 0.072 |
| nonlinear_progress | 10 | 0.428 | 0.068 |
| efficiency | 8 | 0.393 | 0.410 |
| safety_heavy | 8 | 0.390 | 0.180 |
| sparse_goal | 8 | 0.352 | 0.038 |
| potential_shaping | 10 | 0.092 | 0.051 |
| near_zero | 8 | 0.043 | 0.037 |
| weak_unbalanced | 8 | 0.040 | 0.028 |
| distance_dense | 10 | 0.004 | 0.005 |

## Stability across training seeds

- Candidates with ≥2 seeds: **108**; mean success-rate seed spread: **0.068**.
- Largest seed spreads (uncertainty caution):

| candidate_id | seed spread | mean success |
|---|---:|---:|
| efficiency_03 | 0.494 | 0.633 |
| energy_aware_01 | 0.394 | 0.667 |
| balanced_00 | 0.231 | 0.629 |
| balanced_04 | 0.213 | 0.700 |
| nonlinear_progress_09 | 0.206 | 0.510 |

## Training failures

- No failed runs recorded.

## Is the dataset diverse enough for a surrogate?

- **Yes, provisionally:** multiple performance levels exist. Grouped family splits remain necessary to avoid optimistic leakage.

## Limitations

- Toy 2D navigation only; no claim about CrowdNav or multi-agent settings.
- Pilot PPO budgets may under-train some reward designs.
- Evaluation return is descriptive and must not replace task metrics.
