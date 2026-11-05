# Model comparison report

## Research question

Which model family best predicts PPO navigation performance of an **unseen** reward function from reward features, under leave-one-family-out (LOFO)?

## Protocol

Outer evaluation is LOFO over 12 reward families. Hyperparameters and preprocessing (imputation, feature selection, target transforms) are tuned with **inner LOFO on training families only**. Primary metrics emphasize **within-family Spearman** and top-k selection quality, not only pooled Spearman. Repeats=3; bootstrap resamples families (n=2000). Phase 1/2 artifacts were not modified.

## Pre-registered primary results

Pre-registered before looking at results: target=`success_rate`, feature_set=`early+math`, models=['catboost', 'extra_trees'].

| model | spearman | within_family_spearman_mean | within_family_spearman_std | within_family_top3_hit | top10_selected_mean | top10_regret | mae | needs_ppo_steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| extra_trees | 0.9334 | 0.6156 | 0.4527 | 0.7222 | 0.7959 | 0.02333 | 0.06643 | 1.2e+04 |
| catboost | 0.9052 | 0.4944 | 0.4953 | 0.7222 | 0.7557 | 0.06361 | 0.08947 | 1.2e+04 |

This primary table is **not** cherry-picked from the full grid. Skipped primary models (optional deps) are listed in `run_config.json`.

## Exploratory full grid

The tables/figures below explore many models × feature sets. Headline claims must not be the max over this grid without correction.

### Selection-bias guard

| model | median_spearman_across_feature_sets | frac_spearman_gt_0_5 | n_configs |
| --- | --- | --- | --- |
| extra_trees | 0.9378 | 1 | 6 |
| catboost | 0.9161 | 1 | 6 |

### Top exploratory rows by within-family Spearman (success_rate)

| model | feature_set | spearman | within_family_spearman_mean | fold_spearman_std | within_family_top3_hit | needs_ppo_steps |
| --- | --- | --- | --- | --- | --- | --- |
| extra_trees | early+math+behavioral | 0.9352 | 0.6481 | 0.4283 | 0.8333 | 1.2e+04 |
| extra_trees | early+math | 0.9334 | 0.6156 | 0.4527 | 0.7222 | 1.2e+04 |
| catboost | early+math+behavioral | 0.9169 | 0.5429 | 0.4662 | 0.7222 | 1.2e+04 |
| catboost | early+math | 0.9052 | 0.4944 | 0.4953 | 0.7222 | 1.2e+04 |

## Within-family and top-k

Within-family ranking is the practical use-case. See `figures/within_family_spearman_ci.png` and `figures/top10_selected_vs_oracle.png`.

## Accuracy vs PPO cost

Feature sets containing `early` cost ~5000 PPO steps per candidate; static/math/behavioral cost 0. See `figures/accuracy_vs_ppo_cost.png`.

## Neural nets vs trees (honest)

Neural / TabPFN models were skipped or produced no rows.

## Paired comparison vs ExtraTrees

Mean per-family Spearman difference vs ExtraTrees on the same feature set (bootstrap CI over families; **low power with 12 families**).

| repeat | feature_set | target | model | baseline | mean_spearman_diff | diff_ci_lo | diff_ci_hi | n_families | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | early+math | success_rate | catboost | extra_trees | -0.137 | -0.3597 | 0.002077 | 12 | n_families=12 has low power; CI descriptive only |
| 1 | early+math | success_rate | catboost | extra_trees | -0.14 | -0.3118 | -0.02465 | 12 | n_families=12 has low power; CI descriptive only |
| 2 | early+math | success_rate | catboost | extra_trees | -0.08692 | -0.1504 | -0.0218 | 12 | n_families=12 has low power; CI descriptive only |
| 0 | early+math+behavioral | success_rate | catboost | extra_trees | -0.1237 | -0.3442 | 0.01801 | 12 | n_families=12 has low power; CI descriptive only |
| 1 | early+math+behavioral | success_rate | catboost | extra_trees | -0.06334 | -0.1224 | -0.0151 | 12 | n_families=12 has low power; CI descriptive only |
| 2 | early+math+behavioral | success_rate | catboost | extra_trees | -0.1286 | -0.3154 | -0.006825 | 12 | n_families=12 has low power; CI descriptive only |

## Limitations

- n=108 candidates, 12 families; within-family signal is weak historically.
- Toy 2D navigation only; not CrowdNav.
- Optional packages may skip models; see run_config skipped list.
- Exploratory max over many configs is selection-biased.

## Recommendation

Under the **pre-registered** setting, best available row is `extra_trees` (within-family Spearman≈0.616, pooled≈0.933). Use this as the decision baseline; treat exploratory winners as hypotheses only.

## Artifacts

- Experiments: `I:\Code\New folder (2)\reward-surrogate-research\experiments\model_comparison_confirm`
- Leaderboard: `I:\Code\New folder (2)\reward-surrogate-research\results\model_comparison_confirm\leaderboard.csv`
- Figures: `I:\Code\New folder (2)\reward-surrogate-research\results\model_comparison_confirm\figures`
