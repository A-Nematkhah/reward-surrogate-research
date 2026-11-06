# Model comparison report

## Research question

Which model family best predicts PPO navigation performance of an **unseen** reward function from reward features, under leave-one-family-out (LOFO)?

## Protocol

Outer evaluation is LOFO over 12 reward families. Hyperparameters and preprocessing (imputation, feature selection, target transforms) are tuned with **inner LOFO on training families only**. Primary metrics emphasize **within-family Spearman** and top-k selection quality, not only pooled Spearman. Repeats=3; bootstrap resamples families (n=2000). Phase 1/2 artifacts were not modified.

## Pre-registered primary results

Pre-registered before looking at results: target=`success_rate`, feature_set=`early+math+behavioral`, models=['extra_trees'].

| model | spearman | within_family_spearman_mean | within_family_spearman_std | within_family_top3_hit | top10_selected_mean | top10_regret | mae | needs_ppo_steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| extra_trees | 0.6535 | 0.7461 | 0.2338 | 0.9167 | 0.6407 | 0.08576 | 0.1238 | 1.2e+04 |

This primary table is **not** cherry-picked from the full grid. Skipped primary models (optional deps) are listed in `run_config.json`.

## Exploratory full grid

The tables/figures below explore many models × feature sets. Headline claims must not be the max over this grid without correction.

### Selection-bias guard

| model | median_spearman_across_feature_sets | frac_spearman_gt_0_5 | n_configs |
| --- | --- | --- | --- |
| extra_trees | 0.6524 | 1 | 3 |

### Top exploratory rows by within-family Spearman (success_rate)

| model | feature_set | spearman | within_family_spearman_mean | fold_spearman_std | within_family_top3_hit | needs_ppo_steps |
| --- | --- | --- | --- | --- | --- | --- |
| extra_trees | early+math+behavioral | 0.6535 | 0.7461 | 0.2338 | 0.9167 | 1.2e+04 |

## Within-family and top-k

Within-family ranking is the practical use-case. See `figures/within_family_spearman_ci.png` and `figures/top10_selected_vs_oracle.png`.

## Accuracy vs PPO cost

Feature sets containing `early` cost ~5000 PPO steps per candidate; static/math/behavioral cost 0. See `figures/accuracy_vs_ppo_cost.png`.

## Neural nets vs trees (honest)

Neural / TabPFN models were skipped or produced no rows.

## Paired comparison vs ExtraTrees

_No paired table._

## Limitations

- n=72 candidates, 12 families; within-family signal is weak historically.
- CrowdNav++ (ORCA humans); separate from the locked 2D pool.
- Optional packages may skip models; see run_config skipped list.
- Exploratory max over many configs is selection-biased.

## Recommendation

Under the **pre-registered** setting, best available row is `extra_trees` (within-family Spearman≈0.746, pooled≈0.653). Use this as the decision baseline; treat exploratory winners as hypotheses only.

## Artifacts

- Experiments: `D:\Thesis\Implementation\reward-surrogate-research\experiments\crowdnav_v4_full_150k\model_comparison`
- Leaderboard: `D:\Thesis\Implementation\reward-surrogate-research\results\crowdnav_v4_full_150k\leaderboard.csv`
- Figures: `D:\Thesis\Implementation\reward-surrogate-research\results\crowdnav_v4_full_150k\figures`
