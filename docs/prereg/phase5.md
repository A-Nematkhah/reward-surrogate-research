# Pre-registration — Phase 5 (per-env online surrogate)

**Date:** 2026-11-11  
**Status:** pre-registered before confirmatory Phase-5 runs.

## Target
Absolute `success_rate` (primary). Secondary: generation-centered ranks; pairwise within generation (exploratory if time).

## Feature set (locked)
**`early+math+behavioral`** (user decision). Ablations only: `math+behavioral`, `early`.

## Model (reference)
ExtraTreesRegressor (`n_estimators=250`, `min_samples_leaf=2`, `random_state=0`) with median imputation.  
Uncertainty: residual split-conformal (lineage/family-grouped calibration when groups exist).  
OOD: Mahalanobis distance on imputed train features; flag if above train 99th percentile.

## Selection
Top-k with `k_frac` (default 0.5); OOD always kept; audit quota 15% of remainder; log selection probabilities.

## Cold-start (dev only, 2D 108)
Simulate labels arriving in random and family-blocked order; report within-pool recall@k vs n_labels; choose `m_min` and initial `k_frac`.

## Metrics
Phase-1 triage metrics: recall@k, regret@k, pairwise; bootstrap over families/runs.  
Baselines: random-k, Tier-0-only.

## Confirmatory vs exploratory
Anything tuned after seeing Phase-4 live logs is labeled exploratory. Confirm on held-out runs/prompts when enough Phase-4 repeats exist.
