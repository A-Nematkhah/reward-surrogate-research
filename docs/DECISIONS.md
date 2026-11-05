# Research and implementation decisions

This log records important design choices. Entries are appended as steps complete.

## Step 1 — Foundation (pilot numbers)

| Decision | Rationale |
|----------|-----------|
| Standalone package under `reward-surrogate-research/` | Workspace root may contain unrelated material later; keep the research repo self-contained. |
| YAML configs split by concern (`environment`, `ppo`, `reward_generation`, `evaluation`, `surrogate`) | Avoid hardcoding experiment parameters; allow budget/seed changes without code edits. |
| Pilot PPO budget: 50k timesteps, 2 seeds, 4 envs *(pilot; see Wave 3)* | Affordable for iteration; configs explicitly support scaling up for final results. |
| Reward pool: 32 candidates across 7 families *(pilot; see Wave 3)* | Within the 24–40 target; includes balanced, safety-heavy, sparse, and deliberately weak families for diversity. |
| No scalar fitness in evaluation config | Preserve Pareto-comparable task metrics; return is descriptive only. |
| Grouped family split as primary surrogate validation | Prevent leakage from similar template variants into both train and test. |
| `seed_everything` tolerates missing torch at import | Foundation tests can run before the full RL stack is installed. |
| CLI dotted overrides (`ppo.total_timesteps=…`) | Practical experiment control without proliferating YAML copies. |

## Step 2 — Environment

| Decision | Rationale |
|----------|-----------|
| Kinematic velocity control (`action * max_speed`) with `dt=1` | Stable, dependency-free; sufficient for reward-design research. |
| Circular obstacles only | Simple closed-form collision/clearance; easy to randomize. |
| Observation: pose, velocity, relative target, dist, nearest-3 obstacles | Compact Markovian state without raw maps. |
| Missing obstacle slots padded with far-away sentinel | Fixed observation size when fewer than `n_nearest` obstacles exist. |
| Termination priority: success > collision > OOB; timeout truncates | Distinguishes success from failure; Gymnasium terminated vs truncated. |
| Pluggable `reward_fn` with a simple default | Env usable for demos/tests now; full component system comes in Step 3. |
| Scene placement via `reset(options=...)` | Enables deterministic unit tests for collision/goal/bounds. |

## Step 3 — Reward system

| Decision | Rationale |
|----------|-----------|
| Six fixed components (progress, goal, collision, obstacle, smoothness, time) | Interpretable, no LLM code; comparable across candidates. |
| Components return unweighted O(1)-ish values; weights applied separately | Avoids magnitude domination and keeps weight semantics clear. |
| Obstacle term: `clip(1 - clearance/safe_distance, 0, 1)` | Bounded proximity cost without needing collision events. |
| Smoothness: `‖Δa‖ / √2` | Normalized action-change penalty in [-1,1]^2 action space. |
| Collision component includes out-of-bounds | Both are hard failures; one binary failure indicator. |
| 7 families including weak/near-zero | Dataset must contain bad rewards, not only promising ones. |
| Per-candidate `candidate_seed` stored in metadata | Exact regeneration of an individual weight draw. |
| Env logs `reward_components` / `reward_weighted` in info | Enables episode-level component analysis without future leakage. |

## Step 4 — PPO training

| Decision | Rationale |
|----------|-----------|
| Identical PPO hyperparameters for every candidate | Fair comparison; no per-reward tuning. |
| DummyVecEnv (not Subproc) for pilot | Simpler reward_fn copies; enough for n_envs=4 on CPU. |
| Fresh `WeightedRewardFunction` per env thunk | Avoids shared mutable component-log state. |
| Task metrics separate from return; no scalar fitness | Preserve Pareto trade-offs for later analysis. |
| `mean_goal_time` = mean length over successes only (NaN if none) | Avoids treating failures as zero-time goals. |
| `status.json` records failures explicitly; runner can continue | Do not silently drop failed (candidate, seed) pairs. |
| Resume skips runs with completed status + `model.zip` | Interrupted Phase 1 jobs can continue safely. |
| Pilot verification used 10k steps for one candidate | Full 50k×32×2 budget deferred to Phase 1 dataset step. |

## Step 5 — Phase 1 dataset

| Decision | Rationale |
|----------|-----------|
| Pilot budget 20k timesteps × 2 seeds × 20 eval episodes | Affordable full-pool run (~11 min); config still defaults to 50k for later scale-up. |
| Resume checks matching `total_timesteps` | Prevents smoke runs (e.g. 10k) from counting as final results. |
| Dataset retains failed runs | Failures must be visible, not silently dropped. |
| Seed-averaged candidate views for plots/report | Multi-seed rows kept in CSV; figures avoid double-counting candidates. |
| No scalar fitness / Pareto kept as multi-objective view | Success, collision, and goal time remain separate. |
| Honest report: no “universally best” claim | Rankings are budget/env/seed-specific. |

### Pilot outcome (empirical)

- 64/64 completed; success-rate span **0.80** across candidates.
- Family means: balanced ~0.53, progress_heavy ~0.47, …, efficiency ~0.00 (mostly OOB).
- Mean seed success spread ~0.06 (some candidates up to 0.25) — uncertainty is non-trivial.

## Step 6 — Feature extraction

| Decision | Rationale |
|----------|-----------|
| Static features = weights, norms, ratios, activity counts, family one-hot | Go beyond raw weights without using outcomes. |
| Probes: random / goal-seeking / obstacle-avoiding only | Reward-independent; no candidate PPO trajectories. |
| Fixed probe seed=7, 10 traj/policy, saved NPZ+JSON | Same probe set for every candidate; reproducible. |
| Behavioral stats: mean/std/min/max/quantiles/pct-zero, contrib ratios, correlations, per-policy totals | Matches research plan; no eval metrics as inputs. |
| Seed aggregation = mean before features | One row per candidate for surrogate; matches config. |
| Explicit leakage asserts on feature tables | Fail loudly if target columns appear in X. |

## Step 7 — Surrogate models

| Decision | Rationale |
|----------|-----------|
| Grouped K-fold by `family_id` as primary | Prevents near-duplicate family members leaking into both splits. |
| Random holdout secondary only | Comparison, not the main generalization claim. |
| Classical models only (mean/median/ridge/RF/ET/GP) | Dataset is small (~32 candidates in pilot; ~108 after Wave 3). |
| Light grouped hyperparameter search on train folds only | Avoids test leakage; keeps compute modest. |
| Preprocessing / model fitting only on training fold | Standard anti-leakage discipline. |
| Tree std = ensemble disagreement, not calibrated uncertainty | Documented explicitly; GP std is model-based but still unverified. |
| NaN targets (e.g. goal time with 0 successes) dropped per-target | Avoid fabricating labels. |

### Pilot empirical snapshot (grouped OOF)

- Best success-rate ranking: GP + behavioral (Spearman ≈ 0.31); R² ≈ 0.
- Stronger signal on some other metrics (e.g. path length / goal time) with tree/static features.
- Mean/median baselines are weak under grouped CV — as expected when family means differ.
- Results are mixed: some predictive value vs baselines, but far from reliable exact performance prediction. Step 8 must report this honestly.

## Step 8 — Final analysis

| Decision | Rationale |
|----------|-----------|
| Report ranking (Spearman) alongside R² | Screening use-case cares about ordering more than calibrated MSE. |
| Phase 3 gate: success-rate grouped Spearman ≳ 0.5–0.6 | Avoid AL on a surrogate that cannot triage candidates reliably. |
| Publish negative/mixed results | Research integrity over looking successful. |

## Post-audit fixes

| Decision | Rationale |
|----------|-----------|
| Disable family one-hot in Phase 2 features | Held-out family indicators cannot generalize under family CV. |
| Primary CV = leave-one-family-out | Strongest / clearest unseen-family test (pilot: 7 families; Wave 3: 12). |
| Constant baselines: ranking = N/A; compare MAE | Avoid artifactual pooled Spearmans for mean/median. |
| Report pooled OOF **and** fold Spearman mean±std | Dual estimands; high fold_std ⇒ unreliable. |
| AL gate also requires fold_std ≲ 0.35 | Stability constraint from audit. |

## Scale-up: label noise + early behavioral + larger pool

| Decision | Rationale |
|----------|-----------|
| 4 training seeds + 80 eval episodes | Reduce label noise vs 2×20 pilot. |
| 30k PPO timesteps | Affordable scale-up above 20k pilot without full 50k cost. |
| 64 candidates (doubled family counts) | Denser weight-space coverage for surrogate fitting. |
| Early behavioral after 5k warmup (12 rollouts) | Cheap policy-behavior features before full training finishes; valid AL triage inputs. |
| Feature sets: static / behavioral / early / hybrid | Ablate probe-scoring vs short-PPO signals. |
| Prefix early_* feature names | Avoid collision/confusion with full-budget target metrics. |

## Surrogate improvement waves (post audit)

| Decision | Rationale |
|----------|-----------|
| Top-k (1/3/5) + pairwise ranking metrics | Match practical triage use-case, not only Spearman. |
| HistGradientBoosting + nested inner GroupKFold HP | Strong classical baseline; prevent HP leakage into LOFO test. |
| Math + interaction probe features | Landscape geometry without PPO; keep reward-independent probes. |
| 12 families (7 linear + 5 structural components) | LOFO needs true mathematical diversity, not renamed weight templates. |
| ~108 candidates × 3 seeds × 50k | Moderate scale; avoid 5×100k until signal improves. |
| AL gate also requires Top-3 ≳ 0.5 | Ranking alone was insufficient for selection decisions. |

## Wave 3 scale-up (current primary protocol)

| Decision | Rationale |
|----------|-----------|
| **108 candidates / 12 families** (7 linear + 5 structural) | LOFO needs mathematical diversity, not renamed weight templates. |
| **50k PPO steps × 3 seeds**; 80 eval episodes | Label-noise reduction vs pilot 20k×2 / sparse eval. |
| **Early behavioral** after 5k warmup (12 rollouts) | Cheap policy-behavior features before full training finishes. |
| Primary CV = **leave-one-family-out (LOFO)** | Strongest unseen-family generalization test. |
| Family one-hot features **disabled** | Unseen family IDs cannot generalize under LOFO. |
| Mean/median ranking = **N/A** | Constant predictors are not competitive rankers; compare MAE/RMSE. |
| Feature ablations: static / behavioral / math / interaction / early / hybrid | Isolate probe scoring vs weights vs short-PPO signals. |
| AL gate: Spearman ≳ 0.5–0.6, fold std ≲ 0.35, **fold** Top-3 ≳ 0.5 | Ranking + triage + stability before trusting AL. |

Pilot protocol (32 candidates / 7 families / ~20k steps / GroupKFold or early LOFO) remains documented above for history; Wave 3 supersedes it for primary claims.

## Pre-AL quality push

| Decision | Rationale |
|----------|-----------|
| Primary feature set for triage = **behavioral_early** | Best ExtraTrees LOFO tradeoff after Wave 2 (Spearman≈0.835, unc↔\|err\|≈0.74). |
| Keep `rank_extra_trees` diagnostic-only for AL | Strong pooled ranking but poor uncertainty–error correlation / MAE scale. |
| `interaction` demoted from primary ablations | Near-noise under LOFO; wastes compute. |
| Train-fold feature prune (max 40) | Reduce hybrid noise without test leakage. |
| Dense eval = **160 episodes** (checkpoint re-eval, no PPO retrain) | Cut seed success spread ≈0.075→0.068. |
| Soft AL gate still Conditional yes; strict std/Top-3 targets unmet | Proceed to AL prototype with ExtraTrees/behavioral_early, not as a high-stakes skip-PPO oracle. |

## Model comparison — pre-registered confirmatory baseline

| Decision | Rationale |
|----------|-----------|
| Primary claim set = **CatBoost + ExtraTrees** on `early+math` / `success_rate` | Locked after exploratory `--quick` grid + focused full-grid top-3 (`n_repeats=1`). CatBoost led within-family Spearman / top-3 hit; ExtraTrees led pooled Spearman. |
| Paired baseline = **ExtraTrees** | Stable reference for `paired_vs_baseline`; other models are hypotheses vs this anchor. |
| Confirmatory run = full grids, `n_repeats=3`, outputs under `experiments/model_comparison_confirm` | Definitive CI / stability; does not overwrite exploratory `model_comparison` artifacts. |
| HGB / XGB / rankers stay exploratory-only | Top-3 confirm did not support promoting HGB on the primary within-family metric. |

## Surrogate quality push — richer early + cleaner labels

| Decision | Rationale |
|----------|-----------|
| Early checkpoints **2k / 5k / 12k** + schema v2 | Capture learning trajectory, not a single fragile 5k snapshot. |
| Robust rollout stats (median / IQR / MAD / winsor mean) + cross-ckpt deltas | Reduce noise from mean-only early summaries. |
| Early refresh script trains only to 12k (keeps final `model.zip` / eval) | Upgrade features without full 50k redo. |
| Parallel `--workers` (default 4) on early refresh / reeval | ~3–5× wall-clock with same quality; `n_envs=1` under multi-worker refresh. |
| Keep eval at **160** episodes | 200 is costly for little gain; skip full reeval in the default pipeline. |
| Outlier policy = **filter-by-default** (`--drop-outliers-csv`) | Prefer dropping noisy seeds over full PPO invalidate/retrain; optional outlier-only reeval. |
| New LOFO combos: `early+static`, `math+behavioral`, `early+math+behavioral` | Test whether enriched early still wins after refresh. |

## Operating surrogate (locked after confirmatory run)

| Decision | Rationale |
|----------|-----------|
| Model = **ExtraTrees** only | 3-repeat LOFO: within-family Spearman ≈ 0.65, pooled ≈ 0.94, top-3 hit ≈ 0.83. CatBoost was worse on every repeat. |
| Features = **early + math + behavioral** | Best triage set in the confirm (`early+math` stays a close runner-up). |
| Target = **success_rate** | Selection decision. Safety rates stay in the dataset, not in the default fit. |
| Default command = `python scripts/run_quality_pipeline.py` | Flag outliers, refresh early features, rebuild dataset, extract features, fit ExtraTrees, write the report. |
| New pools use **48 candidates** (4 × 12 families) | Size curve on the 108-pool: within-family Spearman stays ≈ 0.63 vs 0.65; top-3 hit 0.77 vs 0.83. The 108-run archive stays in `experiments/phase1`. |
| **Immutable baseline archive** | Frozen copy at `experiments/archives/model_comparison_confirm_2026-11-05/`. Verified means: pooled Spearman ≈ **0.935**, within-family ≈ **0.648**, top-3 hit **0.833**. Do **not** overwrite `experiments/model_comparison_confirm/` or `results/model_comparison_confirm/` without a new run_id / output dir. Do not cite unverified 0.8448/0.5751 as the 2D confirm (those are closer to CrowdNav v1). |
| **Within-family estimand (post-audit)** | Primary within-family Spearman is reported on **identifiable** families (range vs seed-SE gate); all-12 macro remains secondary. See `research_audit/METRIC_POLICY.md`. |
| **Identifiability metrics in LOFO** | Optional via `within_family_identifiability.enabled` (default **false** on confirm config). Benchmark config enables it and writes to `experiments/benchmark_phase2_base/` only — never into the locked confirm dirs. |
| **Fair benchmark entrypoint** | `python scripts/run_fair_benchmark.py` (+ optional `--quick`). Locked contrast: mean / ridge / extra_trees. |
| **Phase 3.1 feature extras** | `ratios` + `early_deltas` CSVs added; train-fold within-family std prune optional. Quick ablation did **not** beat `early+math+behavioral` on identifiable within-Spearman → keep operating features; do not promote ratios/deltas from quick screen alone. |
| **Phase 3.2 ranking contrast** | Pre-registered `pairwise_logistic` vs ExtraTrees on identifiable within-Spearman. Point estimate favored pairwise (+0.07) but family bootstrap 90% CI included 0 → **keep ExtraTrees**; do not escalate rankers/NNs. |
| **Phase 4 adaptive seeds (plan only)** | Cap 36 extra seed-runs focused on `risk_sensitive` + `nonlinear_progress`. Plan at `results/adaptive_seed_plan/`. **No PPO launched** until explicit authorization; never overwrite phase1 archive. |
| **Phase 5 selection sim** | Offline ET top-k on confirm OOF beats random; finds global best by k=5 (k=3 within ~0.015). Supports triage claim; online AL still deferred. |

## Deferred (by design)

- CrowdNav++ is vendored at `third_party/CrowdNavPlusPlus`. `python scripts/run_quality_pipeline.py --env crowdnav` trains Phase 1 on `CrowdNavPlusPlus-v0` (ORCA humans) into versioned roots `experiments/crowdnav_<run_id>/` / `results/crowdnav_<run_id>/` (`configs/crowdnav_run.yaml`). The first campaign is archived at `experiments/crowdnav_v1_50k/` and `results/crowdnav_v1_50k/`. The locked ExtraTrees numbers stay the 2D-nav confirm; `environment.yaml` still defaults to `RewardNavigation2D-v0`.
- LLM reward generation remains future work.
- Full Active Learning loop is next once this push is accepted.
