# Reward Surrogate Research

Standalone research project investigating:

> **Can we predict the performance of an RL agent trained with a particular reward function, without actually training the agent with that reward function?**

This repository implements **Phase 1** (environment + reward–performance dataset) and **Phase 2** (surrogate models under **leave-one-family-out** CV). CrowdNav and full Active Learning are **out of scope** unless the AL gate in the final report is met.

## Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project foundation (structure, configs, utils) | Done |
| 2 | 2D navigation environment | Done |
| 3 | Reward system and candidates | Done |
| 4 | PPO training pipeline | Done |
| 5 | Phase 1 dataset and report | Done |
| 6 | Feature extraction | Done |
| 7 | Surrogate models | Done |
| 8 | Final analysis and report | **Done** |

## Stack

- **Python 3.10 or 3.12** (both used in this project; prefer one venv)
- Gymnasium, Stable-Baselines3 (PPO), PyTorch
- NumPy, SciPy, Pandas, scikit-learn, Matplotlib
- PyYAML, Pytest

## Setup

```bash
cd reward-surrogate-research
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Intended workflow

Operating surrogate is **ExtraTrees** on `early+math+behavioral` → `success_rate`.

CrowdNav++ (ICRA 2023) is vendored at `third_party/CrowdNavPlusPlus`. A short simulator run that does not need Python-RVO2:

```bash
python scripts/run_crowdnav.py --train --timesteps 256 --humans 5
```

That command builds `CrowdNavPlusPlus-v0`: CrowdNav++ dynamics, ORCA humans, and this project's weighted reward inside the existing PPO trainer.

```bash
python scripts/run_quality_pipeline.py
```

That flags outlier seeds, refreshes early features, rebuilds the dataset, extracts features, fits ExtraTrees under LOFO, and writes `results/model_comparison_confirm/`.

The same surrogate steps on CrowdNav++ (separate dirs; the 2D dataset stays put):

```bash
python scripts/run_quality_pipeline.py --env crowdnav
```

Phase 1 on CrowdNav is the long step (ORCA humans). Operating setup:

- **Pool:** 72 CrowdNav-aware candidates (`configs/reward_generation_crowdnav.yaml`) with human proximity / intrusion / crowded-progress terms and denser progress / time-pressure / clearance families
- **PPO budget:** 150k timesteps (`configs/ppo_crowdnav.yaml`; early checkpoints 5k/12k/25k)
- **Versioned roots:** `configs/crowdnav_run.yaml` → `experiments/crowdnav_<run_id>/` and `results/crowdnav_<run_id>/` (bump `run_id` or pass `--run-id` so campaigns never collide)
- Archive of the first CrowdNav campaign: `experiments/crowdnav_v1_50k/`, `results/crowdnav_v1_50k/`

Short pilot: `--env crowdnav --max-jobs 1 --min-completed 1`.

Regenerate the 2D Phase 1 pool only when labels themselves must be rebuilt:

```bash
python scripts/generate_candidates.py
python scripts/run_phase1.py
```

CrowdNav (versioned; does not overwrite 2D or older CrowdNav trees):

```bash
python scripts/run_quality_pipeline.py --env crowdnav
# or a fresh id:
python scripts/run_quality_pipeline.py --env crowdnav --run-id v3_my_trial
```

Primary Phase 2 protocol: **LOFO** (leave-one-family-out). Random holdout is secondary only.  
Pilot one-hot features are disabled. Mean/median ranking metrics are N/A.

Configuration lives under `configs/`. Override with dotted keys where supported:

```bash
python scripts/run_phase1.py --override ppo.total_timesteps=100000 --override ppo.n_seeds=5
```

**Warning:** `run_phase1.py` / `run_phase2.py` / `run_full_improvement_pipeline.py` retrain or overwrite experiment artifacts. Prefer `generate_report.py` when only reports need refreshing.

## Project layout

```text
reward-surrogate-research/
├── configs/           # YAML experiment configuration
├── src/               # Library code (environment, rewards, training, surrogate, …)
├── scripts/           # CLI entry points
├── tests/             # Pytest suite
├── experiments/       # Run artifacts (candidates, checkpoints, metrics)
├── results/           # Figures and reports
└── docs/              # Decisions + archived pilot audit
```

## Research integrity notes

- Task metrics (success, collision, timeout, …) are kept separate from training return.
- No arbitrary scalar fitness that hides safety/efficiency trade-offs.
- Surrogate features must not leak post-training evaluation outcomes.
- Behavioral probes use fixed, reward-independent trajectories only.
- Negative or null surrogate results will be reported honestly.

## Key results

- Phase 1: **108** rewards × **3** seeds across **12** families.
- Operating LOFO surrogate: **ExtraTrees / early+math+behavioral / success_rate** — within-family Spearman ≈ **0.65**, pooled Spearman ≈ **0.94**, within-family top-3 hit ≈ **0.83**.
- Reliable for within-family triage. Not a replacement for full PPO.

## Reports

- `results/model_comparison_confirm/model_comparison_report.md`

## License

Research prototype — no license asserted yet.