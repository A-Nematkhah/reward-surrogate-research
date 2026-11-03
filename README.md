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

Full resumable pipeline (candidates → Phase 1 → dataset → features → Phase 2 → reports):

```bash
python scripts/run_full_improvement_pipeline.py
```

Step-by-step (same stages):

```bash
python scripts/run_environment_demo.py
python scripts/generate_candidates.py
python scripts/run_phase1.py
python scripts/build_dataset.py
python scripts/extract_features.py
# optional: --regenerate-probes ; --override surrogate.include_early_behavioral=false
python scripts/run_phase2.py
python scripts/generate_report.py --phase all
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

## Key results (Wave 3 scale-up)

- Phase 1: **108** rewards × **3** seeds (**324** runs) across **12** families; success span ≈ **0.86**.
- Phase 2 LOFO: best success ranking ≈ **Spearman 0.815** (ExtraTrees / behavioral); fold Top-3 ≈ **0.67**.
- AL gate: **conditional yes** for triage prototyping (see `results/final_research_report.md`).
- Pilot (32×2 / 7 families) audit archived under `docs/archive/pilot_audit/`.

## Reports

- `results/phase1_report.md`
- `results/phase2_report.md`
- `results/final_research_report.md`

## License

Research prototype — no license asserted yet.
