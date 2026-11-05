# Literature Review (Targeted)

**Rule:** Separate verified paper content from our hypotheses. Bibliographic details checked via OpenReview/arXiv/publisher pages (2026-11-05). Full PDF of EWRL 2025 poster was blocked (403); abstract-level verification used.

---

## 1. Performance Prediction in Reinforcement Learning: The Bad and the Ugly

- **Venue:** EWRL 2025 poster (OpenReview `L9J6Xmta4J`)  
- **Authors:** Julian Dierkes, Theresa Eimer, Marius Lindauer, Holger Hoos  
- **Problem:** Surrogates predicting RL **hyperparameter** performance.  
- **Finding (verified from abstract):** RL HP landscapes are rugged/noisy/multimodal; surrogates perform poorly; even insight-only use can deviate from ground truth; caution against blind trust.  
- **Adapt:** Treat noisy landscapes and low trust as default; emphasize uncertainty and selection regret, not point MAE.  
- **Cannot transfer:** Their input is algorithm HPs, **not reward-function features**. Different object of prediction.

---

## 2. Effective Reinforcement Learning through Evolutionary Surrogate-Assisted Prescription (ESP)

- **Venue:** GECCO 2020; arXiv:2002.05368  
- **Authors:** Francon, Gonzalez, Hodjat, Meyerson, Miikkulainen, Qiu, Shahrzad  
- **Problem:** Learn a Predictor from decision→outcome data; evolve a Prescriptor against the surrogate. Extended to sequential/RL tasks.  
- **Input/target:** Context/decision features → predicted outcomes; Prescriptor optimized on surrogate.  
- **Findings:** Sample efficiency, lower variance/regret vs PPO/DQN in reported domains; surrogate regularization.  
- **Adapt:** Surrogate-assisted **selection/optimization** loop (Objective B); evaluate regret under evaluation budget.  
- **Cannot transfer:** They optimize policies/decisions with a surrogate world model; we predict **reward quality for PPO**, not replace the env dynamics.

---

## 3. A Surrogate-Assisted Controller for Expensive Evolutionary Reinforcement Learning

- **Venue:** Information Sciences 2022; arXiv:2201.00129  
- **Authors:** Wang, Zhang, Chang, Wang, Liang, Yuan  
- **Problem:** Reduce expensive fitness evals in evolutionary RL hybrids.  
- **Method:** Critic-based surrogate + generation/individual control + elite protection.  
- **Scale:** MuJoCo continuous control.  
- **Adapt:** Model-management ideas (when to trust surrogate vs real eval); elite protection ≈ don’t discard true top candidates under surrogate error.  
- **Cannot transfer:** Surrogate is over **policies**, using RL critic/replay — not over **reward specifications**.

---

## 4. Surrogate-Assisted Evolutionary RL Based on Autoencoder and Hyperbolic Neural Network

- **Venue:** arXiv:2505.19423 (2025)  
- **Authors:** Li, Jiang, Qian, Tang, Zhou, Yang  
- **Problem:** Surrogate for high-dimensional DNN policies in ERL.  
- **Method:** AE embedding + HNN classification surrogate for pre-selection; Atari/MuJoCo.  
- **Adapt:** Classification/ranking surrogates for pre-filtering; embedding when inputs are ultra-high-dim.  
- **Cannot transfer:** Input is policy weights (millions of dims). Our tabular reward features (~10²) do **not** justify AE-HNN complexity.

---

## 5. The Perils of Trial-and-Error Reward Design (AAAI 2023)

- **Authors:** Booth, Knox, Shah, Niekum, Stone, Allievi  
- **Problem:** Ad hoc dense reward design; overfitting to algorithm/HPs; invalid task specs from myopic state-action scoring.  
- **Findings:** Reward rankings uncorrelated across algorithm/HP changes; experts overfit; many rewards fail to encode the intended task. Recommend separating learning reward from evaluation metric.  
- **Adapt (high relevance):** This repo already separates task metrics from training return — keep that. Surrogate predicting **task metric** under fixed PPO is a response to expensive trial-and-error. Also: claims must fix algorithm/HPs (as this repo does) and not over-claim cross-algorithm generality.  
- **Cannot transfer:** They study human design process / overfitting; not surrogate LOFO prediction.

---

## 6. Adjacent work (brief)

| Paper | Relevance |
|-------|-----------|
| Mateos-Melero et al., EWRL 2024 — image dataset reps for offline RL performance prediction | Different input (datasets), same theme: predict learning performance; mind distribution shift |
| Reward model evaluation / ranking losses (recent RM literature) | Motivates ranking metrics over point accuracy for ordinal outcomes |
| Classical LTR (LambdaMART, RankNet) | Methods candidates; not RL-specific |

---

## Comparison to this repository

| Dimension | This repo | Typical SA-ERL / ESP | HP performance prediction |
|-----------|-----------|----------------------|---------------------------|
| Input | Reward features | Policy / decisions | Hyperparameters |
| Target | Downstream PPO task metric | Episode return / fitness | Config performance |
| Validation | LOFO families | Online learning curves | Surrogate fitness vs true |
| n | ~10² rewards | Many env steps | Config grids |

**Plausible contribution (hypothesis, not claimed novelty without fuller survey):**  
Family-shift evaluation for **reward-function** performance prediction with explicit within-family ranking and budgeted selection — under-explored relative to policy surrogates and HP surrogates.

**Do not claim:** “first surrogate for RL” (false); “solves reward design” (false); transfer to CrowdNav without evidence.

---

## Transferable design lessons

1. Distrust point surrogates on noisy RL landscapes (EWRL 2025).  
2. Optimize/evaluate **regret under eval budget** (ESP / SC).  
3. Protect elites when surrogate errs (SC).  
4. Keep evaluation metric separate from training reward (Booth et al.).  
5. Prefer simple models at small n; complexity needs justification (AE-HNN not transferable).
