# Repository Architecture Audit

## 1. Top-level layout

| Path | Role |
|------|------|
| `rsr/` | Library: environment, rewards, training, evaluation, dataset, features, surrogate, utils |
| `configs/` | YAML by concern (env, PPO, rewards, eval, surrogate, model_comparison, CrowdNav) |
| `scripts/` | CLI entry points |
| `experiments/` | Heavy artifacts (candidates, runs, datasets, features, LOFO outputs) |
| `results/` | Human-facing reports/figures |
| `docs/` | DECISIONS, reports |
| `tests/` | Pytest (leakage, LOFO, metrics, seeds) |
| `third_party/CrowdNavPlusPlus/` | Vendored secondary env |
| `research_audit/` | This audit (write-only; no pipeline mutation) |

---

## 2. End-to-end data flow (2D primary)

```
Reward generation (families × weights)
    → candidates.json / per-candidate specs
PPO training (50k × 3 seeds)
    → runs/<cand>/seed_*/model.zip, status.json, early_behavioral_features.json
Checkpoint evaluation (160 eps, seeds 10000+)
    → evaluation_metrics / status metrics
Dataset build (+ optional outlier filter)
    → reward_performance_dataset.csv (seed-level rows)
Feature extraction
    → static / math / behavioral / interaction / early / combos
    → probe_trajectories.npz
Surrogate / model comparison (outer LOFO)
    → OOF predictions, metrics_summary, bootstrap CIs
Reporting
    → results/model_comparison_confirm/...
Active learning / selection
    → NOT IMPLEMENTED (gate text only)
```

### Stage details

#### Reward generation
- **Code:** `scripts/generate_candidates.py`, `rsr/rewards/candidate_generator.py`
- **Config:** `configs/reward_generation.yaml` (currently **48** intended; archive is **108**)
- **Outputs:** `experiments/phase1/candidates.json`
- **Assumptions:** Family ranges create LOFO-relevant diversity
- **Leakage:** Low (weights only)
- **Cost:** Negligible
- **Risk:** Regenerating overwrites the locked 108 archive

#### PPO training
- **Code:** `scripts/run_phase1.py`, `rsr/training/ppo_trainer.py`, `experiment_runner.py`
- **Config:** `configs/ppo.yaml`, `environment.yaml`
- **Outputs:** `experiments/phase1/runs/...`
- **Cost:** Dominant (108×3×50k)
- **Assumption:** Shared PPO HPs across rewards (reward is the experimental factor)

#### Evaluation
- **Code:** `rsr/evaluation/evaluator.py`, `metrics.py`
- **Config:** `evaluation.yaml` — 160 episodes, deterministic
- **Outputs:** metrics in status / evaluation_metrics.json
- **Assumption:** Eval seeds independent of train seeds

#### Dataset
- **Code:** `rsr/dataset/builder.py`, `validator.py`, `outlier_filter.py`
- **Outputs:** `experiments/phase1/dataset/`
- **Verified:** 324 completed seeds historically; current CSV may show 322 after outlier drop (2 candidates with 2 seeds)

#### Features
| Set | Approx. cols | Cost | Notes |
|-----|-------------:|------|-------|
| static | 60 | 0 | Weights / structure |
| math | 17 | low | Probe scoring |
| behavioral | 197 | low–med | Fixed probes; **low within-family variation** |
| interaction | 50 | low | Demoted (near-noise under LOFO) |
| early | 197 | ~12k PPO steps/cand | Multi-ckpt warmup stats |
| hybrid / combos | — | — | Built at compare time for `early+math+behavioral` |

Leakage assert: `_assert_no_target_leakage` in feature builder.

#### Surrogate / confirm
- **Phase 2:** `scripts/run_phase2.py` + `configs/surrogate.yaml` (now ExtraTrees-focused)
- **Confirm:** `scripts/run_model_comparison.py` → `rsr/surrogate/model_comparison.py`
- **Locked artifacts:** `experiments/model_comparison_confirm/`, `results/model_comparison_confirm/`
- **Missing:** `experiments/model_comparison/` (exploratory grid), `results/phase2_report.md`

#### Active learning
- **Status:** Absent as code. `phase2_analysis` mentions AL gates only.

---

## 3. Dependency map (logical)

```
configs/* ──► scripts/* ──► rsr/*
                │
                ▼
         experiments/phase1 (labels)
                │
                ▼
         experiments/phase2/features
                │
                ▼
         experiments/model_comparison_confirm
                │
                ▼
         results/model_comparison_confirm
```

CrowdNav parallel path: `configs/*crowdnav*` → `experiments/crowdnav_<run_id>/` (v1_50k complete; v2/v3 incomplete).

---

## 4. Docs vs implementation inconsistencies

1. **Pool size 48 vs 108 archive** (`reward_generation.yaml` vs Phase-1 artifacts / tests).
2. **Operating feature set history:** older text elevates `behavioral_early`; locked set is `early+math+behavioral`.
3. **Pre-registered confirm** stored as CatBoost+ET on `early+math`; **operating claim** is ET on `early+math+behavioral`.
4. **PPO cost:** some report text ~5k; accounting uses 12k multi-ckpt.
5. **Absolute paths** in reports point to old `I:\Code\...` machines.
6. **AL “next”** in DECISIONS vs no implementation.
7. **CrowdNav** described as deferred while substantial artifacts exist — numbers must not be mixed into 2D claims.

---

## 5. Reproducibility & overwrite risks

| Mechanism | Assessment |
|-----------|------------|
| `seed_everything` | Present |
| Phase-1 resume skip-complete | Safe |
| Confirm dirs fixed paths | **Overwrite risk** if quality pipeline re-run |
| CrowdNav `run_id` versioning | Better than 2D fixed paths |
| Provenance of confirm report | Stale absolute paths; metrics CSVs are the source of truth |

**Recommendation (Stage 0):** copy confirm outputs to a timestamped/read-only archive before any re-run; bump run ids for new campaigns.

---

## 6. Tests (architecture-relevant)

- `tests/test_features.py` — leakage guards  
- `tests/test_model_comparison.py` — LOFO never sees held-out family  
- `tests/test_surrogate.py` — OOF / ranking  
- `tests/test_metrics.py`, `test_reproducibility.py`, `test_dataset.py`  

**Gap:** no test locking numeric confirm claims; no end-to-end AL test.

---

## 7. Bugs / design issues (non-exhaustive)

| Issue | Severity | Notes |
|-------|----------|-------|
| Config can regenerate/overwrite 108 pool | High (process) | Archive protection needed |
| Quality pipeline overwrites confirm | High (process) | Version outputs |
| Early features from labeled run | Medium (validity) | Disclose in claims |
| Inner HPO `min_size=2` vs outer `min_size=4` | Low | Inconsistent family scoring |
| Global `nan_to_num` before CV | Low | Soft leakage-ish |
| Truncated HP grids | Low | Incomplete search, not leakage |
| Missing exploratory model_comparison | Medium (provenance) | Selection-bias story incomplete |
| Behavioral within-family near-constant features | Medium (science) | Inflates dims, little intra signal |

---

## 8. Information available at prediction time

**Valid for unseen-family deployment:**
- Reward spec / static / math / probe behavioral (policy-independent probes)
- Early features **only if** short PPO warmup is in the budget

**Invalid / transductive:**
- Test-family mean centering using held-out labels
- Family one-hot for a truly new family
- Features that require other test-family candidates (unless deployment is batch-transductive and declared as such)
