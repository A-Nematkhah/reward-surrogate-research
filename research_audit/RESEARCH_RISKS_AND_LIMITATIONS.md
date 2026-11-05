# Research Risks & Limitations

## Threats to validity

### Internal
- **Early↔label coupling:** early features from the same PPO run as `success_rate` can inflate apparent predictability vs a true cold-start surrogate.
- **Researcher degrees of freedom:** reusing the same 12 LOFO folds for many exploratory ideas; selection bias if max over grids is reported as unbiased.
- **Metric contamination:** averaging within-Spearman over flat families; pooled top-10 presented as within-family skill.
- **Seed underpowering:** 3 seeds → unstable pairwise truth for close candidates.
- **Outlier filtering:** dropping seeds changes n and may favor cleaner but optimistic labels — must be pre-specified.

### External
- Toy 2D navigation ≠ CrowdNav / real robots.
- Fixed PPO algorithm & HPs (Booth et al.): rewards may overfit this trainer.
- Probe policies may not match deployment behavior distribution.
- Family taxonomy is researcher-defined; “unseen family” depends on that ontology.

### Construct
- Predicting `success_rate` under dense shaped rewards is not the same as predicting human-intended task quality under all metrics (collision, time, path).
- Surrogate quality ≠ proof that AL reduces wall-clock to better rewards without a closed-loop study.

### Statistical conclusion
- n_families=12 → low power; wide CIs; fragile “significant improvement” claims.
- Dependent pairs must not inflate n.
- Confirm repeats=3 are RNG repeats, not independent datasets.

---

## Claims the evidence **can** support (2D)

1. Under LOFO, ExtraTrees with early+math+behavioral features achieves **high pooled** Spearman (~0.94) and **moderate mean within-family** Spearman (~0.65) with Top-3 hit ~0.83 on the 108-candidate archive.
2. Between-family structure explains a large share of SR variance (~64%).
3. Within-family performance is **heterogeneous** across families.
4. Implementation of outer LOFO + train-only preprocessing is largely correct.

---

## Claims the evidence **cannot** support

1. The brief’s exact table (0.8448 / 0.5751 / …) as the locked 2D ExtraTrees result.  
2. That ranking models will fix within-family weakness (untested as primary).  
3. That the surrogate replaces PPO for final reward selection without residual error.  
4. Transfer to CrowdNav or other envs.  
5. Cross-algorithm / cross-HP reward ranking stability.  
6. That Active Learning works (not implemented).  
7. That deep nets are appropriate or necessary.  
8. Causal claim that specific feature groups “cause” PPO success beyond predictive association.

---

## Process risks

| Risk | Mitigation |
|------|------------|
| Overwriting confirm artifacts | Archive + new run_ids |
| Mixing CrowdNav & 2D numbers | Separate results trees; explicit env in every table |
| Regenerating 48-candidate pool over 108 archive | Read-only archive; generation writes new path |
| Silent missing exploratory grid | Disclose missing `experiments/model_comparison/` |
| Scope creep into CrowdNav PPO | Explicit authorization gate |

---

## Ethical / scientific communication risks

- Overstating surrogate reliability (EWRL 2025 caution applies in spirit).  
- Hiding feature PPO cost (12k steps) while claiming cheap prediction.  
- Presenting exploratory winners as pre-registered.
