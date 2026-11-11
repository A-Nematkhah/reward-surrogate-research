# Gate 2d — corrected ablations + counterfactual steps (EXPLORATORY)

**Gate 2 verdict unchanged: FAIL.** Thresholds not edited. No new PPO.  
Artifacts: `experiments/gate2d/gate2d_report.json`. Script: `scripts/run_gate2d_analysis.py`.

All analytic claims below are **EXPLORATORY**.

## 0. Push status

`experiments/gate2b_diagnostics/` and `experiments/gate2c_harness/` were already on `origin/main` in commit `7dc160a` (phase6bc). No extra push required for item 0.

## 1. Feature ablation with code_hash folds (no hash leak)

Protocol: dedupe by `code_hash` → LOLO with `enforce_no_hash_leak=True` → pools N=16, k=5.  
**Gen0 and plateau are never pooled for a claim.** Random-k is evaluated on the **same** pools.  
Pool composition: mean share gen0 vs plateau printed per slice.

### early+math+behavioral

| Slice | n | surr recall@5 | random@5 (same pools) | share gen0 / plateau |
| --- | ---: | ---: | ---: | ---: |
| gen0 | 16 | 1.00† | ~0.30 | 1.00 / 0.00 |
| plateau | 53 | **0.38** | **0.31** | 0.00 / 1.00 |

### math+behavioral

| Slice | n | surr | random | share g0/plat |
| --- | ---: | ---: | ---: | ---: |
| gen0 | 16 | 1.00† | ~0.33 | 1.00 / 0.00 |
| plateau | 53 | **0.375** | **0.283** | 0.00 / 1.00 |

### early only

| Slice | n | surr | random | share g0/plat |
| --- | ---: | ---: | ---: | ---: |
| gen0 | 16 | 1.00† | ~0.33 | 1.00 / 0.00 |
| plateau | 53 | **0.25** | **0.333** | 0.00 / 1.00 |

†**Degenerate:** only 16 unique gen0 hashes → every N=16 pool is the full set → surrogate top-5 always contains the true best if it scores anyone; random@5 ≈ k/N ≈ 0.31. **Do not use gen0 pooled recall as skill.**

On the **plateau** (the only non-degenerate slice): emb ≈ mb ≳ random; **early alone ≤ random**.

## 2. Why Gate 2c pooled recall 0.493 ≠ “random ~0.30”

| Quantity | Value | Meaning |
| --- | ---: | --- |
| Gate 2c pooled surrogate recall (mixed gen0+plateau, code_hash folds) | 0.493 | EXPLORATORY mixed slice (forbidden as a claim in 6d) |
| Analytic random-k on a uniform N=16 pool | **k/N = 5/16 = 0.3125** | Chance of keeping the single best if you pick 5 of 16 at random |
| MC uniform from 69 distinct hashes | ≈ 0.31 | Matches analytic |

**Composition:** Gate 2c’s 0.493 is surrogate skill on **mixed** expand_groups pools (gen0 + plateau hashes). The ~0.30 figure is the **random-k baseline on any N=16 pool**, not a failed surrogate. On plateau-only same-pool draws, surrogate ~0.38 vs random ~0.31 (emb). The gap to 0.80 remains large; this does not revive Gate 2.

## 3. eps-recall (decision-tied tolerances)

Plateau slice, emb OOF, same-pool surrogate vs random:

| Tolerance | Surrogate | Random |
| --- | ---: | ---: |
| **1 SE** of pool best | 0.70 | 0.58 |
| **0.02** absolute | 0.83 | 0.75 |
| Hard recall (exact best) | 0.38 | 0.31 |

Gen0 eps = 1.0 for surrogate (degenerate pools).  
**EXPLORATORY** until separately preregistered.

## 4. Counterfactual steps-to-target (observed labels only)

Best-so-far success vs cumulative PPO seed-steps (sum of `train_steps` on labeled seeds).

| Policy | Final best success | Cum seed-steps |
| --- | ---: | ---: |
| (a) train-all (observed labels, pay full cost every time) | 0.840 | 15 300 000 |
| (b) + label cache (dup `code_hash` → 0 steps) | 0.840 | 10 350 000 |
| (c) + plateau stop (stop further gens after a gen with no improve &gt; 2·SE) | 0.827 | 6 300 000 |
| (d) + filter (train/audit only) | 0.840 | 10 350 000 |

On this log, (d) matches (b) because every labeled row was already train/audit.

**Flags (required):**
- All labels treated as observed; true train-all would also label skips (missing here).
- Final picks need **independent re-evaluation**.
- Policies must be **saved** in future runs (paths already under `runs/`; pin best-of-run in `run_summary`).

## 5. Real-LLM pilot plan (DO NOT RUN in this phase)

Document only — needs explicit user go-ahead + API key + new prereg if confirmatory:

1. One 2D run: N=16 × G=3 × 3 seeds; `run_id=phase6d_xai_pilot1`.
2. xAI provider; **VERIFY** exact model id (`grok-4.3` was planned).
3. Temperature ≥ 0.9; mutation prompt with an explicit **diversity / near-duplicate reject** rule (`code_hash` / mb NN).
4. Default **run_id seeds** + **label cache** on.
5. Save final policies; independent re-eval of the final pick.
6. Start with `--max-jobs 1` smoke; full pilot only after user approval.

## Stop

No tuning. No threshold changes. Gate 2 FAIL stands. Await human decision.
