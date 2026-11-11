# Phase 6 report — live 2D campaign → Gate 2

**Status: complete. Gate 2 verdict = FAIL** (see `docs/phase_reports/gate2.md`).

## Delivered

1. Pre-reg: `docs/prereg/gate2.md`, `docs/prereg/phase6.md`
2. Live 2D runs (local LLM corpus, real PPO 50k, active after `m_min=12`, auto-adapt):
   - `experiments/phase6_2d_r1b/`
   - `experiments/phase6_2d_r2/`
   - `experiments/phase6_2d_r3/`
3. Analysis: `scripts/run_gate2_analysis.py` → `experiments/gate2_2d/`, `results/gate2_2d/`

## Campaign snapshot

| Metric | Value |
| --- | ---: |
| Runs | 3 |
| Candidates / run | 48 |
| Labeled / run | 34 |
| Pooled labeled | 102 |
| Lineages (labeled) | 48 |
| Wall / run | ~50–51 min |

Partial crash artifact `phase6_2d_r1` (empty store) was abandoned; confirmatory runs are `r1b`+`r2`+`r3`.

## Gate 2 (one-line)

Recall@5 mean **0.56** (need 0.80); Δ vs random CI includes 0; regret and savings pass. **FAIL — stop for human decision.**

## Out of scope

CrowdNav++, IR-SIM, Gate 3, Docker, xAI — not started.
