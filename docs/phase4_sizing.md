# Phase 4 sizing (no PPO launched)

Computed by `scripts/estimate_phase4_cost.py`. Reference CrowdNav job time:
6.73 min/job at 150k steps (v5).

## Candidate design

- Prompts: 6; temperatures: [0.3, 0.7, 1.0]; gen0 size per (prompt,T): 16
- Generation-0 candidates: **288**
- Generations planned: **3** → target candidates/env: **864** (≥300 floor)
- Seeds/candidate: **3** → jobs/env: **2592**

## Lineages for Gate 2 (≥20 held-out)

- Held-out fraction: 20%
- Total lineages needed: **100**
- Held-out lineages: **20**; train: **80**
- Planned lineages (≥ prompts×temps): **100**
- Also hold out **2** prompts in `SPLITS.json`

## Wall-hour estimates

Workers assumed: **4**.

| env | candidates | jobs | min/job | wall-h serial | wall-h parallel |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2d | 864 | 2592 | 2.0 | 86.4 | 21.6 |
| irsim | 864 | 2592 | 3.5 | 151.2 | 37.8 |
| crowdnav | 864 | 2592 | 6.73 | 290.74 | 72.68 |

**Total parallel wall-hours (all envs, sequential envs):** 132.08

Env order: 2D → IR-SIM → CrowdNav++.

## Notes

- Dry-run / `--max-jobs 1` pilots only until the user launches full labeling.
- Audit quota (10% of Tier-0 rejects) adds a small job overhead not counted above.
- Machine-readable copy: `results/phase4_sizing.json`
