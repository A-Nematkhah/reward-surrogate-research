# Phase 4 sizing (live runs; no separate dataset)

Computed by `scripts/estimate_phase4_cost.py`. Labels are a by-product of each
live pipeline run (train-all / off warm-up). No cross-environment transfer.

## Policy: one environment at a time

1. Run **2D** to completion (including repeated runs for Gate 2 CIs).
2. Review results with the user.
3. Only then start **CrowdNav++** or **IR-SIM** from scratch — never auto-chain.

## Runtime choices

- LLM: **xAI / grok-4.3** (VERIFY exact API model id at client wiring)
- Isolation: **Docker**
- Workers: **4**

## One live run (train-all)

- N=16 candidates/generation × G=5 × seeds=3
- Candidates/run: **80**; jobs/run: **240**

## Active now: 2D

- Independent runs for Gate 2 CIs: **5**
- Jobs: **1200**
- Wall-hours @ 4 workers: **10.0** parallel (40.0 serial)

## Deferred (reference only — do not launch until user go-ahead)

| env | runs | jobs | min/job | wall-h parallel |
| --- | ---: | ---: | ---: | ---: |
| irsim | 1 | 240 | 3.5 | 3.5 |
| crowdnav | 1 | 240 | 6.73 | 6.73 |

## Notes

- Dry-run / `--max-jobs 1` pilots only until the user launches full labeling.
- Audit sample (~15% after active) adds a small overhead not counted above.
- Machine-readable copy: `results/phase4_sizing.json`
