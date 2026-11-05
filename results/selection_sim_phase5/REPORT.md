# Offline selection simulation (Phase 5)

- OOF source: confirm `extra_trees` / `early+math+behavioral` / repeat=0
- Candidates: **108**
- Global best SR: **0.8417**
- Smallest k with surrogate best within 0.02 of oracle: **3**

| k | oracle best | surrogate best | random best (mean) | regret vs oracle | hit oracle#1 |
|--:|------------:|---------------:|-------------------:|-----------------:|-------------:|
| 1 | 0.8417 | 0.7969 | 0.3639 | 0.0448 | 0.00 |
| 3 | 0.8417 | 0.8271 | 0.5672 | 0.0146 | 0.00 |
| 5 | 0.8417 | 0.8417 | 0.6740 | 0.0000 | 1.00 |
| 10 | 0.8417 | 0.8417 | 0.7666 | 0.0000 | 1.00 |
| 20 | 0.8417 | 0.8417 | 0.8174 | 0.0000 | 1.00 |
| 30 | 0.8417 | 0.8417 | 0.8304 | 0.0000 | 1.00 |
| 50 | 0.8417 | 0.8417 | 0.8374 | 0.0000 | 1.00 |
| 108 | 0.8417 | 0.8417 | 0.8417 | 0.0000 | 1.00 |

## Gate for real AL

- **This run:** surrogate top-k **beats random** at small k; global best reached by **k=5** (near-oracle by k=3).
- Online AL still deferred until an authorized new candidate pool / run_id is designed.
- Caveat: LOFO OOF scores are slightly optimistic vs a single frozen model trained once.
