# Offline selection simulation (Phase 5)

- OOF source: confirm `extra_trees` / `early+math+behavioral` / repeat=1
- Candidates: **72**
- Global best SR: **0.8083**
- Smallest k with surrogate best within 0.02 of oracle: **20**

| k | oracle best | surrogate best | random best (mean) | regret vs oracle | hit oracle#1 |
|--:|------------:|---------------:|-------------------:|-----------------:|-------------:|
| 1 | 0.8083 | 0.5854 | 0.4257 | 0.2229 | 0.00 |
| 3 | 0.8083 | 0.6167 | 0.6116 | 0.1917 | 0.00 |
| 5 | 0.8083 | 0.6167 | 0.6644 | 0.1917 | 0.00 |
| 10 | 0.8083 | 0.7646 | 0.7220 | 0.0437 | 0.00 |
| 15 | 0.8083 | 0.7646 | 0.7445 | 0.0437 | 0.00 |
| 20 | 0.8083 | 0.8083 | 0.7562 | 0.0000 | 1.00 |
| 30 | 0.8083 | 0.8083 | 0.7735 | 0.0000 | 1.00 |
| 50 | 0.8083 | 0.8083 | 0.7943 | 0.0000 | 1.00 |
| 72 | 0.8083 | 0.8083 | 0.8083 | 0.0000 | 1.00 |

## Gate for real AL

- Proceed to online AL only if surrogate clearly beats random at small k
  and regret vs oracle shrinks monotonically.
- This simulation uses LOFO OOF scores (slightly optimistic vs a single frozen model).
