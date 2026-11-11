# Feature-set decision

**Locked:** `early+math+behavioral`

Evidence (`scripts/probe_local_surrogate.py`, 108 LOFO ExtraTrees):

| set | within-family Spearman |
| --- | ---: |
| math+behavioral | 0.586 |
| early | 0.583 |
| **early+math+behavioral** | **0.641** |

Implication: Phase 4 active scoring runs early checkpoints (2k/5k/12k per `configs/ppo.yaml`) on candidates before surrogate score; full-run PPO only for selected k.
