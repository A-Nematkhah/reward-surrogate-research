# Simulator backends

External and local simulation backends live here, **one folder per environment**.
Gymnasium adapters and the research pipeline stay under `src/environment/`.

| Folder | Backend | Gym / pipeline id | CLI |
|--------|---------|-------------------|-----|
| `navigation_2d/` | Native 2D nav (code in `src/environment/`) | `RewardNavigation2D-v0` | `--env 2d` |
| `crowdnav/` | CrowdNav++ (ICRA 2023) | `CrowdNavPlusPlus-v0` | `--env crowdnav` |
| `ir-sim/` | [IR-SIM](https://github.com/hanruihua/ir-sim) | `IRSimNav-v0` | `--env irsim` |
| `worlds/` | Tracked IR-SIM scenario YAML (outside the clone) | `irsim_pipeline_nav.yaml` | |

```bash
python scripts/run_quality_pipeline.py --env 2d
python scripts/run_quality_pipeline.py --env crowdnav --run-id v5_invisible_24
python scripts/run_quality_pipeline.py --env irsim --run-id v1_nav24
```

Each versioned backend writes only under `experiments/<prefix>_<run_id>/` and `results/<prefix>_<run_id>/`.

## Clone / refresh

```bash
# CrowdNav++ (if missing)
git clone --depth 1 https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph.git env/crowdnav

# IR-SIM (if missing)
git clone --depth 1 https://github.com/hanruihua/ir-sim.git env/ir-sim
```

Heavy trees under `env/crowdnav/` and `env/ir-sim/` are gitignored; this README stays tracked.
