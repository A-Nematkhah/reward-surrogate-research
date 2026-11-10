# Simulator backends

External and local simulation backends live here, **one folder per environment**.
Gymnasium adapters and the research pipeline stay under `rsr/environment/`.

| Folder | Backend | Gym / pipeline id | CLI |
|--------|---------|-------------------|-----|
| `navigation_2d/` | Native 2D nav (code in `rsr/environment/`) | `RewardNavigation2D-v0` | `--env 2d` |
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
# CrowdNav++ pinned clone (preferred)
bash scripts/setup_third_party.sh
# Equivalent: git clone https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph.git env/crowdnav
#            git -C env/crowdnav checkout 390773137be04ed14e27620dc5fd7c5e1a5b1f62

# IR-SIM (if missing)
git clone --depth 1 https://github.com/hanruihua/ir-sim.git env/ir-sim
```

Heavy trees under `env/crowdnav/` and `env/ir-sim/` are gitignored (external clones, not vendored); this README stays tracked.
