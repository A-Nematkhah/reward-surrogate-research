#!/usr/bin/env python3
"""Build Phase-3 extra feature tables without re-running probes/PPO.

Writes into the existing Phase-2 features directory (additive CSVs only):
  - features_ratios.csv
  - features_early_deltas.csv  (subset of early dynamics columns)

Does not overwrite static/behavioral/early full tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rsr.features.feature_builder import aggregate_phase1_dataset
from rsr.features.ratio_features import ratio_feature_matrix_from_dataframe
from rsr.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "phase1"
        / "dataset"
        / "reward_performance_dataset.csv",
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "phase2" / "features",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    log = get_logger("build_phase3_feature_extras")

    ds = pd.read_csv(args.dataset)
    agg = aggregate_phase1_dataset(ds, how="mean")
    meta = pd.read_csv(args.features_dir / "features_meta.csv")
    # Align to feature table candidate order
    agg = agg.set_index("candidate_id").loc[meta["candidate_id"]].reset_index()
    ratios = ratio_feature_matrix_from_dataframe(agg)
    ratios_path = args.features_dir / "features_ratios.csv"
    ratios.to_csv(ratios_path, index=False)
    log.info("Wrote %s (%d cols)", ratios_path, ratios.shape[1])

    early = pd.read_csv(args.features_dir / "features_early.csv")
    delta_cols = [
        c
        for c in early.columns
        if c.startswith("early_d")
        or c.startswith("early_span_")
        or c == "early_n_checkpoints"
    ]
    deltas = early[delta_cols].copy()
    deltas_path = args.features_dir / "features_early_deltas.csv"
    deltas.to_csv(deltas_path, index=False)
    log.info("Wrote %s (%d cols)", deltas_path, deltas.shape[1])

    man_path = args.features_dir / "features_manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["n_ratios_features"] = int(ratios.shape[1])
        man["n_early_deltas_features"] = int(deltas.shape[1])
        man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
        log.info("Updated manifest extras counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
