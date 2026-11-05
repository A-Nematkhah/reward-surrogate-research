#!/usr/bin/env python3
"""Run the Phase-2 fair LOFO benchmark into a dedicated output directory.

Never writes to ``experiments/model_comparison_confirm`` unless you override paths.

Example::

    python scripts/run_fair_benchmark.py --quick
    python scripts/run_fair_benchmark.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.model_comparison import load_yaml_config, run_model_comparison
from src.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "benchmark.yaml",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    log = get_logger("run_fair_benchmark")
    cfg = load_yaml_config(args.config)

    out = Path(cfg.get("output_dir", "experiments/benchmark_phase2_base"))
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if "model_comparison_confirm" in str(out).replace("\\", "/"):
        log.error("Refusing to write benchmark into model_comparison_confirm")
        return 2

    if args.n_jobs is not None:
        cfg["n_jobs"] = int(args.n_jobs)
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    if args.results_dir is not None:
        cfg["results_dir"] = str(args.results_dir)

    t0 = time.perf_counter()
    result = run_model_comparison(cfg, quick=bool(args.quick))
    elapsed = time.perf_counter() - t0
    log.info("Elapsed: %.1fs", elapsed)
    log.info("Output: %s", result["output_dir"])

    summary = result["metrics_summary"]
    cols = [
        c
        for c in (
            "model",
            "feature_set",
            "spearman",
            "within_family_spearman_mean",
            "within_family_spearman_identifiable_mean",
            "within_family_top3_hit",
            "within_family_top3_hit_identifiable",
            "n_families_identifiable",
            "mae",
        )
        if c in summary.columns
    ]
    if cols:
        preview = summary[cols].sort_values(
            "within_family_spearman_identifiable_mean"
            if "within_family_spearman_identifiable_mean" in cols
            else "within_family_spearman_mean",
            ascending=False,
        )
        log.info("Benchmark summary:\n%s", preview.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
