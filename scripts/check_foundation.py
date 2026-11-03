#!/usr/bin/env python3
"""Step 1 smoke check: imports, configs, and seeding."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CONFIG_FILES, load_merged_configs, save_config
from src.utils.logging import get_logger, setup_logging
from src.utils.reproducibility import seed_everything


def main() -> int:
    setup_logging("INFO")
    logger = get_logger("check_foundation")

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Loading all named configs from %s", PROJECT_ROOT / "configs")

    configs = load_merged_configs(config_dir=PROJECT_ROOT / "configs")
    missing = [name for name in CONFIG_FILES if name not in configs]
    if missing:
        logger.error("Missing configs: %s", missing)
        return 1

    for name, cfg in configs.items():
        logger.info("  %-20s keys=%d", name, len(cfg))

    meta = seed_everything(0, deterministic_torch=True)
    logger.info("Seed metadata: %s", meta)

    out_dir = PROJECT_ROOT / "experiments" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / "foundation_config_snapshot.yaml"
    save_config(configs, snapshot_path)
    logger.info("Wrote config snapshot: %s", snapshot_path)

    logger.info("Step 1 foundation check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
