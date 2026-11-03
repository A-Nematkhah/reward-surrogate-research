#!/usr/bin/env python3
"""
Full improvement pipeline (Waves 3–4):

1) regenerate structural candidate pool (~108 / 12 families)
2) Phase 1 PPO (50k × 3 seeds, early behavioral on)
3) build dataset
4) extract features (static/probe/math/interaction/early/hybrid)
5) Phase 2 LOFO surrogates (Top-k / pairwise / HGB / nested HP)
6) regenerate reports

Resume-safe for Phase 1. Run from repo root:

    python scripts/run_full_improvement_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n===", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    py = sys.executable
    run([py, "scripts/generate_candidates.py"])
    run([py, "scripts/run_phase1.py"])
    run([py, "scripts/build_dataset.py", "--min-completed", "50"])
    run([py, "scripts/extract_features.py"])
    run([py, "scripts/run_phase2.py"])
    run([py, "scripts/generate_report.py", "--phase", "all"])
    print("\n=== full improvement pipeline finished OK ===", flush=True)
    print("See results/phase2_report.md for LOFO / Top-k / AL gate.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
