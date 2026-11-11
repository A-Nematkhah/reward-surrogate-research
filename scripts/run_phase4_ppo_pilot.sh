#!/usr/bin/env bash
# Run one Phase-4 PPO pilot job. Prefers Docker; falls back to local Python.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN_ID="${1:-phase4_2d_v1}"

if command -v docker >/dev/null 2>&1; then
  echo "Using Docker isolation"
  exec docker compose -f docker-compose.triage.yml run --rm triage-labeler \
    python scripts/label_phase4_job.py --run-id "$RUN_ID" --max-jobs 1
fi

echo "WARNING: docker not found; running local pilot (weaker isolation)" >&2
exec python scripts/label_phase4_job.py --run-id "$RUN_ID" --max-jobs 1 --seed 0
