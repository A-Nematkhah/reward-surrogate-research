#!/usr/bin/env bash
# Run a command inside the triage sandbox (no network, non-root, read-only repo).
# LLM API keys must not be exported into this process.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not available" >&2
  exit 2
fi

docker compose -f docker-compose.triage.yml build triage-sandbox
exec docker compose -f docker-compose.triage.yml run --rm \
  -e OPENAI_API_KEY= \
  -e ANTHROPIC_API_KEY= \
  triage-sandbox \
  "$@"
