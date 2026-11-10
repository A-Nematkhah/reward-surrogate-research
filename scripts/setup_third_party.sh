#!/usr/bin/env bash
# Clone / pin CrowdNav++ into env/crowdnav (not vendored in-repo).
# VERIFIED against local checkout: upstream URL + commit below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/env/crowdnav"
URL="https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph.git"
PIN="390773137be04ed14e27620dc5fd7c5e1a5b1f62"

mkdir -p "${ROOT}/env"
if [[ ! -d "${DEST}/.git" ]]; then
  git clone "${URL}" "${DEST}"
fi
git -C "${DEST}" fetch --all --tags
git -C "${DEST}" checkout "${PIN}"
echo "CrowdNav++ at ${DEST} @ $(git -C "${DEST}" rev-parse HEAD)"
