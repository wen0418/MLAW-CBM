#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

model_file="model/mlaw_cbm_configurable.py"
model_sha="$(sha256sum "$model_file" | awk '{print $1}')"

python -u -m training.scalp.train_mlaw_cbm \
  --expected-model-sha256 "$model_sha" \
  "$@"
