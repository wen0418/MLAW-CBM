#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

model_file="model/mvpcbm_attribute_wavelet_last_new_v5.py"
model_sha="$(sha256sum "$model_file" | awk '{print $1}')"

python -u train_isic2018_fold04_attribute_wavelet_last_new_v5.py \
  --fold 4 \
  --data-path ./dataset/ISIC2018 \
  --output-dir ./output/isic2018/mlaw_evolution_counterfactual_fold04_seed43 \
  --tensorboard-dir ./log/isic2018/mlaw_evolution_counterfactual_fold04_seed43 \
  --expected-model-sha256 "$model_sha" \
  --gpu 0 \
  --train-workers 8 \
  --eval-workers 2 \
  "$@"
