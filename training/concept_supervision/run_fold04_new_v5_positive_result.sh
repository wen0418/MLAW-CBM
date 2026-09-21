#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

model_file="model/mvpcbm_attribute_wavelet_last_new_v5_positive_result.py"
model_sha="$(sha256sum "$model_file" | awk '{print $1}')"

python -u training/concept_supervision/train_fold04_new_v5_positive_result.py \
  --fold 4 \
  --data-path ./dataset/ISIC2018 \
  --pseudo-label-csv ./dataset/pseudo_labels/training.csv \
  --concept-label-column positive_result \
  --output-dir ./output/isic2018/baseline_cv10_attribute_wavelet_last_new_v5_positive_result_k98_imglevel_fold04_seed43 \
  --tensorboard-dir ./log/isic2018/baseline_cv10_attribute_wavelet_last_new_v5_positive_result \
  --expected-model-sha256 "$model_sha" \
  --attribute-wavelet-top-k 98 \
  --attribute-temperature 0.07 \
  --counterfactual-route-temperature 0.05 \
  --initial-high-feature-scale 0.1 \
  --initial-high-score-weight 1.0 \
  --attribute-wavelet-eps 1e-6 \
  --gpu 0 \
  --train-workers 8 \
  --eval-workers 2 \
  "$@"
