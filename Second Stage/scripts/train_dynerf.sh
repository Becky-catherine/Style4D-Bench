#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 STYLIZED_DATASET STAGE1_MODEL OUTPUT_DIR [CONFIG]" >&2
    exit 2
fi

stylized_dataset=$1
stage1_model=$2
output_dir=$3
config=${4:-arguments/dynerf/default.py}

python train.py \
    -s "$stylized_dataset" \
    -m "$output_dir" \
    --pretrained_model "$stage1_model" \
    --configs "$config" \
    --disable_gui
