#!/usr/bin/env bash

set -euo pipefail

BACKBONES=(
  # speed-first
  # mobilenetv2_035
  # mobilenetv2_050
  # mobilenetv3_small_050
  # mobilenetv3_small_100

  # balanced
  # mobilenetv2_100
  # mobilenetv3_large_100
  # mobilenetv4_conv_small_050
  # mobilenetv4_conv_small
  # resnet18
  # resnet34

  # accuracy-first
  # resnet50
  mobilenetv5_300m
  mobilenetv5_base

  # heavier accuracy candidates
  resnet101
  resnetv2_50
  mobilenetv4_conv_medium
  resnet152
)

get_batch_size() {
  local backbone="$1"

  case "$backbone" in
    mobilenetv2_100|mobilenetv3_large_100|mobilenetv4_conv_small_050|mobilenetv4_conv_small|resnet18|resnet34)
      echo 16
      ;;
    resnet50|mobilenetv5_300m|mobilenetv5_base)
      echo 2
      ;;
    resnet101|resnetv2_50|mobilenetv4_conv_medium|resnet152)
      echo 4
      ;;
    *)
      echo 32
      ;;
  esac
}

for backbone in "${BACKBONES[@]}"; do
  batch_size="$(get_batch_size "${backbone}")"

  echo "============================================================"
  echo "Training backbone: ${backbone}"
  echo "Batch size: ${batch_size}"
  echo "============================================================"
  python main.py train \
    --backbone "${backbone}" \
    --batch_size "${batch_size}"
done