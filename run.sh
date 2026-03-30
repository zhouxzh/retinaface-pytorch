#!/usr/bin/env bash

set -euo pipefail

BACKBONES=(
  mobilenet
  mobilenetv2_035
  mobilenetv2_050
  mobilenetv3_small_050
  mobilenetv2_100
  mobilenetv3_small_100
  resnet18
  mobilenetv3_large_100
  mobilenetv4_conv_small
  resnet34
  resnet50
  resnet101
  resnetv2_50
  mobilenetv5_base
)

for backbone in "${BACKBONES[@]}"; do
  echo "============================================================"
  echo "Training backbone: ${backbone}"
  echo "============================================================"
  python main.py train \
    --backbone "${backbone}" 
done