#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_scannet50_500.sh /path/to/vggt_checkpoint.pt [output_root]
CHECKPOINT=${1:?"usage: $0 CHECKPOINT [OUTPUT_ROOT]"}
OUTPUT_ROOT=${2:-outputs/scannet50_500}
SCANNET_DATA_ROOT=${SCANNET_DATA_ROOT:-/data_SSD1/mmc_lyxiang/dataset/scannet50_data}
DENSE_GPU=${DENSE_GPU:-4}
FAST_GPU=${FAST_GPU:-5}
SELFTR_GPU=${SELFTR_GPU:-6}
PYTHON_BIN=${EVAL_PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
COMMON=(--checkpoint "$CHECKPOINT" --dataset-root "$SCANNET_DATA_ROOT" --num-frames 500 --require-exact-frames --save-visualizations --resume)

CUDA_VISIBLE_DEVICES=$DENSE_GPU "$PYTHON_BIN" scripts/eval_scannet50.py --method densevggt --device cuda:0 --output-dir "$OUTPUT_ROOT/densevggt" "${COMMON[@]}" &
PID_DENSE=$!
CUDA_VISIBLE_DEVICES=$FAST_GPU "$PYTHON_BIN" scripts/eval_scannet50.py --method fastvggt --device cuda:0 --output-dir "$OUTPUT_ROOT/fastvggt" "${COMMON[@]}" &
PID_FAST=$!
CUDA_VISIBLE_DEVICES=$SELFTR_GPU "$PYTHON_BIN" scripts/eval_scannet50.py --method selftr --device cuda:0 --output-dir "$OUTPUT_ROOT/selftr" "${COMMON[@]}" &
PID_SELFTR=$!
wait "$PID_DENSE" "$PID_FAST" "$PID_SELFTR"
