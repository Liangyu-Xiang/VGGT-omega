#!/usr/bin/env bash
set -euo pipefail

# Formal SelTR lambda ablation: full 7Scenes test split, stride 3, no frame cap.
# Usage: bash scripts/run_7scenes_lambda_ablation.sh /path/to/vggt_checkpoint.pt [output_root]
CHECKPOINT=${1:?"usage: $0 CHECKPOINT [OUTPUT_ROOT]"}
OUTPUT_ROOT=${2:-outputs/7scenes_lambda_ablation_stride3_full}
DATASET_ROOT=${DATASET_ROOT:-/data_SSD1/mmc_lyxiang/3D/benchmark_stride_eval/datasets/7scenes}
PYTHON_BIN=${EVAL_PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
GPU_0=${GPU_0:-4}
GPU_1=${GPU_1:-5}
GPU_2=${GPU_2:-6}

run_group() {
  local gpu=$1
  shift
  for lambda_cost in "$@"; do
    local output="$OUTPUT_ROOT/lambda_${lambda_cost}"
    if [[ -f "$output/metrics.json" ]]; then
      echo "lambda=${lambda_cost}: existing metrics found; skipping" >&2
      continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" scripts/eval_7scenes.py \
      --method selftr --checkpoint "$CHECKPOINT" --dataset-root "$DATASET_ROOT" \
      --output-dir "$output" --device cuda:0 --stride 3 --lambda-cost "$lambda_cost"
  done
}

run_group "$GPU_0" 0.02 0.05 0.08 &
PID_0=$!
run_group "$GPU_1" 0.01 0.06 0.09 &
PID_1=$!
run_group "$GPU_2" 0.03 0.07 0.1 &
PID_2=$!
wait "$PID_0" "$PID_1" "$PID_2"
"$PYTHON_BIN" scripts/summarize_7scenes_lambda_ablation.py --root "$OUTPUT_ROOT"
