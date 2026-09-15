#!/usr/bin/env bash
set -euo pipefail

# Formal SelTR lambda ablation: full 7Scenes test split, stride 3, no frame cap.
# Usage: bash scripts/run_7scenes_lambda_ablation.sh CHECKPOINT DATASET_ROOT GPU_LIST [OUTPUT_ROOT]
# GPU_LIST is comma-separated (for example: 0 or 0,1,3). No server path/GPU is assumed.
CHECKPOINT=${1:?"usage: $0 CHECKPOINT DATASET_ROOT GPU_LIST [OUTPUT_ROOT]"}
DATASET_ROOT=${2:?"usage: $0 CHECKPOINT DATASET_ROOT GPU_LIST [OUTPUT_ROOT]"}
GPU_LIST=${3:?"usage: $0 CHECKPOINT DATASET_ROOT GPU_LIST [OUTPUT_ROOT]"}
OUTPUT_ROOT=${4:-outputs/7scenes_lambda_ablation_stride3_full}
PYTHON_BIN=${EVAL_PYTHON:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
(( ${#GPUS[@]} > 0 )) || { echo "GPU_LIST must not be empty" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU id: $gpu" >&2; exit 2; }
done
LAMBDAS=(0.02 0.01 0.03 0.05 0.06 0.07 0.08 0.09 0.1)

run_group() {
  local gpu=$1
  local include_dense=$2
  shift 2
  if [[ "$include_dense" == "true" && ! -f "$OUTPUT_ROOT/densevggt/metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" scripts/eval_7scenes.py \
      --method vggt --checkpoint "$CHECKPOINT" --dataset-root "$DATASET_ROOT" \
      --output-dir "$OUTPUT_ROOT/densevggt" --device cuda:0 --stride 3
  fi
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

PIDS=()
for worker in "${!GPUS[@]}"; do
  GROUP=()
  for index in "${!LAMBDAS[@]}"; do
    if (( index % ${#GPUS[@]} == worker )); then
      GROUP+=("${LAMBDAS[index]}")
    fi
  done
  if (( worker == 0 )); then
    run_group "${GPUS[worker]}" true "${GROUP[@]}" &
  else
    run_group "${GPUS[worker]}" false "${GROUP[@]}" &
  fi
  PIDS+=("$!")
done
wait "${PIDS[@]}"
"$PYTHON_BIN" scripts/summarize_7scenes_lambda_ablation.py --root "$OUTPUT_ROOT"
