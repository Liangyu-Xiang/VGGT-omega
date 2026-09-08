#!/usr/bin/env bash
set -euo pipefail

# Full VGGT-Omega 7Scenes ablation:
#   baseline, fixed compression 10/25/50%, refresh every global block,
#   refresh every 4/8/16 transformer blocks, and last representative token.
#
# Usage:
#   bash scripts/run_7scenes_vggt_ablation.sh [output_root] [gpu ...]
# Required environment:
#   SEVEN_SCENES_ROOT=/path/to/7scenes
#   VGGT_OMEGA_CHECKPOINT=/path/to/vggt_omega_1b_512.pt

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-$ROOT/outputs/7scenes_vggt_ablation_$(date -u +%Y%m%d_%H%M%S)}"
shift || true
GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(3 4)
fi

PYTHON="${VGGT_OMEGA_PYTHON:-python3}"
DATA_ROOT="${SEVEN_SCENES_ROOT:-}"
CHECKPOINT="${VGGT_OMEGA_CHECKPOINT:-}"
TIMING_REPEATS="${TIMING_REPEATS:-3}"
MAX_GEOMETRY_POINTS="${MAX_GEOMETRY_POINTS:-100000}"

if [ -z "$DATA_ROOT" ] || [ -z "$CHECKPOINT" ]; then
  echo "Set SEVEN_SCENES_ROOT and VGGT_OMEGA_CHECKPOINT before running." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT/logs"
QUEUE="$OUT_ROOT/tasks.tsv"
LOCK="$OUT_ROOT/tasks.lock"

if [ ! -f "$QUEUE" ]; then
  {
    printf '%s\t%s\t%s\n' baseline none none
    printf '%s\t%s\t%s\n' fixed_c10 fixed_compression_10 0.10
    printf '%s\t%s\t%s\n' fixed_c25 fixed_compression_25 0.25
    printf '%s\t%s\t%s\n' fixed_c50 fixed_compression_50 0.50
    printf '%s\t%s\t%s\n' refresh_all refresh_every_global none
    printf '%s\t%s\t%s\n' refresh_4 refresh_every_4_blocks 0,4,8,12,16,20
    printf '%s\t%s\t%s\n' refresh_8 refresh_every_8_blocks 0,8,16
    printf '%s\t%s\t%s\n' refresh_16 refresh_every_16_blocks 0,16
    printf '%s\t%s\t%s\n' last_representative last_representative 0,10,17
  } > "$QUEUE"
fi

pop_task() {
  "$PYTHON" - "$QUEUE" "$LOCK" <<'PY'
from pathlib import Path
import fcntl
import sys

queue = Path(sys.argv[1])
lock = Path(sys.argv[2])
with lock.open("w") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    lines = queue.read_text().splitlines() if queue.exists() else []
    if lines:
        print(lines[0])
        queue.write_text("\n".join(lines[1:]) + ("\n" if len(lines) > 1 else ""))
    else:
        print("")
    fcntl.flock(handle, fcntl.LOCK_UN)
PY
}

run_one() {
  local gpu="$1"
  local label="$2"
  local kind="$3"
  local value="$4"
  local output="$OUT_ROOT/$label"
  local log="$OUT_ROOT/logs/${label}.gpu${gpu}.log"
  mkdir -p "$output"
  {
    date -u +"[%Y-%m-%dT%H:%M:%SZ] start label=$label gpu=$gpu"
    args=(
      --data-root "$DATA_ROOT"
      --checkpoint "$CHECKPOINT"
      --output-dir "$output"
      --device cuda:0
      --num-frames 300
      --sampling-unit sequence
      --sampling-strategy uniform
      --sampling-stride 3
      # 384px still exceeds the 32GB budget for 300-frame global attention;
      # all ablations use the same 256px setting so the protocol stays
      # comparable while retaining the requested 300-frame context.
      --image-resolution 256
      --resize-mode max_size
      --timing-repeats "$TIMING_REPEATS"
      --max-depth 10
      --max-geometry-points "$MAX_GEOMETRY_POINTS"
    )
    if [ "$kind" = none ]; then
      args+=(--acceleration-method none --merge-ratio 0 --frame-fusion-mode none)
    else
      args+=(
        --acceleration-method u-m
        --merge-ratio 0
        --frame-fusion-mode u-m
        --frame-fusion-start-layer -1
        --frame-fusion-lambda-cost 0.04
        --frame-fusion-merge-top-similarity-percent 100
        --frame-fusion-min-keep-ratio 0.05
        --frame-fusion-temporal-window 4
        --frame-fusion-spatial-radius 2
        --frame-fusion-attention-variant representative
      )
      case "$kind" in
        fixed_compression_10|fixed_compression_25|fixed_compression_50)
          args+=(
            --frame-fusion-recompute-layers 0,10,17
            --frame-fusion-fixed-compression-ratio "$value"
          )
          ;;
        refresh_every_global)
          args+=(
            --frame-fusion-recompute-each-global
            --frame-fusion-recompute-layers none
          )
          ;;
        refresh_every_4_blocks|refresh_every_8_blocks|refresh_every_16_blocks|last_representative)
          args+=(--frame-fusion-recompute-layers "$value")
          ;;
        *)
          echo "unknown experiment kind: $kind" >&2
          return 2
          ;;
      esac
      if [ "$kind" = last_representative ]; then
        args+=(--frame-fusion-attention-variant last-representative)
      fi
    fi
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" \
    VGGT_UM_TRITON="${VGGT_UM_TRITON:-1}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}" \
      "$PYTHON" "$ROOT/scripts/eval_7scenes_paper.py" "${args[@]}"
    status="$?"
    set -e
    date -u +"[%Y-%m-%dT%H:%M:%SZ] finish label=$label gpu=$gpu status=$status"
    return "$status"
  } > "$log" 2>&1
}

worker() {
  local gpu="$1"
  local worker_log="$OUT_ROOT/logs/worker_gpu${gpu}.log"
  echo "worker gpu=$gpu started" > "$worker_log"
  while true; do
    line="$(pop_task)"
    [ -z "$line" ] && break
    IFS=$'\t' read -r label kind value <<< "$line"
    echo "$(date -u +%FT%TZ) start $label" >> "$worker_log"
    if run_one "$gpu" "$label" "$kind" "$value"; then
      echo "$(date -u +%FT%TZ) ok $label" >> "$worker_log"
    else
      echo "$(date -u +%FT%TZ) fail $label" >> "$worker_log"
      printf '%s\t%s\t%s\n' "$label" "$kind" "$value" >> "$OUT_ROOT/failed_tasks.tsv"
    fi
  done
  echo "worker gpu=$gpu finished" >> "$worker_log"
}

echo "$OUT_ROOT" > "$ROOT/outputs/latest_7scenes_vggt_ablation_path.txt"
echo "Output root: $OUT_ROOT"
echo "GPUs: ${GPUS[*]}"
echo "Python: $PYTHON"
echo "Dataset: $DATA_ROOT"
echo "Checkpoint: $CHECKPOINT"

for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
done
wait

echo "All experiments finished. Summarizing results."
expected_labels=(
  baseline fixed_c10 fixed_c25 fixed_c50 refresh_all refresh_4 refresh_8
  refresh_16 last_representative
)
missing=()
for label in "${expected_labels[@]}"; do
  if [ ! -f "$OUT_ROOT/$label/metrics.json" ]; then
    missing+=("$label")
  fi
done
if [ "${#missing[@]}" -ne 0 ]; then
  echo "Missing completed experiments: ${missing[*]}" >&2
  exit 1
fi
"$PYTHON" "$ROOT/scripts/summarize_7scenes_vggt_ablation.py" \
  "$OUT_ROOT" --output-dir "$OUT_ROOT/summary"
