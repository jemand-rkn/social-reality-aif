#!/usr/bin/env bash
set -euo pipefail

REFRESH=""
NUM_WORKERS_ARG=""
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --refresh) REFRESH="--refresh" ;;
        --num-workers=*) NUM_WORKERS_ARG="${arg#--num-workers=}" ;;
        --num-workers) echo "Error: --num-workers requires a value (e.g. --num-workers=32)"; exit 1 ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

if [ ${#POSITIONAL[@]} -lt 1 ]; then
    echo "Usage: $0 <input-dir> [snapshot-steps] [--refresh] [--num-workers=N]"
    echo "  input-dir:        Hydra output directory"
    echo "  snapshot-steps:   Comma-separated steps for render-snapshot (default: 1000,2000,3000,4000,5000)"
    echo "  --refresh:        Recompute even if artifacts already exist"
    echo "  --num-workers=N:  Max worker processes per layer (capped at nproc-1)"
    exit 1
fi

INPUT_DIR="${POSITIONAL[0]}"
SNAPSHOT_STEPS="${POSITIONAL[1]:-1000,2000,3000,4000,5000}"

# Compute effective num-workers: min(requested, nproc-1)
CPU_LIMIT=$(( $(nproc) - 1 ))
CPU_LIMIT=$(( CPU_LIMIT < 1 ? 1 : CPU_LIMIT ))
CPU_HALF=$(( $(nproc) / 2 ))
CPU_HALF=$(( CPU_HALF < 1 ? 1 : CPU_HALF ))

clamp() {
    local requested="$1"
    local default="$2"
    local cap="$3"
    local val="${requested:-$default}"
    echo $(( val < cap ? val : cap ))
}

W_EXTRACT=$(clamp "$NUM_WORKERS_ARG" 16 "$CPU_LIMIT")
W_STEP=$(clamp    "$NUM_WORKERS_ARG" 10 "$CPU_HALF")
W_GLOBAL=$(clamp  "$NUM_WORKERS_ARG" 80 "$CPU_LIMIT")
W_RENDER=$(clamp  "$NUM_WORKERS_ARG" 30 "$CPU_LIMIT")
W_VIDEO=$(clamp   "$NUM_WORKERS_ARG" 30 "$CPU_LIMIT")

echo "=== Analysis pipeline for: $INPUT_DIR ==="
echo "    CPU limit: $CPU_LIMIT  workers: extract=$W_EXTRACT step=$W_STEP global=$W_GLOBAL render=$W_RENDER video=$W_VIDEO"

echo "[1/6] extract"
uv run python -m analysis.cli extract --input-dir "$INPUT_DIR" --num-workers "$W_EXTRACT" $REFRESH

echo "[2/6] compute-step"
uv run python -m analysis.cli compute-step --input-dir "$INPUT_DIR" --num-workers "$W_STEP" $REFRESH

echo "[3/6] compute-global"
uv run python -m analysis.cli compute-global --input-dir "$INPUT_DIR" --num-workers "$W_GLOBAL" $REFRESH

echo "[4/6] render (whole_step figures)"
uv run python -m analysis.cli render --input-dir "$INPUT_DIR" --figure all --num-workers "$W_RENDER" $REFRESH

echo "[5/6] render-frames (stepwise figures)"
uv run python -m analysis.cli render-frames --input-dir "$INPUT_DIR" --figure all --steps all --num-workers "$W_RENDER" $REFRESH

echo "[6a/6] render-snapshot (steps: $SNAPSHOT_STEPS)"
uv run python -m analysis.cli render-snapshot --input-dir "$INPUT_DIR" --figure all --steps "$SNAPSHOT_STEPS" --num-workers "$W_RENDER" $REFRESH

echo "[6b/6] render-video"
uv run python -m analysis.cli render-video --input-dir "$INPUT_DIR" --figure all --steps all --num-workers "$W_VIDEO" $REFRESH

echo "=== Done ==="
