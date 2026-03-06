#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash apptainer/run_infer.sh python test_infer.py
#   bash apptainer/run_infer.sh python test_infer_gen_image.py
#   bash apptainer/run_infer.sh python test_audio_tasks.py

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_FILE="$ROOT_DIR/apptainer/ming-lite-omni.sif"

if [ ! -f "$SIF_FILE" ]; then
    echo "SIF not found: $SIF_FILE"
    echo "Build it first: bash apptainer/build.sh"
    exit 1
fi

if [ "$#" -eq 0 ]; then
    echo "No command provided. Example:"
    echo "  bash apptainer/run_infer.sh python test_infer.py"
    exit 1
fi

apptainer exec --nv \
    --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64 \
    --bind "$ROOT_DIR":/workspace \
    --pwd /workspace \
    "$SIF_FILE" \
    "$@"
