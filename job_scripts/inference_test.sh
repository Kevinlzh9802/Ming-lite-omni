#!/bin/bash
#SBATCH --job-name="ming-lite-omni_inference_test"
#SBATCH --partition=gpu-a100
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/scratch/zli33/slurm_outputs/ming-lite-omni/slurm_%j.out
#SBATCH --error=/scratch/zli33/slurm_outputs/ming-lite-omni/slurm_%j.err

# Simple Ming-Lite-Omni inference test via Apptainer.
#
# Submit from the project folder:
#   sbatch job_scripts/inference_test.sh
#   sbatch job_scripts/inference_test.sh test_infer_gen_image.py
#   sbatch job_scripts/inference_test.sh test_audio_tasks.py

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_dir="$(pwd)"
sif_file=/scratch/zli33/apptainers/ming-lite-omni.sif

# Default test script; override via first positional argument
test_script="${1:-test_infer.py}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2
    echo "  Build it first: bash apptainer/build.sh" >&2
    exit 1
fi

if [ ! -f "$project_dir/$test_script" ]; then
    echo "[ERROR] Test script not found: $project_dir/$test_script" >&2
    exit 1
fi

# Ensure slurm output directory exists
mkdir -p /scratch/zli33/slurm_outputs/ming-lite-omni

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
echo "[INFO] sif_file   = $sif_file"
echo "[INFO] project_dir= $project_dir"
echo "[INFO] test_script= $test_script"
echo ""

echo "[HOST] nvidia-smi:"
nvidia-smi || true

# Prepend the CUDA 12.1 forward-compat stubs so the container's libcuda.so
# takes priority over the (possibly older) host driver injected by --nv.
# This fixes "error 803: unsupported display driver / cuda driver combination"
# on clusters whose host driver is older than CUDA 12.1.
#
# Debug: show what the container sees
apptainer exec --nv \
    --env LD_LIBRARY_PATH=/usr/local/cuda/compat \
    "$sif_file" \
    bash -c 'echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"; ls -la /usr/local/cuda/compat/ 2>/dev/null || echo "no compat dir"; nvidia-smi; python -c "import torch; print(\"CUDA available:\", torch.cuda.is_available()); print(\"Device count:\", torch.cuda.device_count())"'

apptainer exec --nv \
    --env LD_LIBRARY_PATH=/usr/local/cuda/compat \
    --bind "$project_dir":/workspace \
    --bind /scratch/zli33:/scratch/zli33 \
    --pwd /workspace \
    "$sif_file" \
    python "$test_script"

echo ""
echo "[INFO] Inference test completed successfully."
