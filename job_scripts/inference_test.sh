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
project_dir="/scratch/zli33/models/Ming-lite-omni"
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

# The host driver (580.x, CUDA 13.0) is newer than what the container's
# CUDA 12.1 runtime requires (>=530.30.02).  Apptainer --nv injects the
# host's libcuda.so.1 into /.singularity.d/libs which is correct.
#
# The container also ships old compat stubs (530.x) in /usr/local/cuda/compat.
# That directory must come AFTER /.singularity.d/libs so the dynamic linker
# picks up the host's newer libcuda.so.1 at runtime.  However it must still
# be present because Triton's JIT compiler searches LD_LIBRARY_PATH for the
# unversioned "libcuda.so" symlink, which only exists in /usr/local/cuda/compat.
apptainer exec --nv \
    --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64:/usr/local/cuda/compat \
    --bind "$project_dir":/workspace \
    --bind /scratch/zli33:/scratch/zli33 \
    --pwd /workspace \
    "$sif_file" \
    python "$test_script"

echo ""
echo "[INFO] Inference test completed successfully."
