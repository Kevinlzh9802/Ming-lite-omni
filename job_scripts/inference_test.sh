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

# Triton's JIT compiler needs the unversioned "libcuda.so" to link GPU
# kernels.  Apptainer --nv injects the host driver into /.singularity.d/libs
# (which may include libcuda.so), but Triton's libcuda_dirs() searches
# ldconfig and hardcoded paths like /usr/lib64 — NOT LD_LIBRARY_PATH.
#
# Fix: bind-mount a directory containing a libcuda.so symlink (pointing to
# the --nv-injected library) over /usr/lib64 would be destructive, so
# instead we create the symlink inside the container via the bash wrapper.
apptainer exec --nv --writable-tmpfs \
    --bind "$project_dir":/workspace \
    --bind /scratch/zli33:/scratch/zli33 \
    --pwd /workspace \
    "$sif_file" \
    bash -c '
        # Create unversioned libcuda.so symlink where Triton can find it.
        # /.singularity.d/libs has the --nv injected driver libraries.
        real=$(find /.singularity.d/libs -name "libcuda.so.1" 2>/dev/null | head -1)
        if [ -n "$real" ]; then
            # /usr/lib64 is a standard path Triton checks via ldconfig/hardcode
            ln -sf "$real" /usr/lib64/libcuda.so 2>/dev/null || \
            ln -sf "$real" /usr/local/lib/libcuda.so 2>/dev/null || \
            echo "[WARN] Could not create libcuda.so symlink"
        fi
        # Triton uses `ldconfig -p` to discover libcuda directories.
        # Refresh the cache so it picks up the new symlink.
        ldconfig 2>/dev/null || true
        echo "[CONTAINER] ldconfig libcuda entries:"
        ldconfig -p 2>/dev/null | grep libcuda || true
        python "$@"
    ' _ "$test_script"

echo ""
echo "[INFO] Inference test completed successfully."
