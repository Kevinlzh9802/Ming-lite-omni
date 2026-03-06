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

# Triton's JIT compiler needs the unversioned "libcuda.so" symlink to build
# GPU kernels.  Apptainer --nv injects the host driver as libcuda.so.1 into
# /.singularity.d/libs but does NOT create the unversioned symlink.
# We create it in a writable tmpdir at runtime and prepend that to
# LD_LIBRARY_PATH, leaving the --nv-injected paths intact.
apptainer exec --nv \
    --bind "$project_dir":/workspace \
    --bind /scratch/zli33:/scratch/zli33 \
    --pwd /workspace \
    "$sif_file" \
    bash -c '
        cuda_stub_dir=$(mktemp -d)
        # Find the real libcuda.so.1 that --nv injected
        real_libcuda=$(ldconfig -p 2>/dev/null | grep "libcuda.so.1 " | head -1 | sed "s/.*=> //")
        if [ -z "$real_libcuda" ]; then
            # fallback: search /.singularity.d/libs directly
            real_libcuda=$(find /.singularity.d/libs -name "libcuda.so.1" 2>/dev/null | head -1)
        fi
        if [ -n "$real_libcuda" ]; then
            ln -s "$real_libcuda" "$cuda_stub_dir/libcuda.so"
            echo "[CONTAINER] Created libcuda.so symlink -> $real_libcuda in $cuda_stub_dir"
        else
            echo "[CONTAINER][WARN] Could not find libcuda.so.1"
        fi
        export LD_LIBRARY_PATH="$cuda_stub_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "[CONTAINER] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
        python "$@"
        rm -rf "$cuda_stub_dir"
    ' _ "$test_script"

echo ""
echo "[INFO] Inference test completed successfully."
