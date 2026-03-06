#!/bin/bash
#SBATCH --job-name="ming-lite-omni_inference_test"
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M
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
apptainer exec --nv --writable-tmpfs "$sif_file" bash -lc '
python - <<PY
import triton, inspect
import triton.common.build as b
print("triton version:", triton.__version__)
print("build.py:", inspect.getsourcefile(b))
PY
real=$(find /.singularity.d/libs -name "libcuda.so.1" | head -1)
mkdir -p /usr/local/libcuda-workaround
ln -sf "$real" /usr/local/libcuda-workaround/libcuda.so
echo /usr/local/libcuda-workaround >/etc/ld.so.conf.d/libcuda-workaround.conf
ldconfig
ldconfig -p | grep libcuda || true
'
