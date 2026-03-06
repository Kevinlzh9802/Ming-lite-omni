#!/bin/bash
#SBATCH --job-name="ming-lite-omni_batch_infer"
#SBATCH --partition=gpu-a100
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/scratch/zli33/slurm_outputs/ming-lite-omni/slurm_%j.out
#SBATCH --error=/scratch/zli33/slurm_outputs/ming-lite-omni/slurm_%j.err

# Batch Ming-Lite-Omni inference via Apptainer.
#
# Submit from the project folder:
#   sbatch job_scripts/inference_batch_test.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_dir="/scratch/zli33/models/Ming-lite-omni"
sif_file=/scratch/zli33/apptainers/ming-lite-omni.sif
data_root="/scratch/zli33/data/gestalt_bench/test_run_part"
result_dir="/scratch/zli33/data/gestalt_bench/results/ming-lite-omni"
output_json="${result_dir}/responses_${SLURM_JOB_ID}.json"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2
    echo "  Build it first: bash apptainer/build.sh" >&2
    exit 1
fi

if [ ! -f "$project_dir/batch_infer.py" ]; then
    echo "[ERROR] Script not found: $project_dir/batch_infer.py" >&2
    exit 1
fi

if [ ! -f "$project_dir/prompts/sample.txt" ]; then
    echo "[ERROR] Prompt file not found: $project_dir/prompts/sample.txt" >&2
    exit 1
fi

# Ensure output directories exist
mkdir -p /scratch/zli33/slurm_outputs/ming-lite-omni
mkdir -p "$result_dir"

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
echo "[INFO] sif_file   = $sif_file"
echo "[INFO] project_dir= $project_dir"
echo "[INFO] data_root  = $data_root"
echo "[INFO] output_json= $output_json"
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
  --bind "$project_dir/prompts":/prompts \
  --bind /scratch/zli33:/scratch/zli33 \
  --pwd /workspace \
  "$sif_file" \
  bash -lc '
    set -euo pipefail

    real=$(find /.singularity.d/libs -name "libcuda.so.1" 2>/dev/null | head -1)
    test -n "$real"

    mkdir -p /tmp/triton-libcuda
    ln -sf "$real" /tmp/triton-libcuda/libcuda.so
    ln -sf "$real" /tmp/triton-libcuda/libcuda.so.1

    export TRITON_LIBCUDA_PATH=/tmp/triton-libcuda
    export LD_LIBRARY_PATH=/tmp/triton-libcuda:${LD_LIBRARY_PATH:-}

    python - <<PY
from pathlib import Path
p = Path("/opt/conda/lib/python3.10/site-packages/triton/common/build.py")
txt = p.read_text()
needle = "@functools.lru_cache()\ndef libcuda_dirs():\n"
patch = """@functools.lru_cache()
def libcuda_dirs():
    env_libcuda_path = os.getenv(\"TRITON_LIBCUDA_PATH\")
    if env_libcuda_path:
        return [env_libcuda_path]
"""
if "TRITON_LIBCUDA_PATH" not in txt:
    txt = txt.replace(needle, patch)
    p.write_text(txt)
    print("Patched Triton build.py")
else:
    print("Triton build.py already patched")
PY

    python /workspace/batch_infer.py \
      --data-root "'"$data_root"'" \
      --output-json "'"$output_json"'"
  '

echo ""
echo "[INFO] Batch inference completed successfully."
echo "[INFO] Results saved to: $output_json"

