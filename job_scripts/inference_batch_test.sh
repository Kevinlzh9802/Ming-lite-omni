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

set -euo pipefail

project_dir="/scratch/zli33/models/Ming-lite-omni"
sif_file="/scratch/zli33/apptainers/ming-lite-omni.sif"
data_root="/scratch/zli33/data/gestalt_bench/test_run_part"
result_dir="/scratch/zli33/data/gestalt_bench/results/ming-lite-omni"
output_json="${result_dir}/responses_${SLURM_JOB_ID}.json"

if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2
    exit 1
fi

if [ ! -f "$project_dir/batch_infer.py" ]; then
    echo "[ERROR] batch_infer.py not found in project dir: $project_dir" >&2
    exit 1
fi

if [ ! -f "$project_dir/prompts/sample.txt" ]; then
    echo "[ERROR] Prompt file not found: $project_dir/prompts/sample.txt" >&2
    exit 1
fi

mkdir -p /scratch/zli33/slurm_outputs/ming-lite-omni
mkdir -p "$result_dir"

echo "[INFO] sif_file   = $sif_file"
echo "[INFO] project_dir= $project_dir"
echo "[INFO] data_root  = $data_root"
echo "[INFO] output_json= $output_json"
echo ""

apptainer exec --nv --writable-tmpfs \
  --bind "$project_dir":/workspace \
  --bind "$project_dir/prompts":/prompts \
  --bind /scratch/zli33:/scratch/zli33 \
  --pwd /workspace \
  "$sif_file" \
  bash -lc "
    set -euo pipefail

    real=$(find /.singularity.d/libs -name "libcuda.so.1" 2>/dev/null | head -1)
    test -n \"\$real\"

    mkdir -p /tmp/triton-libcuda
    ln -sf \"\$real\" /tmp/triton-libcuda/libcuda.so
    ln -sf \"\$real\" /tmp/triton-libcuda/libcuda.so.1

    export TRITON_LIBCUDA_PATH=/tmp/triton-libcuda
    export LD_LIBRARY_PATH=/tmp/triton-libcuda:\${LD_LIBRARY_PATH:-}

    python /workspace/batch_infer.py \
      --data-root \"$data_root\" \
      --output-json \"$output_json\"
  "

echo ""
echo "[INFO] Batch inference completed."
echo "[INFO] Results saved to: $output_json"
