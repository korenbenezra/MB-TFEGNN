#!/bin/bash
#SBATCH --job-name=mbtfe
#SBATCH --partition=studentbatch          # TAU CS: 3-day batch partition
#SBATCH --array=0-9%6                     # EDIT: 0-(N-1); cap concurrency at 6 (student limit)
#SBATCH --time=2-00:00:00                 # ≤ 3 days for studentbatch
#SBATCH --gres=gpu:1                      # remove if CPU-only
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --constraint="geforce_rtx_3090"   # optional: request specific GPU models
#SBATCH --account=ronmondshein


# ensure logs dir exists
mkdir -p logs

# (optional) activate your venv
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# keep repo on PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# delegate to run.sh with the array index
bash slurm/run.sh "$SLURM_ARRAY_TASK_ID"
