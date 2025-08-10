#!/bin/bash
#SBATCH --job-name=mbtfe
#SBATCH --account=gpu-research
#SBATCH --partition=killable          # TAU CS: 3-day batch partition
#SBATCH --gres=gpu:1                      # remove if CPU-only
#SBATCH --mem=50000
#SBATCH --constraint=geforce_rtx_3090
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err


mkdir -p logs

export PYTHONPATH="$PYTHONPATH:$(pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate mbtfe

export PYTHONPATH="$PYTHONPATH:$(pwd)"

bash slurm/run.sh "$SLURM_ARRAY_TASK_ID"