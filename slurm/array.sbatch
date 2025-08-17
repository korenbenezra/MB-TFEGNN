#!/bin/bash
#SBATCH --job-name=mbtfe
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --gres=gpu:1
#SBATCH --mem=50000
#SBATCH --constraint=geforce_rtx_3090
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

mkdir -p logs

set -euo pipefail
srun bash slurm/run.sh "$@"
