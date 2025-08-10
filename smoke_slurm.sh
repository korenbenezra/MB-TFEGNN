#!/usr/bin/env bash
# scripts/smoke_slurm.sh
# Build a 2-row jobs.tsv (MB-TFE + TFE on Cora) and submit a small SLURM array

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p scripts logs results/runs

# 1) make tiny job sets (two files) and then combine them
python3 MB-TFEGNN/scripts/make_jobs.py \
  --datasets cora \
  --model mbtfe \
  --seeds 1 \
  --K 5 \
  --bands 3 \
  --tau 0.5,1.5,4.0 \
  --hidden 64 --layers 1 --dropout 0.5 \
  --lr 1e-3 --wd 5e-5 \
  --epochs 50 --patience 10 \
  --out MB-TFEGNN/scripts/jobs_mbtfe.tsv

python3 MB-TFEGNN/scripts/make_jobs.py \
  --datasets cora \
  --model tfe \
  --seeds 1 \
  --K 5 \
  --bands 2 \
  --tau 0.5,1.5 \
  --hidden 64 --layers 1 --dropout 0.5 \
  --lr 1e-3 --wd 5e-5 \
  --epochs 50 --patience 10 \
  --out MB-TFEGNN/scripts/jobs_tfe.tsv

# Combine (keep header from the first, append rows from the second)
cp MB-TFEGNN/scripts/jobs_mbtfe.tsv MB-TFEGNN/scripts/jobs.tsv
tail -n +2 MB-TFEGNN/scripts/jobs_tfe.tsv >> MB-TFEGNN/scripts/jobs.tsv

# 2) submit a tiny array job (index 0-1, concurrency 2)
echo "Submitting SLURM array for 2 smoke jobs..."
sbatch --array=0-1%2 MB-TFEGNN/slurm/array.sbatch
echo "Submitted. Check with: squeue -u $USER"
