#!/usr/bin/env bash
# setup.sh — minimal project setup

set -euo pipefail

echo "== minimal setup =="

# 1) make helper scripts executable (only if they exist)
[[ -f scripts/run_local.sh ]] && chmod +x scripts/run_local.sh && echo "✓ chmod +x scripts/run_local.sh"
[[ -f slurm/run.sh ]]         && chmod +x slurm/run.sh         && echo "✓ chmod +x slurm/run.sh"
[[ -f slurm/array.sbatch ]]   && chmod +x slurm/array.sbatch   && echo "✓ chmod +x slurm/array.sbatch"

# 2) create the few folders we write to
mkdir -p results/runs results/summary logs data/raw data/processed data/splits
echo "✓ created results/, logs/, data/"

# 3) (optional) install requirements into the CURRENT Python env
if [[ -f requirements.txt ]]; then
  echo "→ installing requirements.txt into current Python environment..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo "✓ requirements installed"
else
  echo "• no requirements.txt found — skipping install"
fi

# 4) small sanity note about the jobs grid
[[ ! -file scripts/make_jobs.py ]] && echo "• WARNING: scripts/make_jobs.py not found — generate scripts/jobs.tsv manually" || true

echo "== done =="
echo "next steps:"
echo "  1) generate jobs:   python scripts/make_jobs.py --datasets chameleon --seeds 1,2 --K 8 --bands 3 --tau 0.5,1.5,4.0"
echo "  2) run one job:     bash scripts/run_local.sh 0"
echo "  3) submit on SLURM: sbatch slurm/array.sbatch   (edit --array range first)"
