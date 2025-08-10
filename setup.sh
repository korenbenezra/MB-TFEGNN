#!/usr/bin/env bash
set -euo pipefail

echo "== setup (venv-first) =="

# --- resolve repo root ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# --- parameters ---
PYBIN="${PYBIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "• Python: ${PYBIN}"
echo "• venv  : ${VENV_DIR}"

# --- mark helper scripts executable if present ---
mkdir -p scripts slurm logs results/runs results/summary data/raw data/processed data/splits

for f in scripts/run_local.sh scripts/smoke_local.sh scripts/smoke_slurm.sh scripts/make_jobs.py; do
  if [[ -f "$f" ]]; then
    chmod +x "$f" || true
    echo "✓ chmod +x $f"
  fi
done

for f in slurm/run.sh slurm/array.sbatch; do
  if [[ -f "$f" ]]; then
    chmod +x "$f" || true
    echo "✓ chmod +x $f"
  fi
done

echo "✓ created results/, logs/, data/ folders"
