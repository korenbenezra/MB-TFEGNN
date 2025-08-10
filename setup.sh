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

# --- create or reuse venv ---
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "→ creating virtualenv at ${VENV_DIR}"
  "${PYBIN}" -m venv "${VENV_DIR}"
else
  echo "• reusing existing venv at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -V
pip -V

# --- install requirements into the venv (if present) ---
if [[ -f requirements.txt ]]; then
  echo "→ installing requirements.txt into venv..."
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo "✓ requirements installed"
else
  echo "• no requirements.txt found — skipping install"
fi

echo "== setup done =="
echo "next steps:"
echo "  1) activate venv in this shell:  source ${VENV_DIR}/bin/activate"
echo "  2) quick local smoke test:       bash scripts/smoke_local.sh   (optional)"
echo "  3) build a jobs grid:            python scripts/make_jobs.py --datasets chameleon --seeds 1,2 --K 8 --bands 3 --tau 0.5,1.5,4.0 --out scripts/jobs.tsv"
echo "  4) run one job:                  bash scripts/run_local.sh 0"
echo "  5) submit SLURM array:           sbatch slurm/array.sbatch   (edit --array range first)"
