#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (this script lives in slurm/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOBS_TSV="${JOBS_TSV:-${REPO_ROOT}/scripts/jobs.tsv}"
INDEX="${1:?usage: run.sh <JOB_INDEX>}"

if [[ ! -f "${JOBS_TSV}" ]]; then
  echo "[run.sh] jobs.tsv not found at: ${JOBS_TSV}" >&2
  exit 1
fi

echo "[run.sh] JOBS_TSV=${JOBS_TSV}"
echo "[run.sh] INDEX=${INDEX}"

# Add repo to PYTHONPATH in case sbatch launches from a different CWD
export PYTHONPATH="$PYTHONPATH:${REPO_ROOT}"

python3 -m src.main \
  --from-tsv "${JOBS_TSV}" \
  --index "${INDEX}" \
  --outdir "${REPO_ROOT}/results/runs"
