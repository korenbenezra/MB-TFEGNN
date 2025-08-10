#!/usr/bin/env bash
# scripts/run_local.sh
# Run one experiment locally by index into scripts/jobs.tsv
# Usage: bash scripts/run_local.sh [JOB_INDEX] [EXTRA_ARGS...]
# Example: bash scripts/run_local.sh 0 --device cpu

set -euo pipefail

# Resolve repo root (this script lives in repo/scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Config
JOBS_TSV="${JOBS_TSV:-${REPO_ROOT}/scripts/jobs.tsv}"

# Args
JOB_INDEX="${1:-0}"
shift || true  # remove JOB_INDEX if present; any remaining args are extra and optional

# Checks
if [[ ! -f "${JOBS_TSV}" ]]; then
  echo "[run_local] jobs.tsv not found at: ${JOBS_TSV}" >&2
  echo "            Generate it with: python scripts/make_jobs.py ..." >&2
  exit 1
fi

if ! [[ "${JOB_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[run_local] JOB_INDEX must be a non-negative integer (got '${JOB_INDEX}')" >&2
  exit 1
fi

echo "[run_local] Using jobs file : ${JOBS_TSV}"
echo "[run_local] Running job idx : ${JOB_INDEX}"
echo "[run_local] Extra args      : $*"

# Execute
python -m src.main \
  --from-tsv "${JOBS_TSV}" \
  --index "${JOB_INDEX}" \
  --outdir "${REPO_ROOT}/results/runs" \
  "$@"
