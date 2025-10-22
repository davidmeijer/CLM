#!/usr/bin/env bash
# Submit extract_retromol_fingerprints.py to Slurm, using CURRENT activated conda Python environment.
# Usage: submit_parse_retromol_results.sh <results.jsonl> <outdir>
#
# Optional env overrides (with sensible defaults):
#   PARTITION, CPUS, MEM, TIME, GRES
#   COV_THRESH, NUM_BITS, COUNTED
#
# Examples:
#   PARTITION=compute CPUS=16 MEM=32G TIME=08:00:00 \
#   COV_THRESH=0.9 NUM_BITS=1024 COUNTED=1 \
#   ./submit_parse_retromol_results.sh data/results.jsonl work/fps
#
# Notes:
# - COUNTED=1 adds --counted; anything else omits it.
# - Make sure your conda env with retromol/rdkit is ACTIVATED before running.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <results.jsonl> <outdir>" >&2
  exit 1
fi

RESULTS="$(readlink -f "$1")"
OUTDIR="$(readlink -f "$2")"

# --- Slurm defaults (override via env if desired) ---
PARTITION="${PARTITION:-skinniderlab}"
CPUS="${CPUS:-8}"
MEM="${MEM:-16G}"
TIME="${TIME:-12:00:00}"
GRES="${GRES:-}"               # default CPU-only. Example for GPU (not needed here): gpu:1

# --- Fingerprint defaults ---
COV_THRESH="${COV_THRESH:-1.0}"
NUM_BITS="${NUM_BITS:-512}"
COUNTED="${COUNTED:-0}"        # 1 -> add --counted
if [[ "${COUNTED}" == "1" || "${COUNTED,,}" == "true" ]]; then
  COUNTED_FLAG="--counted"
else
  COUNTED_FLAG=""
fi

# --- Resolve paths & env ---
PY="$(which python || true)"
if [[ -z "${PY}" ]]; then
  echo "Could not find python on PATH. Make sure your conda env is activated." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Adjust filename below if yours is named differently:
SCRIPT="${SCRIPT_DIR}/extract_retromol_fingerprints.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "Cannot find ${SCRIPT}. Put this wrapper next to your Python script or edit SCRIPT path." >&2
  exit 3
fi

mkdir -p "${OUTDIR}/logs"

# --- Build sbatch args ---
SBATCH_ARGS=(
  -J retromol_fp
  -p "${PARTITION}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --time="${TIME}"
  -o "${OUTDIR}/logs/%x_%j.out"
  -e "${OUTDIR}/logs/%x_%j.err"
)

# Optional GPU request (not needed; left here for symmetry)
if [[ -n "${GRES}" ]]; then
  SBATCH_ARGS+=( --gres="${GRES}" )
fi

# --- Submit ---
echo "Submitting job:"
echo "  Python:      ${PY}"
echo "  Script:      ${SCRIPT}"
echo "  Results:     ${RESULTS}"
echo "  Outdir:      ${OUTDIR}"
echo "  Cov-thresh:  ${COV_THRESH}"
echo "  Num-bits:    ${NUM_BITS}"
echo "  Counted:     ${COUNTED}"
echo "  Partition:   ${PARTITION}  CPUs: ${CPUS}  Mem: ${MEM}  Time: ${TIME}  GRES: ${GRES:-<none>}"

sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,OMP_NUM_THREADS="${CPUS}",MKL_NUM_THREADS="${CPUS}",PYTHONUNBUFFERED=1,\
PY="${PY}",SCRIPT="${SCRIPT}",RESULTS="${RESULTS}",OUTDIR="${OUTDIR}",\
COV_THRESH="${COV_THRESH}",NUM_BITS="${NUM_BITS}",COUNTED_FLAG="${COUNTED_FLAG}" \
  --wrap 'set -euo pipefail;
          echo "Node: $(hostname)";
          echo "Using Python: $PY";
          echo "CPUs: ${OMP_NUM_THREADS:-}  Mem limit: '"${MEM}"'";
          mkdir -p "$OUTDIR";
          /usr/bin/time -v "$PY" "$SCRIPT" \
            --results "$RESULTS" \
            --outdir "$OUTDIR" \
            --cov-thresh "$COV_THRESH" \
            --num-bits "$NUM_BITS" \
            ${COUNTED_FLAG}'
