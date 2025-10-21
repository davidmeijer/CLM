#!/usr/bin/env bash
# Submit generate_compounds.py to Slurm, using CURRENT activated conda Python environment.
# Usage: sbatch_generate_compounds.sh <model-dir> <work-dir> <sample-size>
# Optional env overrides:
#   PARTITION, CPUS, MEM, TIME, GRES

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <model-dir> <work-dir> <sample-size>" >&2
  exit 1
fi

MODEL_DIR="$(readlink -f "$1")"
WORK_DIR="$(readlink -f "$2")"
SAMPLE_SIZE="$3"

# --- Slurm defaults (override via env if desired) ---
PARTITION="${PARTITION:-skinniderlab}"
CPUS="${CPUS:-8}"
MEM="${MEM:-16G}"
TIME="${TIME:-24:00:00}"
GRES="${GRES:-gpu:1}"    # set to "" for CPU-only

# --- Resolve paths & env ---
PY="$(which python)"
if [[ -z "${PY}" ]]; then
  echo "Could not find python on PATH. Make sure your conda env is activated." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/generate_compounds.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "Cannot find ${SCRIPT}. Put this wrapper next to generate_compounds.py or edit SCRIPT path." >&2
  exit 3
fi

mkdir -p "${WORK_DIR}/logs"

# --- Build sbatch args ---
SBATCH_ARGS=(
  -J clm_sample
  -p "${PARTITION}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --time="${TIME}"
  -o "${WORK_DIR}/logs/%x_%j.out"
  -e "${WORK_DIR}/logs/%x_%j.err"
)

# Optional GPU request
if [[ -n "${GRES}" ]]; then
  SBATCH_ARGS+=( --gres="${GRES}" )
fi

# --- Submit ---
echo "Submitting job:"
echo "  Python:     ${PY}"
echo "  Script:     ${SCRIPT}"
echo "  Model dir:  ${MODEL_DIR}"
echo "  Work dir:   ${WORK_DIR}"
echo "  Sample size:${SAMPLE_SIZE}"
echo "  Partition:  ${PARTITION}  CPUs: ${CPUS}  Mem: ${MEM}  Time: ${TIME}  GRES: ${GRES:-<none>}"

sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,OMP_NUM_THREADS="${CPUS}",MKL_NUM_THREADS="${CPUS}",PYTHONUNBUFFERED=1,\
PY="${PY}",SCRIPT="${SCRIPT}",MODEL_DIR="${MODEL_DIR}",WORK_DIR="${WORK_DIR}",SAMPLE_SIZE="${SAMPLE_SIZE}" \
  --wrap 'set -euo pipefail;
          echo "Node: $(hostname)";
          echo "Using Python: $PY";
          echo "CPUs: ${OMP_NUM_THREADS:-}; Mem limit: '"${MEM}"'";
          mkdir -p "$WORK_DIR";
          /usr/bin/time -v "$PY" "$SCRIPT" \
            --model-dir "$MODEL_DIR" \
            --work-dir "$WORK_DIR" \
            --sample-size "$SAMPLE_SIZE"'
