#!/usr/bin/env bash
# Submit retromol batch job to Slurm, using CURRENT shell environment.
# Usage: sbatch_retromol.sh <output-dir> <table.csv>
# Optional env overrides:
#   PARTITION, CPUS, MEM, TIME, GRES
#   SEP (default: comma), ID_COL (default: identifier), SMILES_COL (default: canonical_smiles)
#   WORKERS (default: = CPUS)

# NOTE: Make sure to have installed RetroMol first!

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <output-dir> <table.csv>" >&2
  exit 1
fi

OUT_DIR="$(readlink -f "$1")"
TABLE_PATH="$(readlink -f "$2")"

# --- Slurm defaults (override via env if desired) ---
PARTITION="${PARTITION:-skinniderlab}"
CPUS="${CPUS:-16}"
MEM="${MEM:-16G}"
TIME="${TIME:-24:00:00}"
# retromol is CPU-bound; leave GRES empty by default. Set GRES=gpu:1 if you really need a GPU.
GRES="${GRES:-}"

# --- retromol defaults (override via env if desired) ---
SEP="${SEP:-comma}"                   # e.g., comma | tab | semicolon
ID_COL="${ID_COL:-identifier}"
SMILES_COL="${SMILES_COL:-canonical_smiles}"
WORKERS="${WORKERS:-$CPUS}"

# --- Resolve retromol binary ---
RETROMOL_BIN="$(command -v retromol || true)"
if [[ -z "${RETROMOL_BIN}" ]]; then
  echo "Could not find 'retromol' on PATH. Activate the right env or module first." >&2
  exit 2
fi

# --- Basic input checks ---
if [[ ! -f "$TABLE_PATH" ]]; then
  echo "Input table not found: $TABLE_PATH" >&2
  exit 3
fi

mkdir -p "${OUT_DIR}/logs"

# --- Build sbatch args ---
SBATCH_ARGS=(
  -J retromol_batch
  -p "${PARTITION}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --time="${TIME}"
  -o "${OUT_DIR}/logs/%x_%j.out"
  -e "${OUT_DIR}/logs/%x_%j.err"
)

# Optional GPU request
if [[ -n "${GRES}" ]]; then
  SBATCH_ARGS+=( --gres="${GRES}" )
fi

echo "Submitting job:"
echo "  Retromol:   ${RETROMOL_BIN}"
echo "  Output dir: ${OUT_DIR}"
echo "  Table:      ${TABLE_PATH}"
echo "  Sep:        ${SEP}  ID col: ${ID_COL}  SMILES col: ${SMILES_COL}"
echo "  Workers:    ${WORKERS}"
echo "  Partition:  ${PARTITION}  CPUs: ${CPUS}  Mem: ${MEM}  Time: ${TIME}  GRES: ${GRES:-<none>}"

sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,RETROMOL_BIN="${RETROMOL_BIN}",\
OUT_DIR="${OUT_DIR}",TABLE_PATH="${TABLE_PATH}",SEP="${SEP}",ID_COL="${ID_COL}",SMILES_COL="${SMILES_COL}",WORKERS="${WORKERS}" \
  --wrap 'set -euo pipefail;
          echo "Node: $(hostname)";
          echo "Retromol: $RETROMOL_BIN";
          echo "CPUs requested: '"${CPUS}"'  Mem limit: '"${MEM}"'";
          mkdir -p "$OUT_DIR";
          /usr/bin/time -v "$RETROMOL_BIN" \
             -o "$OUT_DIR" \
             batch \
             --table "$TABLE_PATH" \
             --separator "$SEP" \
             --id-col "$ID_COL" \
             --smiles-col "$SMILES_COL" \
             --workers "$WORKERS"'
