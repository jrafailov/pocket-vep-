#!/usr/bin/env bash
# End-to-end pocket-vep pipeline:
#   ClinVar download -> structure cache -> conservation cache ->
#   feature matrix -> experiments -> EDA + result plots.
#
# Each stage skips if its sentinel output exists (use --force to re-run).
# Hardcoded data paths inside the Python scripts are accommodated by
# symlinking <repo>/data -> $DATA_DIR and $DATA_DIR/raw/alphafold -> $ALPHAFOLD_DIR.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_pipeline.sh --alphafold-dir <path> [options]

Required:
  --alphafold-dir <path>   Directory containing AF-{uniprot}-F1-model_v4.pdb.gz
                           (or set ALPHAFOLD_DIR env var)

Options:
  --data-dir <path>        Data process root (default: <repo>/data)
                           Will be symlinked as <repo>/data so the underlying
                           Python scripts find their hardcoded paths.
  --output-dir <path>      Results / EDA / graphs root (default: <repo>/results)
  --n-jobs N               Parallel workers for structure features
                           (default: detected CPU count - 1)
  --seeds "42 43 44 ..."   Space-separated seed list for run_experiments.py
                           (default: "42 43 44 45 46")
  --force                  Re-run every stage even if outputs exist
  -h, --help               Show this help and exit

Env-var equivalents (CLI takes precedence): ALPHAFOLD_DIR, DATA_DIR,
OUTPUT_DIR, N_JOBS, SEEDS, FORCE=1.
EOF
}

ALPHAFOLD_DIR="${ALPHAFOLD_DIR:-}"
DATA_DIR="${DATA_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
N_JOBS="${N_JOBS:-}"
SEEDS="${SEEDS:-42 43 44 45 46}"
FORCE="${FORCE:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --alphafold-dir) ALPHAFOLD_DIR="$2"; shift 2 ;;
        --data-dir)      DATA_DIR="$2";      shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --n-jobs)        N_JOBS="$2";        shift 2 ;;
        --seeds)         SEEDS="$2";         shift 2 ;;
        --force)         FORCE=1;            shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$ALPHAFOLD_DIR" ]]; then
    echo "ERROR: --alphafold-dir is required (or set ALPHAFOLD_DIR)" >&2
    usage >&2
    exit 2
fi
if [[ ! -d "$ALPHAFOLD_DIR" ]]; then
    echo "ERROR: AlphaFold directory does not exist: $ALPHAFOLD_DIR" >&2
    exit 2
fi
ALPHAFOLD_DIR="$(cd "$ALPHAFOLD_DIR" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

[[ -z "$DATA_DIR" ]]   && DATA_DIR="$PROJECT_ROOT/data"
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="$PROJECT_ROOT/results"

if [[ -z "$N_JOBS" ]]; then
    cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null \
            || sysctl -n hw.logicalcpu 2>/dev/null \
            || echo 2)"
    N_JOBS=$(( cpus - 1 ))
    [[ "$N_JOBS" -lt 1 ]] && N_JOBS=1
fi

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

banner() {
    echo
    echo "=================================================="
    echo "  $*"
    echo "=================================================="
}

need_run() {
    local sentinel="$1"
    if [[ "$FORCE" == "1" ]]; then return 0; fi
    if [[ ! -e "$sentinel" ]]; then return 0; fi
    if [[ -d "$sentinel" && -z "$(ls -A "$sentinel" 2>/dev/null)" ]]; then return 0; fi
    return 1
}

skip_msg() { echo "[skip] $1 already present (use --force to re-run)"; }

banner "Stage 0: directory setup"
echo "  project root : $PROJECT_ROOT"
echo "  data dir     : $DATA_DIR"
echo "  output dir   : $OUTPUT_DIR"
echo "  alphafold dir: $ALPHAFOLD_DIR"
echo "  n-jobs       : $N_JOBS"
echo "  seeds        : $SEEDS"
echo "  force        : $FORCE"

mkdir -p "$DATA_DIR/raw" "$DATA_DIR/interim" "$DATA_DIR/processed"

if [[ "$DATA_DIR" != "$PROJECT_ROOT/data" ]]; then
    if [[ -e "$PROJECT_ROOT/data" && ! -L "$PROJECT_ROOT/data" ]]; then
        echo "ERROR: $PROJECT_ROOT/data exists and is not a symlink." >&2
        echo "       Refusing to overwrite. Move/rename it, or pass" >&2
        echo "       --data-dir $PROJECT_ROOT/data to use it directly." >&2
        exit 1
    fi
    echo "[setup] linking $PROJECT_ROOT/data -> $DATA_DIR"
    ln -sfn "$DATA_DIR" "$PROJECT_ROOT/data"
fi

AF_LINK="$DATA_DIR/raw/alphafold"
if [[ "$ALPHAFOLD_DIR" != "$AF_LINK" ]]; then
    if [[ -d "$AF_LINK" && ! -L "$AF_LINK" ]]; then
        if [[ -z "$(ls -A "$AF_LINK" 2>/dev/null)" ]]; then
            rmdir "$AF_LINK"
        else
            echo "ERROR: $AF_LINK exists, is not a symlink, and is non-empty." >&2
            echo "       Move/remove it first, or pass --alphafold-dir $AF_LINK" >&2
            exit 1
        fi
    fi
    echo "[setup] linking $AF_LINK -> $ALPHAFOLD_DIR"
    ln -sfn "$ALPHAFOLD_DIR" "$AF_LINK"
fi

CLINVAR_PARQ="$DATA_DIR/interim/clinvar_labeled.parquet"
UNIPROT_PARQ="$DATA_DIR/interim/uniprot_mapping.parquet"
PLDDT_PARQ="$DATA_DIR/processed/plddt_cache.parquet"
STRUCT_PARQ="$DATA_DIR/processed/structure_features.parquet"
CONSERV_PARQ="$DATA_DIR/processed/conservation_cache.parquet"
MATRIX_PARQ="$DATA_DIR/processed/feature_matrix.parquet"
EXPERIMENTS_CSV="$OUTPUT_DIR/experiments.csv"
EDA_DIR="$OUTPUT_DIR/eda"
GRAPHS_DIR="$OUTPUT_DIR/analysis_graphs"

banner "Stage 1: download + label ClinVar"
if need_run "$CLINVAR_PARQ"; then
    python scripts/download_clinvar.py \
        --raw-dir data/raw \
        --out data/interim/clinvar_labeled.parquet
else
    skip_msg "$CLINVAR_PARQ"
fi

banner "Stage 2: gene -> UniProt mapping"
if need_run "$UNIPROT_PARQ"; then
    python scripts/build_structure_cache.py --stage map
else
    skip_msg "$UNIPROT_PARQ"
fi

banner "Stage 3: AlphaFold availability filter"
# Fast no-op when every PDB is already present in $ALPHAFOLD_DIR; the stage
# still has to run so downstream stages get an in-memory mapping table.
python scripts/build_structure_cache.py --stage download

banner "Stage 4: pLDDT cache"
if need_run "$PLDDT_PARQ"; then
    python scripts/build_structure_cache.py --stage plddt
else
    skip_msg "$PLDDT_PARQ"
fi

banner "Stage 5: fpocket + DSSP structure features"
if need_run "$STRUCT_PARQ"; then
    python scripts/build_structure_cache.py --stage features --n-jobs "$N_JOBS"
else
    skip_msg "$STRUCT_PARQ"
fi

banner "Stage 6: phyloP + phastCons conservation cache"
if need_run "$CONSERV_PARQ"; then
    python scripts/build_conservation_cache.py
else
    skip_msg "$CONSERV_PARQ"
fi

banner "Stage 7: feature matrix"
if need_run "$MATRIX_PARQ"; then
    python scripts/build_feature_matrix.py \
        --data data/interim/clinvar_labeled.parquet \
        --out data/processed/feature_matrix.parquet
else
    skip_msg "$MATRIX_PARQ"
fi

banner "Stage 8: train + evaluate (seeds=$SEEDS)"
if need_run "$EXPERIMENTS_CSV"; then
    # SEEDS is a deliberate space-separated list passed as multiple argv tokens.
    # shellcheck disable=SC2086
    python scripts/run_experiments.py \
        --feature-matrix data/processed/feature_matrix.parquet \
        --out-dir "$OUTPUT_DIR" \
        --seeds $SEEDS
else
    skip_msg "$EXPERIMENTS_CSV"
fi

banner "Stage 9: EDA plots"
if need_run "$EDA_DIR"; then
    python scripts/visualize_features.py \
        --matrix data/processed/feature_matrix.parquet \
        --out "$EDA_DIR"
else
    skip_msg "$EDA_DIR"
fi

banner "Stage 10: results plots"
if need_run "$GRAPHS_DIR"; then
    python scripts/visualize_results.py \
        --csv "$EXPERIMENTS_CSV" \
        --out "$GRAPHS_DIR"
else
    skip_msg "$GRAPHS_DIR"
fi

banner "Pipeline complete"
echo "Data root   : $DATA_DIR"
echo "Output root : $OUTPUT_DIR"
echo "  experiments.csv         : $EXPERIMENTS_CSV"
echo "  experiments_summary.csv : $OUTPUT_DIR/experiments_summary.csv"
echo "  interpretations/        : $OUTPUT_DIR/interpretations/"
echo "  EDA plots               : $EDA_DIR/"
echo "  results plots           : $GRAPHS_DIR/"
