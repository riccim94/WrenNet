#!/bin/bash
set -euo pipefail

IMAGE_NAME="bird_classification_edge"
CONTAINER_NAME_DEFAULT="export_onnx_container"
EXPORT_SCRIPT_REL="birds_distillation_edge/convert-and-quantize/streamable-kws/export_onnx.py"

usage() {
    cat <<'EOF'
Usage:
  ./run_docker_export_onnx.sh [container_name] <checkpoint_path> <output_path> [hydra overrides...]

Examples:
  ./run_docker_export_onnx.sh my_export \
      logs/lightning/birds_distillation_edge/version_7/checkpoints/epoch=0-step=5.ckpt \
      exports/birds_full.onnx

  ./run_docker_export_onnx.sh export_ckpt \
      logs/lightning/birds_distillation_edge/version_7/checkpoints/epoch=0-step=5.ckpt \
      exports/birds_full.onnx \
      model.params.num_classes=9
EOF
}

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

REPO_ROOT=$(pwd)
CONTAINER_NAME="$CONTAINER_NAME_DEFAULT"

# Determine if a custom container name was provided (if the first argument doesn't look like a path)
if [[ "$1" != /* && "$1" != .* && "$1" != ../* && ! "$1" =~ ^[A-Za-z0-9._-]+/.+ ]]; then
    CONTAINER_NAME="$1"
    shift
fi

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

CHECKPOINT_PATH_INPUT="$1"
OUTPUT_PATH_INPUT="$2"
shift 2
HYDRA_OVERRIDES=("$@")

abs_path() {
    python - "$1" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

CHECKPOINT_ABS=$(abs_path "$CHECKPOINT_PATH_INPUT")
if [ ! -f "$CHECKPOINT_ABS" ]; then
    echo "ERRORE: checkpoint non trovato in $CHECKPOINT_ABS"
    exit 1
fi

OUTPUT_ABS=$(abs_path "$OUTPUT_PATH_INPUT")
mkdir -p "$(dirname "$OUTPUT_ABS")"

case "$CHECKPOINT_ABS" in
    "$REPO_ROOT"/*) ;;
    *)
        echo "ERRORE: il checkpoint deve trovarsi dentro la repo ($REPO_ROOT)."
        exit 1
        ;;
esac

case "$OUTPUT_ABS" in
    "$REPO_ROOT"/*) ;;
    *)
        echo "ERRORE: il percorso di output deve trovarsi dentro la repo ($REPO_ROOT)."
        exit 1
        ;;
esac

if [ ! -f "$EXPORT_SCRIPT_REL" ]; then
    echo "ERRORE: script di export non trovato in $EXPORT_SCRIPT_REL"
    exit 1
fi

REL_CHECKPOINT="${CHECKPOINT_ABS#$REPO_ROOT/}"
REL_OUTPUT="${OUTPUT_ABS#$REPO_ROOT/}"
CONTAINER_CHECKPOINT="/workspace/$REL_CHECKPOINT"
CONTAINER_OUTPUT="/workspace/$REL_OUTPUT"
CONTAINER_EXPORT_SCRIPT="/workspace/$EXPORT_SCRIPT_REL"

echo "--- Export ONNX via Docker ---"
echo "Checkpoint: $CHECKPOINT_ABS"
echo "Output ONNX: $OUTPUT_ABS"
# shellcheck disable=SC2128
if [ "${#HYDRA_OVERRIDES[@]}" -eq 0 ]; then
    echo "Hydra overrides: Nessuno"
else
    echo "Hydra overrides: ${HYDRA_OVERRIDES[*]}"
fi
echo "--------------------------------"

docker run --rm \
    --name "$CONTAINER_NAME" \
    -v "$REPO_ROOT":/workspace \
    -w /workspace \
    "$IMAGE_NAME" \
    python "$CONTAINER_EXPORT_SCRIPT" \
        --checkpoint "$CONTAINER_CHECKPOINT" \
        --output "$CONTAINER_OUTPUT" \
        --input-format mel \
        "${HYDRA_OVERRIDES[@]}"

echo "--- Export completato (ONNX salvato in $OUTPUT_ABS) ---"
