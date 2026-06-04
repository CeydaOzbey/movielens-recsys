#!/usr/bin/env bash
# =============================================================================
# Download MovieLens dataset
#
# Usage:
#   bash scripts/00_download_data.sh         # small (default, 100K ratings)
#   bash scripts/00_download_data.sh 25m     # full 25M dataset
# =============================================================================
set -euo pipefail

SIZE="${1:-small}"
DATA_DIR="data"
mkdir -p "${DATA_DIR}"

if [[ "${SIZE}" == "small" ]]; then
    URL="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    TARGET_DIR="ml-latest-small"
elif [[ "${SIZE}" == "25m" ]]; then
    URL="https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    TARGET_DIR="ml-25m"
else
    echo "Error: unknown size '${SIZE}'. Use 'small' or '25m'." >&2
    exit 1
fi

ZIP_PATH="${DATA_DIR}/${TARGET_DIR}.zip"
EXTRACTED_DIR="${DATA_DIR}/${TARGET_DIR}"

if [[ -d "${EXTRACTED_DIR}" ]]; then
    echo "Dataset already exists at ${EXTRACTED_DIR}. Skipping download."
    exit 0
fi

echo "Downloading MovieLens ${SIZE} dataset..."
curl -L -o "${ZIP_PATH}" "${URL}"

echo "Extracting..."
unzip -q "${ZIP_PATH}" -d "${DATA_DIR}"
rm "${ZIP_PATH}"

echo "Done. Dataset at: ${EXTRACTED_DIR}"
ls -lh "${EXTRACTED_DIR}"
