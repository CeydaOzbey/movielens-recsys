#!/usr/bin/env bash
# =============================================================================
# Upload the local MovieLens dataset to a GCS bucket
#
# Usage:
#   BUCKET=movielens-yourname bash scripts/01_upload_to_gcs.sh
# =============================================================================
set -euo pipefail

BUCKET="${BUCKET:?Set the BUCKET env var (without gs:// prefix)}"
DATA_DIR="${DATA_DIR:-data/ml-25m}"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Error: data directory '${DATA_DIR}' not found. Run 00_download_data.sh first." >&2
    exit 1
fi

echo "Creating bucket gs://${BUCKET} (if it doesn't exist)..."
if ! gsutil ls -b "gs://${BUCKET}" &> /dev/null; then
    gsutil mb -l us-central1 "gs://${BUCKET}"
fi

echo "Uploading data..."
gsutil -m cp -r "${DATA_DIR}/"* "gs://${BUCKET}/ml-25m/"

echo "Done."
gsutil ls -l "gs://${BUCKET}/ml-25m/"
