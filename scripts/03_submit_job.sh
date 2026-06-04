#!/usr/bin/env bash
# =============================================================================
# Submit the PySpark pipeline to the running Dataproc cluster.
#
# Prerequisite: cluster must exist (run 02_create_cluster.sh first).
#
# Usage:
#   BUCKET=movielens-yourname bash scripts/03_submit_job.sh
# =============================================================================
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-movielens-cluster}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:?Set the BUCKET env var (without gs:// prefix)}"

echo "Submitting PySpark job to cluster: ${CLUSTER_NAME}"
echo "  Bucket:  gs://${BUCKET}"
echo ""

# We pass the bucket as an environment variable to the PySpark job
gcloud dataproc jobs submit pyspark scripts/run_dataproc.py \
    --cluster="${CLUSTER_NAME}" \
    --region="${REGION}" \
    --properties="spark.executorEnv.BUCKET=${BUCKET},spark.yarn.appMasterEnv.BUCKET=${BUCKET}" \
    -- --bucket "gs://${BUCKET}"

echo ""
echo "Job complete. Don't forget to delete the cluster (04_delete_cluster.sh)."
