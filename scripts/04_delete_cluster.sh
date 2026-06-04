#!/usr/bin/env bash
# =============================================================================
# Delete the Dataproc cluster to stop billing.
#
# IMPORTANT: ALWAYS run this after submit_job.sh completes! The cluster
# accrues charges by the second while running.
#
# Usage:
#   bash scripts/04_delete_cluster.sh
# =============================================================================
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-movielens-cluster}"
REGION="${REGION:-us-central1}"

echo "Deleting cluster '${CLUSTER_NAME}' in region '${REGION}'..."
gcloud dataproc clusters delete "${CLUSTER_NAME}" \
    --region="${REGION}" \
    --quiet

echo "Cluster deleted. Billing has stopped."
echo ""
echo "Persistent state (data, models, results) is preserved in GCS."
