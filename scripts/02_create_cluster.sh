#!/usr/bin/env bash
# =============================================================================
# Provision the GCP Dataproc cluster used in the experiments.
#
# This is the EXACT spec used for the final 25M run reported in the paper.
# Cluster cost: approximately $0.30 / hour (US$). Free trial covers it.
#
# Usage:
#   bash scripts/02_create_cluster.sh
# =============================================================================
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-movielens-cluster}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"

echo "================================================="
echo "Creating Dataproc cluster:"
echo "  Name:    ${CLUSTER_NAME}"
echo "  Region:  ${REGION}"
echo "  Zone:    ${ZONE}"
echo "  Master:  n1-standard-2"
echo "  Workers: 2 x n1-standard-2"
echo "================================================="
echo ""
echo "Note: this provisions a billable resource."
echo "Use scripts/04_delete_cluster.sh to tear it down when done."
echo ""

gcloud dataproc clusters create "${CLUSTER_NAME}" \
    --region="${REGION}" \
    --zone="${ZONE}" \
    --master-machine-type=n1-standard-2 \
    --master-boot-disk-size=50 \
    --num-workers=2 \
    --worker-machine-type=n1-standard-2 \
    --worker-boot-disk-size=50 \
    --image-version=2.1-debian11 \
    --enable-component-gateway

echo ""
echo "Cluster created. Submit a job with scripts/03_submit_job.sh."
