# Reproducibility Guide

This document describes exactly how to reproduce the results reported in the final paper.

## Environment

| Component | Version |
|---|---|
| Python | 3.11 |
| Apache Spark | 3.5 |
| PySpark | 3.5.0 |
| Cloud platform | Google Cloud Platform |
| Dataproc image | 2.1-debian11 |

All other Python dependencies are pinned in [`requirements.txt`](requirements.txt).

## Cluster Specification

The full 25M experiment was run on a Google Cloud Dataproc cluster with the following spec:

| Role   | Count | Machine type   | vCPU | RAM   | Disk |
|--------|-------|----------------|------|-------|------|
| Master | 1     | n1-standard-2  | 2    | 7.5 GB | 50 GB |
| Worker | 2     | n1-standard-2  | 2    | 7.5 GB | 50 GB |

Region: `us-central1`, Zone: `us-central1-a`.

The exact configuration is captured in [`configs/cluster_config.yaml`](configs/cluster_config.yaml) and provisioned via [`scripts/02_create_cluster.sh`](scripts/02_create_cluster.sh).

## Step-by-Step Reproduction

### 1. GCP Setup (one-time)

```bash
# Sign up for the free trial at https://cloud.google.com/free
# Then enable the required APIs:
gcloud services enable dataproc.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable compute.googleapis.com

# Grant the default Compute service account the Dataproc Worker role
PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUM=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
    --role="roles/dataproc.worker"
```

### 2. Data Setup

```bash
# Download MovieLens 25M (about 250 MB)
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip

# Create a GCS bucket and upload
export BUCKET="movielens-yourname-$(date +%s)"
gsutil mb -l us-central1 gs://$BUCKET
gsutil -m cp -r ml-25m/* gs://$BUCKET/ml-25m/
```

### 3. Run the Pipeline

```bash
# Provision the cluster (~3 minutes)
bash scripts/02_create_cluster.sh

# Submit the PySpark job (~15 minutes on the 3-node cluster)
bash scripts/03_submit_job.sh

# IMPORTANT: shut the cluster down to stop charges
bash scripts/04_delete_cluster.sh
```

### 4. Retrieve Results

After the job completes, results are written back to GCS:

```bash
# Download metrics and the model
gsutil -m cp -r gs://$BUCKET/results/ ./results/

# Inspect the metrics
cat results/als_25m_metrics.json
```

## Expected Output

If reproduction is successful, you should see metrics close to:

```json
{
  "rmse": 0.8133,
  "mae":  0.6312,
  "n_ratings": 25000095,
  "n_users": 162541,
  "n_movies": 59047,
  "training_time_seconds": 319.2,
  "total_pipeline_seconds": 1001.7
}
```

Exact RMSE/MAE values may differ in the third decimal place depending on the random seed used in the train/test split. The seed used in our run is `42`, set in `configs/config.yaml`.

## Random Seeds

The following components depend on randomness; all are seeded:

- Train/test split (`ratings.randomSplit([0.8, 0.2], seed=42)`)
- ALS initialization (`seed=42` in the ALS constructor)

## Cost Estimate

| Resource | Duration | Approximate Cost |
|---|---|---|
| Dataproc cluster (3 nodes, n1-standard-2) | 20 minutes | $0.10 |
| GCS storage (1.1 GB, one month) | n/a | $0.02 |
| Network egress | minimal | < $0.01 |
| **Total** | | **< $0.50** |

Well within the $300 free-trial credit.

## Cleanup

After running, **always** delete the cluster:

```bash
gcloud dataproc clusters delete movielens-cluster --region=us-central1 --quiet
```

This is also automated in [`scripts/04_delete_cluster.sh`](scripts/04_delete_cluster.sh).

To remove everything (including the bucket):

```bash
gsutil -m rm -r gs://$BUCKET
```
