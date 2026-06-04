# System Architecture

## Overview

The pipeline is a 4-stage data flow that reads ratings from object storage,
preprocesses and trains a collaborative filtering model on a managed Spark
cluster, and writes results back to object storage. The same code runs
identically on a local machine (using small subset) and on the cloud (using
the full 25M dataset).

```
+--------------+      +-----------------+      +-------------------+      +-------------------+
| 1. Data      |      | 2. Preprocessing|      | 3. Model Training |      | 4. Evaluation     |
| Ingestion    | ---> |   (Spark)       | ---> |    (ALS)          | ---> |    & Output       |
| (GCS/S3)     |      |                 |      |  Spark MLlib      |      |                   |
+--------------+      +-----------------+      +-------------------+      +-------------------+
     |                                                                            |
     |                                                                            |
   CSV files                                                              RMSE, MAE, Top-N
   ratings.csv                                                            recommendations.parquet
   movies.csv                                                             als_model/
```

## Cloud Architecture (GCP)

```
                +-------------------------------+
                |    Google Cloud Storage       |
                |    (bucket: movielens-ceyda)  |
                |                               |
                |  /ml-25m/                     |
                |    ratings.csv  (678 MB)      |
                |    movies.csv     (3 MB)      |
                |    tags.csv      (38 MB)      |
                |  /results/                    |
                |    als_model/                 |
                |    top10_movies.csv           |
                |    top10_recommendations/     |
                |    metrics.json               |
                +---------------+---------------+
                                |
                                | gs:// reads / writes
                                v
                +-------------------------------+
                |  Dataproc Cluster             |
                |  (us-central1)                |
                |                               |
                |  Master (n1-standard-2):      |
                |    + Spark driver             |
                |    + YARN ResourceManager     |
                |                               |
                |  Workers (2 x n1-standard-2): |
                |    + Spark executors          |
                |    + YARN NodeManager         |
                +-------------------------------+
```

## Why Spark (and not single-machine)?

The MovieLens 25M dataset has 25M rows and produces a 162K x 59K user-item
matrix. Even sparse, the matrix and the ALS factor matrices do not fit
comfortably in RAM on a single laptop. ALS as implemented in Spark MLlib
partitions both U and V across workers and does the least-squares solves
in parallel, which is why training takes ~5 minutes on a 3-node cluster
rather than hours on a single machine.

## Data Flow Detail

```
ratings.csv (25M rows)
     |
     v  load with explicit schema
DataFrame(userId int, movieId int, rating float, ts long)
     |
     v  groupBy(userId).agg(count) >= 20
filtered DataFrame (cold-start removed)
     |
     v  randomSplit([0.8, 0.2], seed=42)
+---------------+         +---------------+
| train (~20M)  |         | test (~5M)    |
+---------------+         +---------------+
     |
     v  ALS.fit()
ALSModel (rank=10 factors per user, per item)
     |
     v  .transform(test)
predictions DataFrame
     |
     v  RegressionEvaluator (RMSE), (MAE)
final metrics
```

## Hyperparameters

The v1 baseline uses the Spark MLlib defaults:

| Parameter | Value | Justification |
|---|---|---|
| `rank` | 10 | Default, good starting point for small/medium datasets |
| `maxIter` | 10 | Usually sufficient for ALS convergence |
| `regParam` | 0.1 | L2 regularization, default |
| `coldStartStrategy` | `"drop"` | Avoids NaN predictions for unseen users/items |
| `nonnegative` | True | Forces non-negative factors (more interpretable) |
| `seed` | 42 | Reproducibility |

A grid search over rank, maxIter, regParam can be run with
`scripts/run_local.py --tune`.

## Reproducibility

- All random splits use `seed=42`.
- The Spark version is pinned (3.5) via `requirements.txt`.
- The Dataproc image version is pinned (`2.1-debian11`) in
  `configs/cluster_config.yaml`.
- Hyperparameters live in `configs/config.yaml`.
- The exact commands to provision and tear down infrastructure are in
  `scripts/02_*.sh` and `scripts/04_*.sh`.
