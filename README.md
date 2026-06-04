# Movie Recommendation System on Big Data
### CSE 458 / 541 — Big Data Analytics, Spring 2026

A scalable collaborative filtering recommender system trained on the full **MovieLens 25M** dataset (25 million ratings) using Apache Spark, deployed on **Google Cloud Dataproc**.

**Student:** Ceyda Özbey
**ID:** 1801042636
**Instructor:** Dr. Salih Sarp
**Institution:** Gebze Technical University

---

## Key Results

| Metric | Value | Notes |
|---|---|---|
| **Test RMSE** | **0.8133** | Target was < 0.85 |
| **Test MAE**  | **0.6312** | |
| Total ratings processed | 25,000,095 | Full MovieLens 25M |
| Unique users | 162,541 | |
| Unique movies | 59,047 | |
| ALS training time | 319 s (~5 min) | 3-node Dataproc cluster |
| Total pipeline runtime | 1001 s (~17 min) | End-to-end |
| Cloud platform | GCP Dataproc | 1 master + 2 workers, n1-standard-2 |

See [`results/`](results/) for raw output and [`docs/results.md`](docs/results.md) for full analysis.

---

## Project Structure

```
movielens-recsys/
├── README.md                  # This file
├── REPRODUCIBILITY.md         # How to re-run the experiments
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── config.yaml            # Pipeline hyperparameters and paths
│   └── cluster_config.yaml    # GCP Dataproc cluster specification
│
├── src/                       # Library code
│   ├── data_loader.py         # Load MovieLens into Spark DataFrames
│   ├── preprocessing.py       # Cold-start filter, train/test split
│   ├── eda.py                 # Exploratory data analysis
│   ├── evaluation.py          # RMSE, MAE, Precision@K, NDCG@K
│   ├── utils.py               # Logging, config, Spark session helpers
│   └── models/
│       ├── als_model.py       # Primary: Spark MLlib ALS
│       ├── svd_baseline.py    # Baseline: Surprise SVD
│       └── knn_baseline.py    # Baseline: Item-based KNN (cosine sim)
│
├── scripts/                   # Executable entry points
│   ├── 00_download_data.sh    # Download MovieLens 25M
│   ├── 01_upload_to_gcs.sh    # Upload to GCS bucket
│   ├── 02_create_cluster.sh   # Provision Dataproc cluster
│   ├── 03_submit_job.sh       # Submit PySpark job
│   ├── 04_delete_cluster.sh   # IMPORTANT: tear down to stop billing
│   ├── run_local.py           # Run pipeline locally / on Colab
│   └── run_dataproc.py        # Run on Dataproc (full 25M)
│
├── notebooks/
│   └── demo.ipynb             # Interactive demo for the presentation
│
├── results/                   # Real outputs from the cloud run
│   ├── als_25m_metrics.json
│   ├── top10_movies.csv
│   └── pipeline_logs.txt
│
└── docs/
    ├── architecture.md        # System design diagrams and notes
    └── results.md             # Detailed result analysis
```

---

## Quick Start

### Option 1 — Run locally on the small subset (Colab / laptop)

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Download the MovieLens latest-small dataset (~1 MB)
bash scripts/00_download_data.sh small

# 3) Run the end-to-end pipeline
python scripts/run_local.py --config configs/config.yaml
```

### Option 2 — Run on GCP Dataproc with the full 25M dataset

```bash
# 1) Set environment variables
export PROJECT_ID="your-gcp-project"
export BUCKET="movielens-yourname"
export REGION="us-central1"

# 2) Download + upload data
bash scripts/00_download_data.sh 25m
bash scripts/01_upload_to_gcs.sh

# 3) Create cluster (~3 minutes, costs ~$0.005 / minute)
bash scripts/02_create_cluster.sh

# 4) Submit the PySpark job (~15 minutes on 3-node cluster)
bash scripts/03_submit_job.sh

# 5) IMPORTANT: tear down the cluster to stop billing
bash scripts/04_delete_cluster.sh
```

A complete reproduction sequence is documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Models

Three collaborative-filtering models are implemented and compared:

| Model | Library | Distributed? | Use case |
|---|---|---|---|
| **ALS (primary)** | Spark MLlib | Yes | Scales to 25M+ ratings |
| SVD (baseline)    | Surprise    | No  | Single-machine reference |
| Item-based KNN    | scikit-learn + scipy | No | Interpretable baseline |

The ALS model is the one we deploy on Dataproc. SVD and KNN run on a smaller subset (Surprise and scikit-learn do not natively distribute).

---

## Dataset

[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) — published by GroupLens Research at the University of Minnesota.

- 25,000,095 ratings
- 162,541 unique users
- 62,423 unique movies
- Ratings on a 0.5–5.0 scale, in 0.5 increments
- Date range: January 1995 – November 2019

The dataset is publicly available under a non-commercial research license.

---

## Cost

Total cloud spend for the full 25M experiment: **< $0.50** (out of the $300 free trial credit).

The Dataproc cluster (1 master + 2 workers, n1-standard-2) costs approximately $0.30 per hour. The job took ~17 minutes, so total compute cost was about $0.08–0.10. GCS storage is within the Always Free quota (5 GB).

---

## License

This project is submitted for academic evaluation at Gebze Technical University, CSE 458/541, Spring 2026.
