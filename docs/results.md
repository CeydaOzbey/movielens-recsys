# Results

## Headline Numbers

The primary ALS model achieves the following on the held-out test set
(20% of the cold-start-filtered MovieLens 25M):

| Metric | Value | Target | Status |
|---|---|---|---|
| **Test RMSE** | **0.8133** | < 0.85 | OK Beats target |
| **Test MAE**  | **0.6312** |  -     | OK |

Compared to the naive baseline (predicting the global mean rating, 3.53):

| Model | RMSE | MAE | Improvement over naive |
|---|---|---|---|
| Naive (global mean) | 1.117 | 0.934 | -- |
| **ALS (this work)** | **0.8133** | **0.6312** | **27% RMSE reduction** |

## Dataset

| Property | Value |
|---|---|
| Source | MovieLens 25M (GroupLens, 2019) |
| Total ratings | 25,000,095 |
| Unique users | 162,541 |
| Unique movies | 59,047 |
| Rating scale | 0.5 - 5.0 (half-star steps) |
| Date range | January 1995 - November 2019 |

## Runtime Breakdown (full 25M on 3-node cluster)

| Stage | Time (s) | Time (min) |
|---|---|---|
| Load + count | 111.7 | 1.9 |
| Cold-start filter | 88.8 | 1.5 |
| Train/test split + cache | ~80 | 1.3 |
| **ALS training** | **319.2** | **5.3** |
| Top-N recommendations | ~250 | 4.2 |
| Save model + recs to GCS | ~150 | 2.5 |
| **Total pipeline** | **1001.7** | **16.7** |

## Hyperparameters Used

```yaml
als:
  rank: 10
  maxIter: 10
  regParam: 0.1
  coldStartStrategy: "drop"
  nonnegative: true
  seed: 42
```

These are the Spark MLlib defaults, chosen as a sensible v1 baseline. A grid
search over (rank in [10, 20, 50], maxIter in [10, 20], regParam in
[0.05, 0.1, 0.2]) can be run with `scripts/run_local.py --tune`.

## Interesting Patterns in the Data

Looking at the top 10 most-rated movies:

| # | Title | num_ratings | avg_rating |
|---|---|---|---|
| 1 | Forrest Gump (1994) | 81,491 | 4.05 |
| 2 | Shawshank Redemption, The (1994) | 81,482 | 4.41 |
| 3 | Pulp Fiction (1994) | 79,672 | 4.19 |
| 4 | Silence of the Lambs, The (1991) | 74,127 | 4.15 |
| 5 | Matrix, The (1999) | 72,674 | 4.15 |
| 6 | Star Wars: Episode IV - A New Hope (1977) | 68,717 | 4.12 |
| 7 | Jurassic Park (1993) | 64,144 | 3.68 |
| 8 | Schindler's List (1993) | 60,411 | 4.25 |
| 9 | Braveheart (1995) | 59,184 | 4.00 |
| 10 | Fight Club (1999) | 58,773 | 4.23 |

**Observations:**

1. **Temporal bias:** Nine of the top ten most-rated movies are from the
   1990s. This is because the MovieLens dataset has been collected
   continuously since 1996, so older popular films have had more time to
   accumulate ratings.

2. **Quality stays high at the top:** Average ratings for the most-rated
   films cluster around 4.0 - 4.4, with the notable exception of Jurassic
   Park (3.68). This suggests that the heavy-tail of "popular" films are
   also generally well-liked.

3. **Implication for the recommender:** A pure popularity baseline would
   push newer movies down, because they have fewer ratings just by virtue
   of being newer. The ALS model partially corrects for this by working in
   latent factor space, but a temporal-decay term in the loss could improve
   recommendations for recent releases. This is a candidate improvement for
   future work.

## Error Analysis

### Predictions outside the rating range

Spark MLlib's ALS does not constrain predictions to [0.5, 5.0]. Inspecting
the predictions DataFrame, the predicted ratings span roughly [-0.5, 6.5],
with the tails caused by extreme latent factors for very active or very
selective users. Clipping the predictions to [0.5, 5.0] reduces RMSE by
about 0.01 in our experiments (an additional small win, but not the main
story).

### Cold-start dropped rows

Approximately 0.1% of test-set rows were dropped because they involved a
user or movie not seen during training (the `coldStartStrategy="drop"`
behaviour). For a more principled solution, future work could:
- Use a content-based fallback (genre similarity) for genuinely new users
- Use the global mean rating as a fallback (mild but stable)

## Comparison with Single-Machine Baselines

On the small subset (100K ratings) — distributed methods are not needed
at this scale — we additionally trained an SVD model (Surprise library)
and an item-based KNN model (scikit-learn / scipy):

| Model | RMSE | MAE | Notes |
|---|---|---|---|
| ALS (Spark MLlib)  | 0.88 | 0.68 | Identical settings as on 25M |
| SVD (Surprise)     | 0.87 | 0.66 | n_factors=50, n_epochs=20 |
| Item-based KNN     | 0.92 | 0.71 | k=20, cosine similarity |
| Naive (global avg) | 1.04 | 0.85 | Baseline |

On the small subset, SVD slightly edges out ALS, which is consistent with
the literature: SVD's epoch-based SGD converges more carefully on small
data, while ALS's strength is parallelism on large data.

When we move to the full 25M dataset on the cluster, ALS RMSE drops to
0.8133, while SVD becomes impractical (Surprise pulls all data into a
single pandas DataFrame; even loading 25M rows takes minutes and the
training would not finish in a reasonable time on a single machine).

## Cost

The full 25M experiment cost approximately **$0.10** in GCP credits:

- Dataproc cluster: ~17 minutes x ~$0.30/hour = $0.08
- GCS storage:   1.1 GB x $0.02/GB/month, prorated to a few cents
- Egress / API:  negligible

This is well within the $300 free trial and demonstrates that distributed
ML on a managed cloud platform is now genuinely cheap for student projects.
