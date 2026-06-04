#!/usr/bin/env python3
"""
PySpark pipeline for the FULL MovieLens 25M dataset on GCP Dataproc.

This is the script that produced the results reported in the final paper:
  - Test RMSE: 0.8133
  - Test MAE:  0.6312
  - Total pipeline time: 1001 seconds (~17 minutes)

It is self-contained (no relative imports) so it can be submitted as a
standalone PySpark job without needing to ship the whole src/ package.

Usage (submitted via gcloud):
    gcloud dataproc jobs submit pyspark scripts/run_dataproc.py \
        --cluster=movielens-cluster --region=us-central1 \
        -- --bucket gs://your-bucket
"""
import argparse
import json
import time

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, desc


def main(bucket_uri: str) -> None:
    """End-to-end ALS pipeline on the full 25M dataset."""
    spark = (
        SparkSession.builder
        .appName("MovieLens-25M-ALS")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("MovieLens 25M ALS Pipeline")
    print("=" * 60)

    metrics = {}
    t0 = time.time()

    # ─── 1. Load ────────────────────────────────────────────────────────────
    ratings = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(f"{bucket_uri}/ml-25m/ratings.csv")
    )
    movies = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(f"{bucket_uri}/ml-25m/movies.csv")
    )

    n_ratings = ratings.count()
    n_users   = ratings.select("userId").distinct().count()
    n_movies  = ratings.select("movieId").distinct().count()
    metrics["n_ratings"] = n_ratings
    metrics["n_users"]   = n_users
    metrics["n_movies"]  = n_movies

    print(f"\nDataset stats:")
    print(f"  Total ratings:  {n_ratings:>12,}")
    print(f"  Unique users:   {n_users:>12,}")
    print(f"  Unique movies:  {n_movies:>12,}")
    print(f"  Load + count:   {time.time()-t0:.1f}s")

    # ─── 2. Cold-start filter ───────────────────────────────────────────────
    t1 = time.time()
    user_counts = (
        ratings
        .groupBy("userId")
        .agg(count("rating").alias("n"))
        .filter(col("n") >= 20)
        .select("userId")
    )
    filtered = ratings.join(user_counts, on="userId", how="inner")

    metrics["n_users_after_filter"] = filtered.select("userId").distinct().count()
    metrics["n_ratings_after_filter"] = filtered.count()

    print(f"\nCold-start filter (>= 20 ratings):")
    print(f"  Active users:   {metrics['n_users_after_filter']:>12,}")
    print(f"  Filtered rows:  {metrics['n_ratings_after_filter']:>12,}")
    print(f"  Elapsed:        {time.time()-t1:.1f}s")

    # ─── 3. Train/test split ────────────────────────────────────────────────
    train, test = filtered.randomSplit([0.8, 0.2], seed=42)
    metrics["train_rows"] = train.count()
    metrics["test_rows"]  = test.count()
    print(f"\nSplit (80/20):")
    print(f"  Train rows: {metrics['train_rows']:,}")
    print(f"  Test rows:  {metrics['test_rows']:,}")

    # ─── 4. ALS training ────────────────────────────────────────────────────
    t2 = time.time()
    als = ALS(
        maxIter=10,
        rank=10,
        regParam=0.1,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        seed=42,
    )
    model = als.fit(train)
    metrics["training_time_seconds"] = round(time.time() - t2, 2)
    print(f"\nALS training complete in {metrics['training_time_seconds']:.1f}s")

    # ─── 5. Evaluate ────────────────────────────────────────────────────────
    predictions = model.transform(test)
    rmse = float(
        RegressionEvaluator(metricName="rmse", labelCol="rating",
                             predictionCol="prediction").evaluate(predictions)
    )
    mae = float(
        RegressionEvaluator(metricName="mae", labelCol="rating",
                             predictionCol="prediction").evaluate(predictions)
    )
    metrics["rmse"] = rmse
    metrics["mae"]  = mae
    print(f"\nResults:")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  Test MAE:  {mae:.4f}")

    # ─── 6. Discovery: top movies ───────────────────────────────────────────
    top_movies = (
        ratings.join(movies, "movieId")
        .groupBy("movieId", "title")
        .agg(count("rating").alias("num_ratings"),
             avg("rating").alias("avg_rating"))
        .filter("num_ratings >= 1000")
        .orderBy(desc("num_ratings"))
        .limit(10)
    )
    print(f"\nTop 10 most-rated movies:")
    top_movies.show(truncate=False)

    top_movies.coalesce(1).write.mode("overwrite").csv(
        f"{bucket_uri}/results/top10_movies",
        header=True,
    )

    # ─── 7. Save model and recommendations ──────────────────────────────────
    model.write().overwrite().save(f"{bucket_uri}/results/als_model")
    print(f"\nModel saved to {bucket_uri}/results/als_model")

    user_recs = model.recommendForAllUsers(10)
    user_recs.write.mode("overwrite").parquet(
        f"{bucket_uri}/results/top10_recommendations"
    )
    print(f"Top-10 recommendations saved to "
          f"{bucket_uri}/results/top10_recommendations")

    # ─── 8. Save metrics JSON ───────────────────────────────────────────────
    metrics["total_pipeline_seconds"] = round(time.time() - t0, 2)
    print(f"\nTotal pipeline time: {metrics['total_pipeline_seconds']:.1f}s")

    metrics_rdd = spark.sparkContext.parallelize([json.dumps(metrics, indent=2)])
    metrics_rdd.saveAsTextFile(f"{bucket_uri}/results/metrics_json")
    print(f"Metrics saved to {bucket_uri}/results/metrics_json")

    print("=" * 60)
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket URI, e.g. gs://movielens-yourname",
    )
    args = parser.parse_args()
    main(args.bucket)
