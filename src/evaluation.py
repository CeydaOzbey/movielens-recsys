"""
Evaluation metrics for recommender systems.

Regression metrics: RMSE, MAE.
Ranking metrics:    Precision@K, Recall@K, NDCG@K.
Plus a naive baseline (predict the global mean) for comparison.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col

from .utils import setup_logger

logger = setup_logger("evaluation")


# ─── Regression Metrics ─────────────────────────────────────────────────────

def naive_baseline(train: DataFrame, test: DataFrame) -> Dict[str, float]:
    """Predict the global mean rating for everyone, compute RMSE and MAE."""
    global_mean = float(train.agg({"rating": "avg"}).first()[0])

    test_pdf = test.select("rating").toPandas()
    truths = test_pdf["rating"].values
    preds  = np.full_like(truths, fill_value=global_mean, dtype=float)

    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    mae  = float(np.mean(np.abs(preds - truths)))

    logger.info(
        f"Naive baseline (mean={global_mean:.3f}): RMSE={rmse:.4f}, MAE={mae:.4f}"
    )
    return {"rmse": rmse, "mae": mae, "global_mean": global_mean}


# ─── Ranking Metrics ────────────────────────────────────────────────────────

def precision_at_k(
    model: ALSModel,
    test: DataFrame,
    k: int = 10,
    rating_threshold: float = 4.0,
) -> float:
    """
    For each user, the fraction of the top-K recommended items that are
    in the user's set of "relevant" items (ratings >= threshold in test).
    """
    relevant = (
        test
        .filter(col("rating") >= rating_threshold)
        .groupBy("userId")
        .agg(F.collect_set("movieId").alias("relevant_items"))
    )
    user_recs = (
        model.recommendForAllUsers(k)
        .selectExpr("userId",
                     "transform(recommendations, x -> x.movieId) as recommended")
    )
    joined = user_recs.join(relevant, on="userId", how="inner")
    if joined.count() == 0:
        logger.warning("No users with both recommendations and relevant items")
        return 0.0

    joined = joined.withColumn(
        "hits",
        F.size(F.array_intersect(col("recommended"), col("relevant_items"))),
    )
    precision = float(
        joined
        .agg((F.sum("hits") / (F.count("*") * k)).alias("precision"))
        .first()[0] or 0.0
    )
    logger.info(f"Precision@{k} = {precision:.4f}")
    return precision


def recall_at_k(
    model: ALSModel,
    test: DataFrame,
    k: int = 10,
    rating_threshold: float = 4.0,
) -> float:
    """
    For each user, the fraction of their relevant items that appear in the
    top-K recommendations.
    """
    relevant = (
        test
        .filter(col("rating") >= rating_threshold)
        .groupBy("userId")
        .agg(F.collect_set("movieId").alias("relevant_items"))
    )
    user_recs = (
        model.recommendForAllUsers(k)
        .selectExpr("userId",
                     "transform(recommendations, x -> x.movieId) as recommended")
    )
    joined = user_recs.join(relevant, on="userId", how="inner")
    if joined.count() == 0:
        return 0.0

    joined = (
        joined
        .withColumn(
            "hits",
            F.size(F.array_intersect(col("recommended"), col("relevant_items"))),
        )
        .withColumn("n_relevant", F.size(col("relevant_items")))
    )
    recall = float(
        joined
        .filter(col("n_relevant") > 0)
        .agg(F.avg(col("hits") / col("n_relevant")).alias("recall"))
        .first()[0] or 0.0
    )
    logger.info(f"Recall@{k} = {recall:.4f}")
    return recall


def ndcg_at_k(model: ALSModel, test: DataFrame, k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    Rewards placing relevant items higher in the recommendation list.
    """
    user_recs_pdf = model.recommendForAllUsers(k).toPandas()
    test_pdf = test.toPandas()
    user_test = (
        test_pdf
        .groupby("userId")
        .apply(lambda g: dict(zip(g["movieId"], g["rating"])))
        .to_dict()
    )

    scores = []
    for _, row in user_recs_pdf.iterrows():
        user = row["userId"]
        if user not in user_test:
            continue

        recs = [r["movieId"] for r in row["recommendations"]]
        relevances = [user_test[user].get(m, 0.0) for m in recs]

        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

        ideal = sorted(user_test[user].values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))

        if idcg > 0:
            scores.append(dcg / idcg)

    if not scores:
        return 0.0

    ndcg = float(np.mean(scores))
    logger.info(f"NDCG@{k} = {ndcg:.4f}")
    return ndcg


# ─── Combined Evaluation ────────────────────────────────────────────────────

def evaluate_all(
    model: ALSModel,
    train: DataFrame,
    test: DataFrame,
    config: dict,
) -> Dict[str, float]:
    """Run the full evaluation suite and return a metrics dict."""
    k = config["evaluation"]["top_k"]
    threshold = config["evaluation"]["rating_threshold"]

    predictions = model.transform(test)
    rmse_eval = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction",
    )
    mae_eval = RegressionEvaluator(
        metricName="mae", labelCol="rating", predictionCol="prediction",
    )

    metrics = {
        "rmse":           float(rmse_eval.evaluate(predictions)),
        "mae":            float(mae_eval.evaluate(predictions)),
        "precision_at_k": precision_at_k(model, test, k=k, rating_threshold=threshold),
        "recall_at_k":    recall_at_k(model, test, k=k, rating_threshold=threshold),
        "ndcg_at_k":      ndcg_at_k(model, test, k=k),
    }

    logger.info("=" * 50)
    logger.info("Evaluation Summary")
    logger.info("=" * 50)
    for name, value in metrics.items():
        logger.info(f"  {name:20s} = {value:.4f}")
    logger.info("=" * 50)

    return metrics
