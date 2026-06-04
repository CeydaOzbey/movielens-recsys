"""
SVD baseline using the Surprise library.

This is a SINGLE-MACHINE baseline. Surprise pulls the whole rating set into
memory, so it only works on the small subset, not on the full 25M dataset.
The point is to provide a non-distributed reference for the ALS results.
"""
from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame
from surprise import SVD, Dataset, Reader, accuracy

from ..utils import setup_logger

logger = setup_logger("svd_baseline")


def spark_to_surprise(df: DataFrame) -> Dataset:
    """Convert a Spark DataFrame to a Surprise Dataset."""
    pdf = df.select("userId", "movieId", "rating").toPandas()
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(pdf[["userId", "movieId", "rating"]], reader)


def train_svd(train_df: DataFrame, n_factors: int = 50, n_epochs: int = 20):
    """Train an SVD model on the training data."""
    logger.info(f"Training SVD: n_factors={n_factors}, n_epochs={n_epochs}")

    data = spark_to_surprise(train_df)
    trainset = data.build_full_trainset()

    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
    algo.fit(trainset)
    logger.info("SVD training complete")
    return algo


def evaluate_svd(algo, test_df: DataFrame) -> tuple[float, float]:
    """Compute RMSE and MAE for the SVD model on the test set."""
    test_pdf = test_df.select("userId", "movieId", "rating").toPandas()
    testset = list(zip(
        test_pdf["userId"],
        test_pdf["movieId"],
        test_pdf["rating"],
    ))

    predictions = algo.test(testset)
    rmse = float(accuracy.rmse(predictions, verbose=False))
    mae  = float(accuracy.mae(predictions,  verbose=False))

    logger.info(f"SVD Test RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse, mae
