"""SVD baseline using Surprise (single-machine, small dataset only)."""
from __future__ import annotations

from pyspark.sql import DataFrame
from surprise import SVD, Dataset, Reader, accuracy

from ..utils import setup_logger

logger = setup_logger("svd_baseline")


def spark_to_surprise(df: DataFrame) -> Dataset:
    pdf = df.select("userId", "movieId", "rating").toPandas()
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(pdf[["userId", "movieId", "rating"]], reader)


def train_svd(train_df: DataFrame, n_factors: int = 50, n_epochs: int = 20):
    logger.info(f"Training SVD: n_factors={n_factors}, n_epochs={n_epochs}")
    data = spark_to_surprise(train_df)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
    algo.fit(trainset)
    logger.info("SVD training complete")
    return algo


def evaluate_svd(algo, test_df: DataFrame):
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
