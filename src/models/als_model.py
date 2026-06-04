"""
ALS (Alternating Least Squares) collaborative filtering model.

This is the PRIMARY model for the project. Uses Spark MLlib's distributed
implementation, which scales to hundreds of millions of ratings.
"""
from __future__ import annotations

from itertools import product
from typing import Optional

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame

from ..utils import setup_logger

logger = setup_logger("als_model")


def build_als(config: dict) -> ALS:
    """Construct an ALS estimator from the config values."""
    cfg = config["als"]
    return ALS(
        rank=cfg["rank"],
        maxIter=cfg["max_iter"],
        regParam=cfg["reg_param"],
        userCol=cfg["user_col"],
        itemCol=cfg["item_col"],
        ratingCol=cfg["rating_col"],
        coldStartStrategy=cfg["cold_start_strategy"],
        nonnegative=cfg.get("nonnegative", True),
        implicitPrefs=False,
        seed=config["project"]["random_seed"],
    )


def train_als(train: DataFrame, config: dict) -> ALSModel:
    """Fit ALS on the training data and return the fitted model."""
    als = build_als(config)
    logger.info(
        f"Training ALS: rank={als.getRank()}, maxIter={als.getMaxIter()}, "
        f"regParam={als.getRegParam()}"
    )
    model = als.fit(train)
    logger.info("ALS training complete")
    return model


def evaluate_rmse(model: ALSModel, test: DataFrame, config: dict) -> float:
    """Compute Test RMSE."""
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol=config["als"]["rating_col"],
        predictionCol="prediction",
    )
    rmse = float(evaluator.evaluate(predictions))
    logger.info(f"Test RMSE: {rmse:.4f}")
    return rmse


def evaluate_mae(model: ALSModel, test: DataFrame, config: dict) -> float:
    """Compute Test MAE."""
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="mae",
        labelCol=config["als"]["rating_col"],
        predictionCol="prediction",
    )
    mae = float(evaluator.evaluate(predictions))
    logger.info(f"Test MAE: {mae:.4f}")
    return mae


def hyperparameter_search(
    train: DataFrame,
    val: DataFrame,
    config: dict,
) -> tuple[ALSModel, dict]:
    """
    Manual grid search over the values in `als_tuning`.

    Returns the best model and a dict of {params_string: rmse}.
    """
    space = config["als_tuning"]
    grid = list(product(space["rank"], space["max_iter"], space["reg_param"]))
    logger.info(f"Hyperparameter search: {len(grid)} configurations")

    results = {}
    best_rmse = float("inf")
    best_model: Optional[ALSModel] = None
    best_params: Optional[tuple] = None

    for rank, max_iter, reg_param in grid:
        cfg = {**config, "als": {**config["als"],
                                  "rank": rank,
                                  "max_iter": max_iter,
                                  "reg_param": reg_param}}
        model = train_als(train, cfg)
        rmse = evaluate_rmse(model, val, cfg)

        key = f"rank={rank}, maxIter={max_iter}, regParam={reg_param}"
        results[key] = rmse
        logger.info(f"  {key} -> RMSE = {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = (rank, max_iter, reg_param)

    logger.info(
        f"Best config: rank={best_params[0]}, maxIter={best_params[1]}, "
        f"regParam={best_params[2]} -> RMSE = {best_rmse:.4f}"
    )
    return best_model, results


def generate_top_n_recommendations(model: ALSModel, n: int = 10) -> DataFrame:
    """Generate top-N movie recommendations for every user."""
    return model.recommendForAllUsers(n)


def save_model(model: ALSModel, path: str) -> None:
    """Save a trained ALS model (path can be local, gs://, or s3://)."""
    model.write().overwrite().save(path)
    logger.info(f"Saved ALS model to: {path}")


def load_model(path: str) -> ALSModel:
    """Load a previously saved ALS model."""
    logger.info(f"Loading ALS model from: {path}")
    return ALSModel.load(path)
