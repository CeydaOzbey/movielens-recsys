"""Cold-start filtering and train/test splitting."""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, from_unixtime, to_timestamp

from .utils import setup_logger

logger = setup_logger("preprocessing")


def filter_cold_start_users(ratings: DataFrame, min_ratings: int = 20) -> DataFrame:
    user_counts = (
        ratings
        .groupBy("userId")
        .agg(count("rating").alias("n_ratings"))
        .filter(col("n_ratings") >= min_ratings)
        .select("userId")
    )
    filtered = ratings.join(user_counts, on="userId", how="inner")

    n_before = ratings.select("userId").distinct().count()
    n_after  = filtered.select("userId").distinct().count()
    logger.info(
        f"Cold-start filter: kept {n_after:,} of {n_before:,} users "
        f"(>= {min_ratings} ratings)"
    )
    return filtered


def random_train_test_split(ratings: DataFrame, train_ratio: float = 0.8, seed: int = 42):
    train, test = ratings.randomSplit([train_ratio, 1.0 - train_ratio], seed=seed)
    logger.info(f"Random split: train={train.count():,}, test={test.count():,}")
    return train, test


def time_based_split(ratings: DataFrame, threshold_date: str = "2018-01-01"):
    df = ratings.withColumn("dt", to_timestamp(from_unixtime(col("timestamp"))))
    threshold = F.lit(threshold_date).cast("timestamp")

    train = df.filter(col("dt") <  threshold).drop("dt")
    test  = df.filter(col("dt") >= threshold).drop("dt")
    logger.info(
        f"Time split @ {threshold_date}: "
        f"train={train.count():,}, test={test.count():,}"
    )
    return train, test


def attach_movie_metadata(ratings: DataFrame, movies: DataFrame) -> DataFrame:
    enriched = ratings.join(movies, on="movieId", how="inner")
    logger.info(f"Joined with movies metadata. Rows: {enriched.count():,}")
    return enriched


def preprocess(ratings: DataFrame, config: dict):
    pp = config["preprocessing"]
    seed = config["project"]["random_seed"]

    filtered = filter_cold_start_users(ratings, pp["min_user_ratings"])

    if pp.get("use_time_split", False):
        train, test = time_based_split(filtered, pp["time_split_threshold"])
    else:
        train, test = random_train_test_split(filtered, pp["train_split"], seed)

    train.cache()
    test.cache()
    return train, test
