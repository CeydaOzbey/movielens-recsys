"""
Preprocessing steps:
  1. Filter cold-start users (those with fewer than N ratings).
  2. Split ratings into train and test sets (random or time-based).
  3. Cache the resulting DataFrames for repeated use.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, from_unixtime, to_timestamp

from .utils import setup_logger

logger = setup_logger("preprocessing")


def filter_cold_start_users(ratings: DataFrame, min_ratings: int = 20) -> DataFrame:
    """
    Remove users who have fewer than `min_ratings` ratings. Cold-start users
    have too little history for collaborative filtering to work well.
    """
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


def random_train_test_split(
    ratings: DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[DataFrame, DataFrame]:
    """Random 80/20 split. Fast and simple, used for the v1 baseline."""
    train, test = ratings.randomSplit([train_ratio, 1.0 - train_ratio], seed=seed)
    logger.info(f"Random split: train={train.count():,}, test={test.count():,}")
    return train, test


def time_based_split(
    ratings: DataFrame,
    threshold_date: str = "2018-01-01",
) -> tuple[DataFrame, DataFrame]:
    """
    Time-based split: ratings before the threshold go to training, the rest to
    test. This is more realistic than random for production recommenders since
    it mimics serving on data from the future.
    """
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
    """Inner-join ratings with movies to attach title and genres."""
    enriched = ratings.join(movies, on="movieId", how="inner")
    logger.info(f"Joined with movies metadata. Rows: {enriched.count():,}")
    return enriched


def preprocess(
    ratings: DataFrame,
    config: dict,
) -> tuple[DataFrame, DataFrame]:
    """
    Full preprocessing pipeline. Returns (train, test).
    """
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
