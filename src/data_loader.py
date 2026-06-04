"""
Load the MovieLens dataset into Spark DataFrames.

Uses explicit schemas (faster than inferSchema, especially at 25M scale)
and validates basic invariants after loading.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from .utils import resolve_path, setup_logger

logger = setup_logger("data_loader")


# ─── Schemas ────────────────────────────────────────────────────────────────

RATINGS_SCHEMA = StructType([
    StructField("userId",    IntegerType(), nullable=False),
    StructField("movieId",   IntegerType(), nullable=False),
    StructField("rating",    FloatType(),   nullable=False),
    StructField("timestamp", LongType(),    nullable=False),
])

MOVIES_SCHEMA = StructType([
    StructField("movieId", IntegerType(), nullable=False),
    StructField("title",   StringType(),  nullable=False),
    StructField("genres",  StringType(),  nullable=False),
])

TAGS_SCHEMA = StructType([
    StructField("userId",    IntegerType(), nullable=False),
    StructField("movieId",   IntegerType(), nullable=False),
    StructField("tag",       StringType(),  nullable=False),
    StructField("timestamp", LongType(),    nullable=False),
])


# ─── Loaders ────────────────────────────────────────────────────────────────

def load_ratings(spark: SparkSession, config: dict) -> DataFrame:
    """Load ratings.csv with explicit schema."""
    path = resolve_path(config["paths"]["base"], config["paths"]["ratings"])
    logger.info(f"Loading ratings from {path}")

    df = (
        spark.read
        .option("header", "true")
        .schema(RATINGS_SCHEMA)
        .csv(path)
    )
    n = df.count()
    logger.info(f"Loaded {n:,} ratings")
    return df


def load_movies(spark: SparkSession, config: dict) -> DataFrame:
    """Load movies.csv with explicit schema."""
    path = resolve_path(config["paths"]["base"], config["paths"]["movies"])
    logger.info(f"Loading movies from {path}")

    df = (
        spark.read
        .option("header", "true")
        .schema(MOVIES_SCHEMA)
        .csv(path)
    )
    n = df.count()
    logger.info(f"Loaded {n:,} movies")
    return df


def load_tags(spark: SparkSession, config: dict) -> DataFrame:
    """Load tags.csv with explicit schema (optional; not used in v1 model)."""
    path = resolve_path(config["paths"]["base"], config["paths"]["tags"])
    logger.info(f"Loading tags from {path}")
    return (
        spark.read
        .option("header", "true")
        .schema(TAGS_SCHEMA)
        .csv(path)
    )


# ─── Stats ──────────────────────────────────────────────────────────────────

def report_dataset_stats(ratings: DataFrame, movies: DataFrame) -> dict:
    """Compute and log high-level dataset statistics."""
    stats = {
        "n_ratings":  ratings.count(),
        "n_users":    ratings.select("userId").distinct().count(),
        "n_movies":   ratings.select("movieId").distinct().count(),
        "n_titles":   movies.count(),
        "min_rating": float(ratings.agg({"rating": "min"}).first()[0]),
        "max_rating": float(ratings.agg({"rating": "max"}).first()[0]),
        "avg_rating": float(ratings.agg({"rating": "avg"}).first()[0]),
    }

    logger.info("=" * 50)
    logger.info("Dataset Statistics")
    logger.info("=" * 50)
    logger.info(f"  Total ratings:  {stats['n_ratings']:>12,}")
    logger.info(f"  Unique users:   {stats['n_users']:>12,}")
    logger.info(f"  Unique movies:  {stats['n_movies']:>12,}")
    logger.info(f"  Movie titles:   {stats['n_titles']:>12,}")
    logger.info(f"  Rating range:   {stats['min_rating']} - {stats['max_rating']}")
    logger.info(f"  Average rating: {stats['avg_rating']:.3f}")
    logger.info("=" * 50)

    return stats
