"""EDA: rating distribution, top movies, user activity."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, count, desc

from .utils import setup_logger

logger = setup_logger("eda")


def plot_rating_distribution(ratings: DataFrame, output_dir: str) -> str:
    rating_counts = (
        ratings
        .groupBy("rating")
        .agg(count("*").alias("count"))
        .orderBy("rating")
        .toPandas()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        rating_counts["rating"],
        rating_counts["count"],
        width=0.4,
        color="#2E5FA3",
        edgecolor="#1F3864",
    )
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Rating Distribution - MovieLens")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = str(Path(output_dir) / "rating_distribution.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info(f"Saved rating distribution plot: {out_path}")
    return out_path


def top_n_movies(ratings: DataFrame, movies: DataFrame, n: int = 10, min_ratings: int = 50) -> DataFrame:
    result = (
        ratings
        .join(movies, on="movieId")
        .groupBy("movieId", "title")
        .agg(
            count("rating").alias("num_ratings"),
            avg("rating").alias("avg_rating"),
        )
        .filter(f"num_ratings >= {min_ratings}")
        .orderBy(desc("num_ratings"))
        .limit(n)
    )
    logger.info(f"Top {n} most-rated movies:")
    result.show(n, truncate=False)
    return result


def user_activity_summary(ratings: DataFrame) -> dict:
    activity = ratings.groupBy("userId").agg(count("rating").alias("n_ratings"))
    pdf = activity.toPandas()
    summary = {
        "min_ratings_per_user":    int(pdf["n_ratings"].min()),
        "max_ratings_per_user":    int(pdf["n_ratings"].max()),
        "median_ratings_per_user": int(pdf["n_ratings"].median()),
        "mean_ratings_per_user":   float(pdf["n_ratings"].mean()),
    }
    logger.info("User activity summary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    return summary


def run_eda(ratings: DataFrame, movies: DataFrame, output_dir: str) -> dict:
    return {
        "rating_distribution_plot": plot_rating_distribution(ratings, output_dir),
        "top_movies":    top_n_movies(ratings, movies),
        "user_activity": user_activity_summary(ratings),
    }
