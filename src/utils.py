"""
Logging, configuration, and Spark session helpers.

Importing this module does NOT start a SparkSession - that is created
explicitly via `get_spark_session()` when needed.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Spark is only imported inside functions that need it, so that this module
# can be imported in environments where pyspark is not installed (e.g. when
# only running the SVD/KNN baselines).


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logger(name: str = "movielens", level: str = "INFO") -> logging.Logger:
    """
    Create a logger that writes to stdout with a consistent format.

    Idempotent: calling twice with the same name does not duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ─── Config ─────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load a YAML config file and return it as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Path Helpers ───────────────────────────────────────────────────────────

def resolve_path(base: str, sub: str) -> str:
    """
    Join a base path (local, gs://, or s3://) with a sub-path.

    Examples:
        resolve_path("data/", "ratings.csv")               -> "data/ratings.csv"
        resolve_path("gs://bucket/ml-25m/", "ratings.csv") -> "gs://bucket/ml-25m/ratings.csv"
    """
    if base.startswith(("gs://", "s3://", "s3a://")):
        return base.rstrip("/") + "/" + sub.lstrip("/")
    return os.path.join(base, sub)


def ensure_output_dir(path: str) -> None:
    """Create a local output directory if it does not exist (cloud paths are skipped)."""
    if not path.startswith(("gs://", "s3://", "s3a://")):
        Path(path).mkdir(parents=True, exist_ok=True)


# ─── Spark Session ──────────────────────────────────────────────────────────

def get_spark_session(config: dict):
    """
    Create or return the active Spark session.

    Imports pyspark lazily so this module can be used in environments without
    Spark (e.g. SVD-only or KNN-only runs).
    """
    from pyspark.sql import SparkSession  # noqa: WPS433 (intentional lazy import)

    cfg = config.get("spark", {})
    spark = (
        SparkSession.builder
        .appName(cfg.get("app_name", "MovieLens-RecSys"))
        .config("spark.sql.shuffle.partitions", cfg.get("shuffle_partitions", 200))
        .config("spark.driver.memory",   cfg.get("driver_memory",   "4g"))
        .config("spark.executor.memory", cfg.get("executor_memory", "4g"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(cfg.get("log_level", "WARN"))
    return spark
