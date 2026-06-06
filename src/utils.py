"""Logging, config, and Spark session helpers."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def setup_logger(name: str = "movielens", level: str = "INFO") -> logging.Logger:
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


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: str, sub: str) -> str:
    if base.startswith(("gs://", "s3://", "s3a://")):
        return base.rstrip("/") + "/" + sub.lstrip("/")
    return os.path.join(base, sub)


def ensure_output_dir(path: str) -> None:
    if not path.startswith(("gs://", "s3://", "s3a://")):
        Path(path).mkdir(parents=True, exist_ok=True)


def get_spark_session(config: dict):
    from pyspark.sql import SparkSession  # lazy import: pyspark not required for SVD/KNN runs

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
