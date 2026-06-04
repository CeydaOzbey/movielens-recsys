"""
Lightweight unit tests for the pipeline.

Run with:
    pytest tests/ -v

These tests do not require a Spark session - they exercise the pure-Python
helpers in src/utils.py and the config loading.
"""
from pathlib import Path

import pytest
import yaml

from src.utils import load_config, resolve_path


def test_resolve_path_local():
    assert resolve_path("data/ml-25m/", "ratings.csv") == "data/ml-25m/ratings.csv"


def test_resolve_path_gcs():
    assert (
        resolve_path("gs://bucket/ml-25m/", "ratings.csv")
        == "gs://bucket/ml-25m/ratings.csv"
    )


def test_resolve_path_s3():
    assert (
        resolve_path("s3://bucket/ml-25m", "ratings.csv")
        == "s3://bucket/ml-25m/ratings.csv"
    )


def test_load_config_exists():
    """The bundled config file should load and contain the expected keys."""
    cfg = load_config("configs/config.yaml")
    assert "project" in cfg
    assert "paths" in cfg
    assert "als" in cfg
    assert "evaluation" in cfg


def test_als_config_values():
    """Verify the documented ALS hyperparameters are present."""
    cfg = load_config("configs/config.yaml")
    als = cfg["als"]
    assert als["rank"] == 10
    assert als["max_iter"] == 10
    assert als["reg_param"] == 0.1
    assert als["cold_start_strategy"] == "drop"


def test_seed_is_42():
    """The reported run used seed=42 - this must not change without intent."""
    cfg = load_config("configs/config.yaml")
    assert cfg["project"]["random_seed"] == 42


def test_load_config_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
