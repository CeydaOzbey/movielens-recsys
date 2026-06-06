"""Lightweight unit tests for the pipeline - run with: pytest tests/ -v"""
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
    cfg = load_config("configs/config.yaml")
    assert "project" in cfg
    assert "paths" in cfg
    assert "als" in cfg
    assert "evaluation" in cfg


def test_als_config_values():
    cfg = load_config("configs/config.yaml")
    als = cfg["als"]
    assert als["rank"] == 10
    assert als["max_iter"] == 10
    assert als["reg_param"] == 0.1
    assert als["cold_start_strategy"] == "drop"


def test_seed_is_42():
    cfg = load_config("configs/config.yaml")
    assert cfg["project"]["random_seed"] == 42


def test_load_config_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
