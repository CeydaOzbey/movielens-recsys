#!/usr/bin/env python3
"""End-to-end pipeline for LOCAL or COLAB execution."""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_movies, load_ratings, report_dataset_stats
from src.eda import run_eda
from src.evaluation import evaluate_all, naive_baseline
from src.models.als_model import (
    generate_top_n_recommendations,
    save_model,
    train_als,
)
from src.preprocessing import preprocess
from src.utils import (
    ensure_output_dir,
    get_spark_session,
    load_config,
    resolve_path,
    setup_logger,
)

log = setup_logger("pipeline")


def main(config_path, run_baselines):
    start = time.time()
    log.info(f"Starting pipeline with config: {config_path}")

    config = load_config(config_path)
    output_dir = config["paths"]["output_dir"]
    ensure_output_dir(output_dir)

    spark = get_spark_session(config)
    log.info(f"Spark version: {spark.version}")
    log.info(f"Spark master:  {spark.sparkContext.master}")

    try:
        # 1. Load
        ratings = load_ratings(spark, config)
        movies  = load_movies(spark, config)
        stats   = report_dataset_stats(ratings, movies)

        # 2. EDA
        log.info("Running EDA...")
        eda_artifacts = run_eda(ratings, movies, output_dir)

        # 3. Preprocess
        log.info("Preprocessing data...")
        train, test = preprocess(ratings, config)

        # 4. Naive baseline
        log.info("Computing naive baseline...")
        naive_metrics = naive_baseline(train, test)

        # 5. Train ALS (primary)
        log.info("Training ALS model...")
        als_model = train_als(train, config)

        # 6. Evaluate ALS
        log.info("Evaluating ALS model...")
        als_metrics = evaluate_all(als_model, train, test, config)

        # 7. Top-K recommendations
        k = config["evaluation"]["top_k"]
        log.info(f"Generating top-{k} recommendations for all users...")
        user_recs = generate_top_n_recommendations(als_model, n=k)

        # Save as JSON instead of parquet to avoid the Hadoop/winutils
        # dependency on Windows. Pandas writes JSON natively, no Hadoop needed.
        recs_path = resolve_path(output_dir, f"top_{k}_recommendations.json")
        user_recs_pdf = user_recs.toPandas()
        # Convert the list-of-Rows column to plain dicts for clean JSON
        user_recs_pdf["recommendations"] = user_recs_pdf["recommendations"].apply(
            lambda recs: [{"movieId": r["movieId"], "score": float(r["rating"])} for r in recs]
        )
        user_recs_pdf.to_json(recs_path, orient="records", lines=True)
        log.info(f"Saved recommendations to: {recs_path}")

        # 8. Save model
        # On Windows, Spark's model.save also needs Hadoop. We skip it locally
        # and rely on the cloud run for the persisted model artefact.
        model_path = resolve_path(output_dir, "als_model")
        try:
            save_model(als_model, model_path)
        except Exception as exc:
            log.warning(
                f"Skipping local model save (likely Windows/Hadoop issue): {exc}"
            )

        # 9. Optional baselines (SVD + KNN)
        baseline_metrics = {}
        if run_baselines:
            log.info("Training baseline models (SVD + KNN)...")
            from src.models.svd_baseline import evaluate_svd, train_svd
            from src.models.knn_baseline import evaluate_knn

            svd_model = train_svd(train, n_factors=50, n_epochs=20)
            svd_rmse, svd_mae = evaluate_svd(svd_model, test)
            baseline_metrics["svd"] = {"rmse": svd_rmse, "mae": svd_mae}

            knn_rmse, knn_mae = evaluate_knn(train, test, k=20)
            baseline_metrics["knn"] = {"rmse": knn_rmse, "mae": knn_mae}

        # 10. Save metrics report
        elapsed = time.time() - start
        report = {
            "dataset_stats":    stats,
            "user_activity":    eda_artifacts["user_activity"],
            "naive_baseline":   naive_metrics,
            "als_metrics":      als_metrics,
            "baseline_metrics": baseline_metrics,
            "config_used": {
                "rank":      config["als"]["rank"],
                "max_iter":  config["als"]["max_iter"],
                "reg_param": config["als"]["reg_param"],
                "min_user_ratings": config["preprocessing"]["min_user_ratings"],
            },
            "elapsed_seconds":  round(elapsed, 2),
        }

        metrics_path = Path(output_dir) / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"Saved metrics to: {metrics_path}")

        log.info(f"Pipeline complete in {elapsed:.1f}s")
        return 0

    except Exception as e:
        log.exception(f"Pipeline failed: {e}")
        return 1
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MovieLens recommendation pipeline (local)")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="YAML config",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Also train SVD and KNN baselines",
    )
    args = parser.parse_args()
    sys.exit(main(args.config, args.baselines))
