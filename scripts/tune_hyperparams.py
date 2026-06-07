"""ALS hyperparameter sensitivity sweep — rank, regParam, maxIter."""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

from src.data_loader import load_ratings
from src.preprocessing import preprocess
from src.utils import get_spark_session, load_config, setup_logger

log = setup_logger("tune_hyperparams")

RANK_SWEEP    = [5, 10, 20, 50, 100]
REG_SWEEP     = [0.01, 0.05, 0.1, 0.2, 0.5]
MAX_ITER_SWEEP = [5, 10, 15, 20]

DEFAULT_RANK     = 10
DEFAULT_REG      = 0.1
DEFAULT_MAX_ITER = 10


def build_als(config, rank, reg_param, max_iter):
    cfg = config["als"]
    return ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        userCol=cfg["user_col"],
        itemCol=cfg["item_col"],
        ratingCol=cfg["rating_col"],
        coldStartStrategy=cfg["cold_start_strategy"],
        nonnegative=cfg.get("nonnegative", True),
        implicitPrefs=False,
        seed=42,
    )


def measure_rmse(model, test, rating_col):
    preds = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol=rating_col,
        predictionCol="prediction",
    )
    return round(float(evaluator.evaluate(preds)), 4)


def run_sweep(train, test, config, param_name, values, defaults):
    rating_col = config["als"]["rating_col"]
    results = {}
    for v in values:
        rank     = v if param_name == "rank"     else defaults["rank"]
        reg      = v if param_name == "reg"      else defaults["reg"]
        max_iter = v if param_name == "max_iter" else defaults["max_iter"]

        log.info(f"  Fitting: rank={rank}, regParam={reg}, maxIter={max_iter}")
        model = build_als(config, rank, reg, max_iter).fit(train)
        rmse  = measure_rmse(model, test, rating_col)
        log.info(f"  -> RMSE = {rmse:.4f}")
        results[str(v)] = rmse
    return results


def main(config_path):
    config = load_config(config_path)
    spark  = get_spark_session(config)
    log.info(f"Spark {spark.version} | master: {spark.sparkContext.master}")

    try:
        ratings      = load_ratings(spark, config)
        train, test  = preprocess(ratings, config)

        defaults = {"rank": DEFAULT_RANK, "reg": DEFAULT_REG, "max_iter": DEFAULT_MAX_ITER}

        log.info("=== Sweep 1: rank ===")
        rank_results = run_sweep(train, test, config, "rank", RANK_SWEEP, defaults)

        log.info("=== Sweep 2: regParam ===")
        reg_results = run_sweep(train, test, config, "reg", REG_SWEEP, defaults)

        log.info("=== Sweep 3: maxIter ===")
        iter_results = run_sweep(train, test, config, "max_iter", MAX_ITER_SWEEP, defaults)

        output = {
            "rank_sweep": rank_results,
            "reg_sweep":  reg_results,
            "iter_sweep": iter_results,
        }

        out_path = PROJECT_ROOT / "results" / "hyperparam_sensitivity.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        log.info(f"Results saved to: {out_path}")
        log.info("=== Final Results ===")
        log.info(f"rank_sweep:  {rank_results}")
        log.info(f"reg_sweep:   {reg_results}")
        log.info(f"iter_sweep:  {iter_results}")
        return 0

    except Exception as e:
        log.exception(f"Sweep failed: {e}")
        return 1
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALS hyperparameter sensitivity sweep")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    sys.exit(main(args.config))
