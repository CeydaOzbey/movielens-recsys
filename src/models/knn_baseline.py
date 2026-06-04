"""
Item-based K-Nearest Neighbors baseline using cosine similarity.

For each (user, item) pair in the test set, predict the rating by:
  1. Finding the K most similar items (by cosine similarity of their rating vectors)
     that the user has already rated.
  2. Taking a similarity-weighted average of the user's ratings for those K items.

Like the SVD baseline, this is a SINGLE-MACHINE method. It pulls the rating
matrix into a SciPy sparse matrix, so it does not scale to 25M ratings.
"""
from __future__ import annotations

import numpy as np
from pyspark.sql import DataFrame
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import setup_logger

logger = setup_logger("knn_baseline")


def build_user_item_matrix(df: DataFrame) -> tuple[csr_matrix, dict, dict]:
    """
    Build a sparse user-item matrix and return it along with the
    user_id -> row_index and movie_id -> col_index mappings.
    """
    pdf = df.select("userId", "movieId", "rating").toPandas()

    user_ids  = sorted(pdf["userId"].unique())
    movie_ids = sorted(pdf["movieId"].unique())
    user_to_idx  = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

    rows = pdf["userId"].map(user_to_idx).values
    cols = pdf["movieId"].map(movie_to_idx).values
    data = pdf["rating"].values

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    )

    nnz_ratio = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100
    logger.info(
        f"User-item matrix: shape={matrix.shape}, density={nnz_ratio:.4f}%"
    )
    return matrix, user_to_idx, movie_to_idx


def compute_item_similarity(matrix: csr_matrix):
    """Cosine similarity between all item pairs (sparse output)."""
    logger.info("Computing item-item cosine similarity...")
    item_matrix = matrix.T  # items x users
    sim = cosine_similarity(item_matrix, dense_output=False)
    logger.info(f"Item similarity matrix shape: {sim.shape}")
    return sim


def predict_rating(
    user_idx: int,
    item_idx: int,
    user_item: csr_matrix,
    item_sim,
    k: int = 20,
    default: float = 3.5,
) -> float:
    """Predict the rating that `user_idx` would give to `item_idx`."""
    user_ratings = user_item[user_idx].toarray().flatten()
    rated_idx = np.where(user_ratings > 0)[0]

    if len(rated_idx) == 0:
        return default

    sims = item_sim[item_idx, rated_idx].toarray().flatten()
    if len(sims) > k:
        top_k = np.argsort(sims)[-k:]
        top_sims = sims[top_k]
        top_ratings = user_ratings[rated_idx[top_k]]
    else:
        top_sims = sims
        top_ratings = user_ratings[rated_idx]

    if top_sims.sum() == 0:
        return default
    return float(np.dot(top_sims, top_ratings) / top_sims.sum())


def evaluate_knn(
    train_df: DataFrame,
    test_df: DataFrame,
    k: int = 20,
) -> tuple[float, float]:
    """Build matrix, compute similarities, evaluate on the test set."""
    matrix, user_to_idx, movie_to_idx = build_user_item_matrix(train_df)
    item_sim = compute_item_similarity(matrix)

    test_pdf = test_df.select("userId", "movieId", "rating").toPandas()
    valid = test_pdf[
        test_pdf["userId"].isin(user_to_idx) &
        test_pdf["movieId"].isin(movie_to_idx)
    ].copy()
    logger.info(f"Evaluating KNN on {len(valid):,} valid test rows")

    preds, truths = [], []
    for _, row in valid.iterrows():
        u_idx = user_to_idx[row["userId"]]
        m_idx = movie_to_idx[row["movieId"]]
        preds.append(predict_rating(u_idx, m_idx, matrix, item_sim, k=k))
        truths.append(row["rating"])

    preds, truths = np.array(preds), np.array(truths)
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    mae  = float(np.mean(np.abs(preds - truths)))
    logger.info(f"KNN Test RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse, mae
