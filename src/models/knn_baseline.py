"""Item-based K-Nearest Neighbors baseline using cosine similarity."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import setup_logger

log = setup_logger("knn_baseline")


def build_user_item_matrix(df):
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
    log.info(
        f"User-item matrix: shape={matrix.shape}, density={nnz_ratio:.4f}%"
    )
    return matrix, user_to_idx, movie_to_idx


def compute_item_similarity(matrix):
    log.info("Computing item-item cosine similarity...")
    item_matrix = matrix.T  # items x users
    sim = cosine_similarity(item_matrix, dense_output=False)
    log.info(f"Item similarity matrix shape: {sim.shape}")
    return sim


def predict_rating(user_idx, item_idx, user_item, item_sim, k=20, default=3.5):
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


def evaluate_knn(train_df, test_df, k=20):
    matrix, user_to_idx, movie_to_idx = build_user_item_matrix(train_df)
    item_sim = compute_item_similarity(matrix)

    test_pdf = test_df.select("userId", "movieId", "rating").toPandas()
    valid = test_pdf[
        test_pdf["userId"].isin(user_to_idx) &
        test_pdf["movieId"].isin(movie_to_idx)
    ].copy()
    log.info(f"Evaluating KNN on {len(valid):,} valid test rows")

    preds, truths = [], []
    for _, row in valid.iterrows():
        u_idx = user_to_idx[row["userId"]]
        m_idx = movie_to_idx[row["movieId"]]
        preds.append(predict_rating(u_idx, m_idx, matrix, item_sim, k=k))
        truths.append(row["rating"])

    preds, truths = np.array(preds), np.array(truths)
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    mae  = float(np.mean(np.abs(preds - truths)))
    log.info(f"KNN Test RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse, mae
