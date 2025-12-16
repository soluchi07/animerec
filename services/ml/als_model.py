# from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from .db import engine

def load_ratings():
    query = """
        SELECT user_id, anime_id, rating
        FROM user_ratings
        WHERE source = 'synthetic'
    """
    return pd.read_sql(query, engine)

def build_interaction_matrix(df):
    user_ids = df["user_id"].unique()
    anime_ids = df["anime_id"].unique()

    user_map = {u: i for i, u in enumerate(user_ids)}
    anime_map = {a: i for i, a in enumerate(anime_ids)}

    rows = df["user_id"].map(user_map)
    cols = df["anime_id"].map(anime_map)

    # Convert ratings → confidence
    # (rating 1-10 → confidence 0-1)
    confidence = df["rating"] / 10.0

    matrix = csr_matrix(
        (confidence, (rows, cols)),
        shape=(len(user_map), len(anime_map))
    )

    return matrix, user_map, anime_map

def train_als(interaction_matrix):
    model = AlternatingLeastSquares(
        factors=64,
        regularization=0.05,
        iterations=30,
        random_state=42
    )

    # implicit expects item-user matrix
    model.fit(interaction_matrix.T)

    return model

def recommend_for_user(model, interaction_matrix, user_map, anime_map, user_id, k=10):
    user_idx = user_map[user_id]

    scores, item_idxs = model.recommend(
        user_idx,
        interaction_matrix,
        N=k
    )

    inv_anime_map = {v: k for k, v in anime_map.items()}

    return [
        (inv_anime_map[i], float(s))
        for i, s in zip(item_idxs, scores)
    ]


def build_cf_model():
    ratings = load_ratings()
    matrix, user_map, anime_map = build_interaction_matrix(ratings)
    model = train_als(matrix)

    return {
        "model": model,
        "interaction_matrix": matrix,
        "user_map": user_map,
        "anime_map": anime_map,
    }

if __name__ == "__main__":
    artifacts = build_cf_model()
    print("ALS model trained")