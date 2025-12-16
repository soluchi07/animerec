import pandas as pd
import numpy as np

import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .db import engine

ARTIFACTS_PATH = "/app/services/ml/artifacts/" 
MODEL_FILENAME = "content_model_artifacts.joblib"

def load_anime():
    query = """
        SELECT
            a.anime_id,
            a.title,
            a.synopsis,
            a.rank,
            a.popularity
        FROM anime a
        WHERE a.synopsis IS NOT NULL
    """
    return pd.read_sql(query, engine)

def build_tfidf_matrix(anime_df):
    tfidf = TfidfVectorizer(
        max_features=15000, # max_features=20000,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X = tfidf.fit_transform(anime_df["synopsis"].fillna(""))
    return X, tfidf

def precompute_neighbors(anime, X, k=10):
    # sim = cosine_similarity(X)
    sim = cosine_similarity(X, dense_output=True)
    anime_ids = anime["anime_id"].values
    neighbors = {}

    for idx in range(sim.shape[0]):
        # sims = similarity_matrix[idx]
        top_k = np.argsort(-sim[idx])[1 : k + 1]
        neighbors[anime_ids[idx]] = anime_ids[top_k].tolist()
        # neighbors[idx] = top_k.tolist()

    return neighbors

def build_content_model():
    anime = load_anime()
    X, tfidf = build_tfidf_matrix(anime)
    neighbors = precompute_neighbors(anime, X)

    return {
        "anime_df": anime,
        "tfidf": tfidf,
        "neighbors": neighbors,
    }


def save_model(model_artifacts, path):
    """Saves the entire content model dictionary to a file using joblib."""
    try:
        # joblib.dump is optimized for large numpy arrays and sklearn objects
        joblib.dump(model_artifacts, path)
        print(f"Successfully saved content model artifacts to {path}")
    except Exception as e:
        print(f"Error saving model artifacts: {e}")
        # Optionally, check if the directory exists and try to create it here
        # E.g., os.makedirs(os.path.dirname(path), exist_ok=True)
    
if __name__ == "__main__":
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    
    model = build_content_model()
    print(f"Built content model for {len(model['anime_df'])} anime")
    
    full_path = os.path.join(ARTIFACTS_PATH, MODEL_FILENAME)
    save_model(model, full_path)