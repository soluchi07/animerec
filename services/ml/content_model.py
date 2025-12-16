import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .db import engine

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
    
if __name__ == "__main__":
    model = build_content_model()
    print(f"Built content model for {len(model['anime_df'])} anime")
