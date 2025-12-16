import numpy as np
import pandas as pd

from .db import engine

def load_anime():
    query = """
        SELECT
            a.anime_id,
            a.rank,
            g.name AS genre
        FROM anime a
        JOIN anime_genres ag ON a.anime_id = ag.anime_id
        JOIN genres g ON ag.genre_id = g.genre_id
        WHERE a.rank IS NOT NULL
    """

    df = pd.read_sql(query, engine)
    
    return (
        df.groupby("anime_id")
        .agg({
            "rank": "first",
            "genre": lambda x: list(set(x))
        })
        .reset_index()
        .rename(columns={"genre": "genres"})
    )
    
def generate_synthetic_user_ratings(anime, all_genres, num_users=10_000, ratings_per_user=50):
    ratings = []

    for user_id in range(num_users):
        preferred_genres = np.random.choice(all_genres, size=3, replace=False)

        candidate_anime = anime[
            anime["genres"].apply(
                lambda gs: any(g in gs for g in preferred_genres)
            )
        ]

        sampled = candidate_anime.sample(ratings_per_user, replace=len(candidate_anime) < ratings_per_user)

        for _, row in sampled.iterrows():
            quality = 1 / (row["rank"] + 1)
            noise = np.random.normal(0, 0.3)
            rating = min(10, max(1, 7 + 3*quality + noise))

            ratings.append((user_id, row["anime_id"], rating, "synthetic"))
    
    return pd.DataFrame(
        ratings,
        columns=["user_id", "anime_id", "rating", "source"]
    )

        
        
if __name__ == "__main__":
    anime = load_anime()
    all_genres = sorted({g for genres in anime["genres"] for g in genres})
    
    ratings_df = generate_synthetic_user_ratings(anime, all_genres)
    
    ratings_df.to_sql(
        "user_ratings",
        engine,
        if_exists="append",
        index=False,
        method="multi"
    )

    print(f"Inserted {len(ratings_df)} synthetic ratings")
    print(ratings_df.head())

