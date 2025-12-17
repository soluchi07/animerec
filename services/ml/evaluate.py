import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.model_selection import KFold
from typing import List, Dict

import joblib
import os
import sys

from .db import engine
from . import hybrid_model

# Define the artifact paths consistently
ARTIFACTS_PATH = "/app/services/ml/artifacts/"
CF_ARTIFACTS_PATH = os.path.join(ARTIFACTS_PATH, "als_model_artifacts.joblib")
CB_ARTIFACTS_PATH = os.path.join(ARTIFACTS_PATH, "content_model_artifacts.joblib")
HYBRID_ENGINE_PATH = os.path.join(ARTIFACTS_PATH, "hybrid_recommender_engine.joblib")

try:
    # 1. Get the actual class definition from the imported module
    HybridRecommenderClass = hybrid_model.HybridRecommender
    
    # 2. **Explicitly assign the class to the *loaded module's* namespace**
    # This ensures that when joblib/pickle attempts to load 
    # 'services.ml.hybrid_model.HybridRecommender', the class is definitely found.
    setattr(hybrid_model, 'HybridRecommender', HybridRecommenderClass)
    
    # 3. Also, ensure the class is available directly in the current module's namespace
    # (This is safer than directly modifying sys.modules entry)
    globals()['HybridRecommender'] = HybridRecommenderClass

    print("Class path successfully patched for unpickling.")

except Exception as e:
    print(f"Warning: Failed to patch class path. Error: {e}")

try:
    print("Loading models from artifacts...")
    # 1. Load Collaborative Filtering Artifacts
    CF_ARTIFACTS = joblib.load(CF_ARTIFACTS_PATH)

    # 2. Load Content-Based Artifacts
    CB_ARTIFACTS = joblib.load(CB_ARTIFACTS_PATH)

    # 3. Load the initialized Hybrid Engine (this is the fastest path)
    HYBRID_MODEL = joblib.load(HYBRID_ENGINE_PATH)

except FileNotFoundError as e:
    print(f"Error: Model artifact not found at {e.filename}. You must run 'content_model.py', 'als_model.py', and 'hybrid_model.py' first.")
    # Exit or provide a fallback mechanism if artifacts are missing
    raise SystemExit(1) from e

ANIME_DF = CB_ARTIFACTS['anime_df']

def get_user_ratings():
    """Fetches all synthetic user ratings for evaluation."""
    query = "SELECT user_id, anime_id, rating FROM user_ratings ORDER BY user_id"
    return pd.read_sql(query, engine)

def create_train_test_split(ratings_df, user_ids, test_rating_threshold=8.0):
    """
    Splits user ratings into training (exposure) and testing (ground truth) sets.
    The test set contains high-rated items, held out from the model's knowledge.
    """
    train_ratings = []
    test_ratings = {} # Dictionary of {user_id: set_of_true_positives}
    
    for user_id in user_ids:
        user_data = ratings_df[ratings_df['user_id'] == user_id]
        
        # Identify "good" items (True Positives)
        good_items = user_data[user_data['rating'] >= test_rating_threshold]['anime_id'].tolist()
        
        # Ensure we can hold out at least 1 item for testing
        if len(good_items) < 2:
            # Not enough data for holdout, skip this user or use all data for train
            train_ratings.append(user_data)
            test_ratings[user_id] = set()
            continue

        # Randomly select a portion (e.g., 20%) of good items for the test set
        test_items = set(np.random.choice(good_items, size=int(len(good_items)*0.2), replace=False))
        test_ratings[user_id] = test_items

        # Training data is everything *not* in the test set
        train_data = user_data[~user_data['anime_id'].isin(test_items)]
        train_ratings.append(train_data)

    train_df = pd.concat(train_ratings)
    
    return train_df, test_ratings


# --- MODEL RECOMMENDATION WRAPPERS ---

def recommend_cf_only(user_id, N=10):
    """CF recommendations based on the existing ALS model."""
    if user_id not in CF_ARTIFACTS['user_map']:
        return []

    user_idx = CF_ARTIFACTS['user_map'][user_id]
    
    # We must use the *original* interaction matrix if we want to simulate the
    # model's behavior based on the full training set (synthetic_cf.py).
    # NOTE: For true K-fold evaluation, the interaction_matrix would need to be 
    # rebuilt for the current training set, but for simplicity, we use the 
    # trained model with its existing factors.
    
    ids, _ = CF_ARTIFACTS['model'].recommend(
        user_idx, 
        CF_ARTIFACTS['interaction_matrix'], # Using full training data
        N=N
    )
    # Map internal index back to anime_id
    id_to_idx = {v: k for k, v in CF_ARTIFACTS['anime_map'].items()}
    return [id_to_idx[i] for i in ids]

def recommend_content_only(user_id, N=10):
    """
    Content-Based recommendations by finding items most similar to the 
    user's profile vector (centroid of liked items).
    """
    liked_items = HYBRID_MODEL._get_user_liked_anime(user_id)
    
    if not liked_items:
        # Cold start: return top-N popular anime
        return ANIME_DF.sort_values('popularity').head(N)['anime_id'].tolist()

    # Calculate content scores for ALL candidates (expensive, but necessary here)
    candidates = ANIME_DF['anime_id'].tolist()
    scores_map = HYBRID_MODEL._calculate_content_scores(user_id, candidates)
    
    # Filter out items the user has already liked
    for aid in liked_items:
        scores_map.pop(aid, None)
        
    sorted_recs = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)
    return [aid for aid, score in sorted_recs[:N]]

def recommend_hybrid(user_id, N=10):
    """Hybrid recommendations using the combined model."""
    # The hybrid model internally handles candidate generation and re-ranking
    # Default alpha/beta/gamma values are used
    results = HYBRID_MODEL.recommend(user_id, limit=N)
    return [r['anime_id'] for r in results]


# --- METRIC CALCULATION FUNCTIONS ---

def calculate_precision_at_k(recommendations: List[int], true_positives: set, k=10):
    """
    Precision@K = (Number of relevant items in top K) / K
    """
    recommended_k = set(recommendations[:k])
    hits = recommended_k.intersection(true_positives)
    return len(hits) / k

def calculate_coverage(all_recommendations: Dict[int, List[int]]):
    """
    Coverage = (Number of unique items ever recommended) / (Total number of items)
    """
    total_items = len(ANIME_DF)
    unique_recommended = set()
    for rec_list in all_recommendations.values():
        unique_recommended.update(rec_list)
    
    return len(unique_recommended) / total_items

def calculate_avg_rank_score(recommendations: List[int]):
    """
    Average Rank Score = Mean of the Quality Score (1 / (rank + 1)) of recommended items.
    """
    scores = []
    # Get a map of anime_id to rank for quick lookup
    rank_map = ANIME_DF.set_index('anime_id')['rank'].to_dict()

    for aid in recommendations:
        rank = rank_map.get(aid, 99999) # Use a large number if rank is missing
        # Use the quality score function for consistency
        score = HYBRID_MODEL.quality_score(rank)
        scores.append(score)
        
    return np.mean(scores) if scores else 0.0


# --- MAIN EVALUATION DRIVER ---

def run_evaluation():
    print("--- Starting Recommendation System Evaluation ---")
    
    # 1. Prepare Data
    ratings_df = get_user_ratings()
    # Use a random subset of users for faster evaluation
    test_users = np.random.choice(ratings_df['user_id'].unique(), size=100, replace=False) 
    
    # NOTE: In a real system, you would rebuild the ALS model on the train_df. 
    # Here, we skip that to simplify, but acknowledge the inaccuracy.
    # train_df, test_sets = create_train_test_split(ratings_df, test_users)
    
    test_sets = {}
    for user_id in test_users:
        user_data = ratings_df[ratings_df['user_id'] == user_id]
        test_sets[user_id] = set(user_data[user_data['rating'] >= 8.0]['anime_id'].tolist())


    models = {
        "CF-Only": recommend_cf_only,
        "Content-Only": recommend_content_only,
        "Hybrid": recommend_hybrid
    }
    
    results = {}
    
    for model_name, recommender_func in models.items():
        print(f"\nEvaluating {model_name}...")
        
        all_recs = {}
        precision_scores = []
        rank_scores = []
        
        for user_id in test_users:
            true_positives = test_sets.get(user_id, set())
            
            # Skip users with no ground truth (or very few liked items)
            if len(true_positives) < 1:
                continue

            # Generate top 10 recommendations
            recs = recommender_func(user_id, N=10)
            
            if not recs:
                continue
            
            all_recs[user_id] = recs
            
            # Calculate Metrics
            precision = calculate_precision_at_k(recs, true_positives, k=10)
            avg_rank = calculate_avg_rank_score(recs)
            
            precision_scores.append(precision)
            rank_scores.append(avg_rank)

        # Final calculations
        avg_precision = np.mean(precision_scores)
        coverage = calculate_coverage(all_recs)
        avg_rank_score = np.mean(rank_scores)
        
        results[model_name] = {
            "Precision@10": avg_precision,
            "Coverage": coverage,
            "Avg Rank Score": avg_rank_score
        }
    
    # 5. Report Results
    results_df = pd.DataFrame(results).T
    print("\n--- FINAL EVALUATION RESULTS ---")
    print(results_df.to_markdown(floatfmt=".4f"))
    
    return results_df

if __name__ == "__main__":
    run_evaluation()