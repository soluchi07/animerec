import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import text

from .db import engine

import joblib
import os


ARTIFACTS_PATH = "/app/services/ml/artifacts/" 
MODEL_FILENAME = "hybrid_recommender_engine.joblib"

CF_ARTIFACTS_PATH = os.path.join(ARTIFACTS_PATH, "als_model_artifacts.joblib")
CB_ARTIFACTS_PATH = os.path.join(ARTIFACTS_PATH, "content_model_artifacts.joblib")

class HybridRecommender:
    def __init__(self):
        print("Loading Hybrid Recommender Components...")

        # 1. LOAD Collaborative Filtering Artifacts (DO NOT CALL build_cf_model)
        # Assuming the function build_cf_model() is now replaced by loading the joblib file
        try:
            self.cf_artifacts = joblib.load(CF_ARTIFACTS_PATH)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "CF Artifacts not found. Run als_model.py first."
            ) from e

        self.cf_model = self.cf_artifacts['model']
        self.user_map = self.cf_artifacts['user_map']
        self.item_map = self.cf_artifacts['anime_map']
        self.id_to_idx_cf = {v: k for k, v in self.item_map.items()}

        # 2. LOAD Content-Based Artifacts (DO NOT CALL build_content_model)
        try:
            self.cb_artifacts = joblib.load(CB_ARTIFACTS_PATH)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "CB Artifacts not found. Run content_model.py first."
            ) from e

        self.anime_df = self.cb_artifacts['anime_df']
        self.tfidf = self.cb_artifacts['tfidf']

        # 3. Rebuild TF-IDF Matrix for on-the-fly scoring
        print("Recomputing TF-IDF matrix for hybrid scoring...")
        self.tfidf_matrix = self.tfidf.transform(self.anime_df['synopsis'].fillna(""))

        # Map anime_id to matrix row index
        self.anime_id_to_matrix_idx = {
            row.anime_id: idx 
            for idx, row in self.anime_df.iterrows()
        }

    def quality_score(self, rank):
        """Signal 3.1: Inverse Rank (High rank = High quality)"""
        # Handle None or 0 ranks (unranked items)
        return 0.0 if not rank or rank <= 0 else 1.0 / (rank + 1)

    def exposure_penalty(self, popularity):
        """Signal 3.1: Inverse Popularity (High popularity = High penalty)"""
        return 0.0 if not popularity or popularity <= 0 else 1.0 / (popularity + 1)

    def hybrid_score(self, content_sim, cf_score, rank, alpha=0.6, beta=0.25, gamma=0.15):
        """
        Signal 3.2: Hybrid Scoring Function
        Note: Popularity is excluded here as it is used for penalties/filtering, 
        not as a positive reward signal in the primary score.
        """
        q_score = self.quality_score(rank)
        
        return (
            alpha * content_sim +
            beta * cf_score +
            gamma * q_score
        )
    
    def _get_user_liked_anime(self, user_id, threshold=7.0):
        """Fetch anime IDs derived from synthetic user ratings > threshold"""
        query = text("""
            SELECT anime_id 
            FROM user_ratings 
            WHERE user_id = :user_id AND rating >= :threshold
        """)
        with engine.connect() as conn:
            result = conn.execute(query, {"user_id": user_id, "threshold": threshold})
            return [row[0] for row in result]

    def _calculate_content_scores(self, user_id, candidate_ids):
        """
        Content Score = Cosine Similarity(User Profile Vector, Candidate Item Vector)
        User Profile Vector = Average of TF-IDF vectors of items the user liked.
        """
        liked_items = self._get_user_liked_anime(user_id)
        
        if not liked_items:
            # Cold start fallback: return 0.0 scores or popularity
            return {aid: 0.0 for aid in candidate_ids}

        # 1. Build User Profile Vector
        liked_indices = [
            self.anime_id_to_matrix_idx[aid] 
            for aid in liked_items 
            if aid in self.anime_id_to_matrix_idx
        ]
        
        if not liked_indices:
            return {aid: 0.0 for aid in candidate_ids}

        # Calculate centroid (average vector) of user's liked items
        user_profile_vec = np.mean(self.tfidf_matrix[liked_indices], axis=0)
        # Convert to numpy array if it's a matrix
        user_profile_vec = np.asarray(user_profile_vec).reshape(1, -1)

        # 2. Score Candidates
        scores = {}
        candidate_indices = []
        valid_candidate_ids = []

        for aid in candidate_ids:
            if aid in self.anime_id_to_matrix_idx:
                candidate_indices.append(self.anime_id_to_matrix_idx[aid])
                valid_candidate_ids.append(aid)
            else:
                scores[aid] = 0.0

        if candidate_indices:
            candidate_matrix = self.tfidf_matrix[candidate_indices]
            # Cosine similarity (dot product since vectors are normalized by TfidfVectorizer)
            sim_scores = (user_profile_vec @ candidate_matrix.T).flatten()
            
            for aid, score in zip(valid_candidate_ids, sim_scores):
                scores[aid] = float(score)

        return scores

    def _calculate_cf_scores(self, user_id, candidate_ids):
        """
        CF Score = Dot Product(User Factor, Item Factor) from ALS model
        """
        if user_id not in self.user_map:
            return {aid: 0.0 for aid in candidate_ids}

        user_idx = self.user_map[user_id]
        user_factors = self.cf_model.user_factors[user_idx]
        
        scores = {}
        for aid in candidate_ids:
            if aid in self.item_map:
                item_idx = self.item_map[aid]
                item_factors = self.cf_model.item_factors[item_idx]
                # Dot product
                score = np.dot(user_factors, item_factors)
                scores[aid] = float(score)
            else:
                scores[aid] = 0.0
        
        return scores

    def recommend(self, user_id, alpha=0.6, beta=0.25, gamma=0.15, limit=20):
        # 1. Generate Candidates (CF + Popular fallback)
        if user_id in self.user_map:
            user_idx = self.user_map[user_id]
            ids, _ = self.cf_model.recommend(user_idx, self.cf_artifacts['interaction_matrix'], N=50)
            candidates = [self.id_to_idx_cf[i] for i in ids]
        else:
            # Cold start: Use generic popular items
            candidates = self.anime_df.sort_values('popularity').head(50)['anime_id'].tolist()

        # 2. Compute Component Scores
        cf_scores_map = self._calculate_cf_scores(user_id, candidates)
        cb_scores_map = self._calculate_content_scores(user_id, candidates)

        # 3. Normalize Component Scores (MinMax)
        # We need these normalized before plugging into your weighted sum
        def normalize(score_map):
            if not score_map: return {}
            values = list(score_map.values())
            min_v, max_v = min(values), max(values)
            if max_v == min_v: return {k: 0.5 for k in score_map}
            return {k: (v - min_v) / (max_v - min_v) for k, v in score_map.items()}

        cf_norm = normalize(cf_scores_map)
        cb_norm = normalize(cb_scores_map)

        # 4. Final Scoring & Re-ranking (Section 3.3)
        scored_candidates = []
        
        for aid in candidates:
            # Fetch Metadata
            meta = self.anime_df[self.anime_df['anime_id'] == aid].iloc[0]
            rank = meta['rank']
            popularity = meta['popularity']

            # Get normalized component scores
            s_content = cb_norm.get(aid, 0.0)
            s_cf = cf_norm.get(aid, 0.0)

            # Calculate Raw Hybrid Score (Reward)
            raw_score = self.hybrid_score(s_content, s_cf, rank, alpha, beta, gamma)

            # Apply Exposure Penalty (Optional: You can subtract it or use it as a post-filter)
            # "Popularity is used... to avoid over-recommending blockbusters"
            # Here we apply it as a dampener to the final score
            penalty = self.exposure_penalty(popularity)
            
            # Example: Dampen highly popular items slightly to allow discovery
            # (Adjust logic based on how aggressive you want the penalty to be)
            final_score = raw_score * (1.0 + penalty) 

            scored_candidates.append({
                "anime_id": int(aid),
                "title": meta['title'],
                "final_score": final_score,
                "metrics": {
                    "content": s_content,
                    "collaborative": s_cf,
                    "quality_rank": self.quality_score(rank),
                    "exposure_penalty": penalty
                }
            })

        # Sort descending by final score
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        return scored_candidates[:limit]
    

def save_hybrid_engine(engine_instance, path):
    """Saves the initialized HybridRecommender instance."""
    try:
        joblib.dump(engine_instance, path)
        print(f"Successfully saved Hybrid Recommender Engine to {path}")
    except Exception as e:
        print(f"Error saving hybrid engine: {e}")

# Note: The singleton instance below is what the API would typically load.
# We are saving this instance.
recommender_engine = HybridRecommender()

if __name__ == "__main__":
    # If this script is run directly, it will build the engine (above) and save it.
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    
    # Save the initialized instance
    full_path = os.path.join(ARTIFACTS_PATH, MODEL_FILENAME)
    save_hybrid_engine(recommender_engine, full_path)