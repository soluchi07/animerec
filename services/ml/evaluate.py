import os
import joblib
import sys
import pandas as pd
import numpy as np
from typing import List, Dict
from scipy.sparse import csr_matrix

from .db import engine
from . import hybrid_model

ARTIFACTS_PATH = "/app/services/ml/artifacts/"
CF_ARTIFACTS_PATH = os.path.join(ARTIFACTS_PATH, "als_model_artifacts.joblib")
CB_ARTIFACTS_PATH = os.path.join(ARTIFACTS_PATH, "content_model_artifacts.joblib")
HYBRID_ENGINE_PATH = os.path.join(ARTIFACTS_PATH, "hybrid_recommender_engine.joblib")


def load_artifacts():  # sourcery skip: use-contextlib-suppress
    cf = joblib.load(CF_ARTIFACTS_PATH)
    cb = joblib.load(CB_ARTIFACTS_PATH)
    # Ensure HybridRecommender class is available in this module's namespace
    # so joblib/pickle can find it when unpickling an instance saved earlier.
    try:
        HybridRecommenderClass = hybrid_model.HybridRecommender
        setattr(hybrid_model, 'HybridRecommender', HybridRecommenderClass)
        globals()['HybridRecommender'] = HybridRecommenderClass
        # Also attach to this module object explicitly
        sys.modules[__name__].HybridRecommender = HybridRecommenderClass # type: ignore
    except Exception:
        pass

    hybrid = joblib.load(HYBRID_ENGINE_PATH)
    return cf, cb, hybrid


def get_user_ratings():
    query = "SELECT user_id, anime_id, rating FROM user_ratings ORDER BY user_id"
    return pd.read_sql(query, engine)


def precision_at_k(recommendations: List[int], true_positives: set, k: int = 10) -> float:
    if not recommendations:
        return 0.0
    recommended_k = recommendations[:k]
    hits = len([r for r in recommended_k if r in true_positives])
    return hits / k


def coverage_metric(all_recommendations: Dict[int, List[int]], total_items: int) -> float:
    if total_items == 0:
        return 0.0
    unique = set()
    for recs in all_recommendations.values():
        unique.update(recs)
    return len(unique) / total_items


def avg_rank_quality(recommendations: List[int], anime_df: pd.DataFrame, hybrid_engine) -> float:
    if not recommendations:
        return 0.0
    rank_map = anime_df.set_index('anime_id')['rank'].to_dict()
    scores = []
    for aid in recommendations:
        rank = rank_map.get(aid, None)
        if rank is None:
            continue
        scores.append(hybrid_engine.quality_score(rank))
    return float(np.mean(scores))


def recommend_cf(cf_artifacts, user_id, N=10):
    user_map = cf_artifacts['user_map']
    anime_map = cf_artifacts['anime_map']
    model = cf_artifacts['model']
    matrix = cf_artifacts['interaction_matrix']

    if user_id not in user_map:
        return []

    user_idx = user_map[user_id]
    # Compute scores directly from factor matrices to avoid shape mismatches
    try:
        user_factors = model.user_factors[user_idx]
        item_factors = model.item_factors
        # scores per item = item_factors dot user_factors
        scores = item_factors @ user_factors
        top_idxs = np.argsort(-scores)[:N]
        inv_map = {v: k for k, v in anime_map.items()}
        return [int(inv_map[int(i)]) for i in top_idxs]
    except Exception:
        return []


def recommend_content(cb_artifacts, hybrid_engine, user_id, N=10):
    anime_df = cb_artifacts['anime_df']
    candidates = anime_df['anime_id'].tolist()
    scores_map = hybrid_engine._calculate_content_scores(user_id, candidates)
    # remove items user already liked
    liked = set(hybrid_engine._get_user_liked_anime(user_id))
    for l in liked:
        scores_map.pop(l, None)
    sorted_items = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)
    return [int(aid) for aid, _ in sorted_items[:N]]


def recommend_hybrid(hybrid_engine, user_id, N=10):
    results = hybrid_engine.recommend(user_id, limit=N)
    return [int(r['anime_id']) for r in results]


def run_evaluation(n_users: int = 100):
    np.random.seed(42)
    cf, cb, hybrid = load_artifacts()
    anime_df = cb['anime_df']

    ratings = get_user_ratings()
    cf_users = set(cf['user_map'].keys())
    all_user_ids = ratings['user_id'].unique()
    eligible = list(cf_users.intersection(set(all_user_ids)))
    if not eligible:
        raise SystemExit('No eligible users for evaluation')

    n_eval = min(n_users, len(eligible))
    test_users = np.random.choice(eligible, size=n_eval, replace=False)

    # build ground-truth sets (ratings >= 8.0)
    test_sets = {}
    top_k = 20
    for uid in test_users:
        u = ratings[ratings['user_id'] == uid]
        test_sets[uid] = set(
            u.sort_values("rating", ascending=False)
            .head(top_k)["anime_id"]#.tolist()
        )

    models = {
        'CF-Only': lambda uid: recommend_cf(cf, uid, N=10),
        'Content-Only': lambda uid: recommend_content(cb, hybrid, uid, N=10),
        'Hybrid': lambda uid: recommend_hybrid(hybrid, uid, N=10)
    }

    results = {}

    for name, fn in models.items():
        precisions = []
        avg_ranks = []
        all_recs = {}

        for uid in test_users:
            true_pos = test_sets.get(uid, set())
            if len(true_pos) < 1:
                continue
            recs = fn(uid)
            if not recs:
                continue
            all_recs[uid] = recs
            precisions.append(precision_at_k(recs, true_pos, k=10))
            avg_ranks.append(avg_rank_quality(recs, anime_df, hybrid))

        avg_prec = float(np.mean(precisions)) if precisions else 0.0
        cov = coverage_metric(all_recs, total_items=len(anime_df))
        avg_rank = float(np.mean(avg_ranks)) if avg_ranks else 0.0

        results[name] = {
            'Precision@10': avg_prec,
            'Coverage': cov,
            'Avg Rank Score': avg_rank
        }

    df = pd.DataFrame(results).T
    print(df.to_markdown(floatfmt='.4f'))
    return df


if __name__ == '__main__':
    run_evaluation()