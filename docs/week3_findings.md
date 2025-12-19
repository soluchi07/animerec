# Week 3 Findings — Recommendation Models & Evaluation

## Overview

Week 3 focused on implementing and validating a multi-stage recommendation system consisting of:

* A content-based recommender (TF-IDF + cosine similarity)
* A collaborative filtering model (ALS via `implicit`)
* A hybrid recommender combining both signals

The emphasis was on correct experimental staging, scalability, and clean separation between data generation, modeling, and persistence.
<![Recommendation Architecture Diagram](./reccommendation_arch.png)>
---

## Content-Based Recommender

### Baseline Strategy

A text-only TF-IDF baseline was implemented first to validate semantic similarity before introducing additional feature dimensions.

Key design choices:

* Genre fusion and persistence were intentionally deferred to avoid premature dimensionality growth
* A full (O(N^2)) similarity matrix was not computed or stored
* Only top-K similar neighbors per anime were precomputed

This approach allows semantic validity to be confirmed before scaling feature complexity.

---

### Implementation Details

* TF-IDF vectors are built from anime synopses
* Cosine similarity is used to compute nearest neighbors
* Only anime with non-null synopses are included
* Rank and popularity are retained for analysis and explainability, not as model features

The model persists only the top-K neighbors rather than the full similarity matrix.

---

## Collaborative Filtering (ALS)

### Modeling Approach

Collaborative filtering was implemented using Alternating Least Squares (ALS) from the `implicit` library.

Key characteristics:

* Ratings are treated as implicit feedback and converted to confidence weights
* The interaction matrix is constructed as **items × users**, as required by `implicit`
* Explicit mappings are maintained between real IDs and matrix indices

This setup allows efficient learning over a large sparse interaction matrix.

---

## Synthetic Data Generation

### Data Generation Strategy

Synthetic user-anime interactions were generated using:

* Genre affinity
* Rank-based quality signals
* Controlled random noise

The generator:

* Produces a DataFrame rather than writing directly to the database
* Is isolated from model training logic

This separation allows data generation and persistence to evolve independently.

---

### Schema Design

The `user_ratings` table includes a `source` column to distinguish:

* `synthetic` training data
* future `implicit` or real user feedback

This enables incremental dataset replacement without schema changes.

---

## Hybrid Recommender

### Score Normalization

The hybrid model combines two fundamentally different scoring signals:

* Collaborative Filtering: unbounded confidence scores
* Content-Based: cosine similarity in `[0, 1]`

To prevent dominance of one signal:

* Min-Max normalization is applied per user
* Both signals are clamped to `[0, 1]` before weighted combination

---

### User Profile Construction

Because synthetic users do not have precomputed text profiles:

* A user content centroid is built dynamically
* Synopses of anime rated above a threshold are retrieved
* TF-IDF vectors are averaged to form a user profile vector
* Candidate anime are scored against this centroid

---

### Candidate Selection Optimization

Scoring all items for every user is computationally expensive.

To reduce cost:

* A candidate pool (Top 50) is selected primarily from the CF model
* Content-based scoring is applied only to this candidate set

This significantly reduces compute while preserving ranking quality.

---

## Model Integration

The hybrid recommender:

* Reuses the database engine from `db.py`
* Reuses TF-IDF artifacts and anime metadata from `content_model.py`
* Maintains consistent feature representations across models

`implicit` API constraints are respected:

* `userids` and `user_items` must align row-for-row
* Even single-user recommendations require array-like inputs

---

## Evaluation

### Evaluation Setup

Evaluation was performed using synthetic user-anime interactions derived from:

* Genre affinity
* Rank-based quality priors

Ground truth relevance was defined using per-user top-N rated items.

---

### Observations

* Absolute Precision@10 values are low due to intentionally noisy synthetic data
* Coverage remains stable across models
* The hybrid recommender consistently outperforms both content-only and CF-only baselines

These results confirm correct integration, normalization, and score fusion behavior.

---

## Persisted Artifacts

The following artifacts were saved for downstream use:

* `content_neighbors.pkl`
* `tfidf.pkl`
* `cf_model.pkl`
* `hybrid_config.json`

These artifacts are consumed by the API layer in Week 4.

---

## Summary

Week 3 delivered:

* A validated content-based baseline
* A scalable ALS collaborative filtering model
* A normalized hybrid recommender
* Clean persistence and evaluation workflows

The system is designed to prioritize cold-start recommendations using content similarity while leveraging collaborative filtering to improve diversity and ranking quality.
