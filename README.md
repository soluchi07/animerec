# Anime Recommendation Engine with Scalable Database + ML Layer

A full-stack recommendation system demonstrating data modeling, machine learning, and scalable API design using anime data.

## 🎯 Project Purpose

This project showcases enterprise-level skills in:
- **Database Design**: Normalized schema (3NF), strategic indexing, and query optimization
- **Machine Learning**: Hybrid recommendation system combining content-based and collaborative filtering
- **API Development**: RESTful service with caching and performance considerations
- **Data Engineering**: ETL pipeline for large-scale dataset ingestion
- **System Design**: Tradeoff analysis between SQL/NoSQL, caching strategies, and scalability patterns

Built for technical interviews and portfolio demonstrations, this project tells a compelling story about handling real-world data challenges in a domain with broad appeal.

## 🏗️ Architecture Overview

```
Data Sources (AniList/MAL API)
         ↓
    ETL Pipeline
         ↓
   PostgreSQL Database (normalized schema)
         ↓
   ML Recommendation Engine
   (Content + Collaborative)
         ↓
    FastAPI REST Service
    (with Redis caching)
         ↓
  Web Dashboard (Streamlit/React)
```

## 📋 TODO

### Phase 1: Data Foundation
- [ ] Set up PostgreSQL database
- [ ] Design normalized schema (3NF)
  - [ ] Create `Anime`, `Studio`, `Genre`, `AnimeGenre` tables
  - [ ] Create `User`, `UserAnimeRating` tables
  - [ ] Add strategic indexes (user_id, anime_id, GIN for genres)
- [ ] Create ER diagram and document normalization decisions
- [ ] Build ETL pipeline
  - [ ] Fetch data from AniList/MAL API (or load dataset)
  - [ ] Clean and transform data
  - [ ] Populate database with validation

### Phase 2: ML Recommendation Engine
- [ ] **Content-Based Filtering**
  - [ ] Extract and vectorize anime synopses (TF-IDF)
  - [ ] Create genre one-hot encodings
  - [ ] Generate studio embeddings
  - [ ] Implement cosine similarity ranking
- [ ] **Collaborative Filtering**
  - [ ] Implement using Surprise library (KNNBaseline/SVD)
  - [ ] Alternative: Build matrix factorization from scratch
  - [ ] Train on user-anime rating data
- [ ] Create hybrid recommendation strategy
- [ ] Write Jupyter notebooks documenting experiments

### Phase 3: API Layer
- [ ] Build FastAPI service with endpoints:
  - [ ] `GET /recommend?user_id=XX` - personalized recommendations
  - [ ] `GET /similar?anime_id=XX` - similar anime
  - [ ] `GET /anime/{id}` - anime details
  - [ ] `POST /rate` - submit user rating
- [ ] Add request validation and error handling
- [ ] Implement Redis caching for hot recommendations
- [ ] Add logging and monitoring
- [ ] Write API documentation (OpenAPI/Swagger)

### Phase 4: Visualization & Demo
- [ ] Build interactive dashboard (Streamlit or React)
  - [ ] Display personalized recommendations
  - [ ] Show anime similarity networks
  - [ ] Visualize user preference clustering
  - [ ] Explain recommendation reasoning
- [ ] Create data visualizations (popularity trends, genre distributions)
- [ ] Record demo video or create live deployment

### Phase 5: Performance & Analysis
- [ ] Benchmark query performance with different index strategies
- [ ] Compare SQL vs NoSQL retrieval speeds
- [ ] Profile ML model inference times
- [ ] Load test API endpoints
- [ ] Document performance findings

### Phase 6: Documentation
- [ ] Write technical report covering:
  - [ ] Data modeling rationale (why 3NF, which indexes)
  - [ ] ML algorithm comparison (content vs collaborative)
  - [ ] Caching strategy justification
  - [ ] Scalability considerations
  - [ ] Future improvements
- [ ] Create setup/deployment guide
- [ ] Document API usage with examples
- [ ] Add inline code comments and docstrings

### Bonus Features
- [ ] Implement A/B testing framework for recommendation algorithms
- [ ] Add real-time recommendation updates with WebSockets
- [ ] Create genre trend analysis over time
- [ ] Build user preference learning system
- [ ] Add seasonal anime tracking
- [ ] Implement collaborative filtering cold-start solutions

## 🛠️ Technology Stack

**Database**: PostgreSQL (with optional MongoDB comparison)  
**Backend**: Python, FastAPI, SQLAlchemy  
**ML**: scikit-learn, Surprise, pandas, numpy  
**Caching**: Redis  
**Frontend**: Streamlit / React  
**Deployment**: Docker, Docker Compose  
**Testing**: pytest, locust (load testing)

## 📊 Expected Deliverables

1. **GitHub Repository** with clean, documented code
2. **Schema Diagrams** showing relationships and indexes
3. **ETL Scripts** for reproducible data pipeline
4. **ML Notebooks** with experimentation and evaluation
5. **REST API** with comprehensive documentation
6. **Live Demo** (Streamlit app or deployed web service)
7. **Technical Writeup** explaining design decisions

## 🎓 Interview Talking Points

- "Implemented hybrid recommendation using TF-IDF and collaborative filtering, improving cold-start performance by X%"
- "Optimized PostgreSQL queries with strategic indexing, reducing recommendation latency from Xms to Yms"
- "Designed normalized schema following 3NF principles while balancing query performance"
- "Built scalable API with Redis caching layer, supporting Z concurrent users"

## 🚀 Getting Started

```bash
# Clone repository
git clone https://github.com/yourusername/anime-recommendation-engine.git
cd anime-recommendation-engine

# Setup instructions coming soon
# See docs/SETUP.md for detailed installation guide
```

## 📝 License

MIT

---

**Status**: 🚧 In Development  
**Last Updated**: December 2025