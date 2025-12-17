#!/bin/bash
set -e

echo "Waiting for PostgreSQL database to start..."
until pg_isready -h postgres -p 5432 -U anime_user; do
  echo "Postgres is unavailable - sleeping"
  sleep 2
done

echo "PostgreSQL is up and running!"

echo "Starting Synthetic Data Generation..."

# Run ETL steps (assuming these populate core tables)
# python -m services.etl.fetch_anime
# python -m services.etl.clean_anime
# python -m services.etl.build_implicit_signals

# Generate synthetic users and ratings
python -m services.ml.synthetic_cf

# echo "Data preparation complete. Starting model training..."

# --- 3. Run Model Training Scripts ---

# Train Content-Based Model and save artifacts
# python -m services.ml.content_model

# Train Collaborative Filtering Model and save artifacts
# python -m services.ml.als_model

# Build Hybrid Model (which loads and combines the above) and save artifact
# python -m services.ml.hybrid_model

# echo "All ML models trained and artifacts saved."

# --- 4. Keep container running (Crucial for a service) ---
# This ensures the container stays alive after the setup finishes
# You can replace this with whatever command you need to run your API/server.
# If this container is only for training, you might exit, but if it serves models, you need a server here.
# For now, we keep it alive for observation:
tail -f /dev/null