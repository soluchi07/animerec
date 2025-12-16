import os
from sqlalchemy import create_engine

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://anime_user:anime_pass@localhost:5432/anime_db"
)

engine = create_engine(DATABASE_URL)
