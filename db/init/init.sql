-- Core Anime Table
CREATE TABLE anime (
  anime_id INT PRIMARY KEY,
  title TEXT NOT NULL,
  alternate_titles TEXT[],
  synopsis TEXT,
  year INT,
  type TEXT,
  rating TEXT,
  episodes INT,
  score FLOAT,
  rank INT,
  popularity INT,
  members INT,
  favorites INT,
);

-- Genres
CREATE TABLE genres (
  genre_id SERIAL PRIMARY KEY,
  name TEXT UNIQUE
);

CREATE TABLE anime_genres (
  anime_id INT REFERENCES anime(anime_id),
  genre_id INT REFERENCES genres(genre_id),
  PRIMARY KEY (anime_id, genre_id)
);

-- Studios
CREATE TABLE studios (
  studio_id INT PRIMARY KEY,
  name TEXT UNIQUE
);

CREATE TABLE anime_studios (
  anime_id INT REFERENCES anime(anime_id),
  studio_id INT REFERENCES studios(studio_id),
  PRIMARY KEY (anime_id, studio_id)
);

-- Users
CREATE TABLE users (
  user_id SERIAL PRIMARY KEY,
  user_type TEXT CHECK (user_type IN ('synthetic', 'implicit')),
  description TEXT
);

-- Implicit interactions
CREATE TABLE implicit_interactions (
  interaction_id SERIAL PRIMARY KEY,
  anime_id INT REFERENCES anime(anime_id),
  signal_type TEXT CHECK (
    signal_type IN ('rank', 'popularity', 'favorites', 'members')
  ),
  strength FLOAT
);

-- Synthetic ratings
CREATE TABLE user_ratings (
  user_id INT REFERENCES users(user_id),
  anime_id INT REFERENCES anime(anime_id),
  rating FLOAT CHECK (rating BETWEEN 1 AND 10),
  source TEXT CHECK (source IN ('synthetic', 'implicit')),
  PRIMARY KEY (user_id, anime_id)
);

-- Indexing
CREATE INDEX idx_anime_rank ON anime(rank);
CREATE INDEX idx_anime_popularity ON anime(popularity);
CREATE INDEX idx_anime_title ON anime(title);
CREATE INDEX idx_anime_genres_genre ON anime_genres(genre_id);
CREATE INDEX idx_implicit_anime ON implicit_interactions(anime_id);

COPY anime FROM '/docker-entrypoint-initdb.d/anime.csv' CSV HEADER;
COPY studios FROM '/docker-entrypoint-initdb.d/studios.csv' CSV HEADER;
COPY genres FROM '/docker-entrypoint-initdb.d/genres.csv' CSV HEADER;
COPY anime_genres FROM '/docker-entrypoint-initdb.d/anime_genres.csv' CSV HEADER;
COPY anime_studios FROM '/docker-entrypoint-initdb.d/anime_studios.csv' CSV HEADER;