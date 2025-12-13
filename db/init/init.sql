CREATE TABLE studios (
  studio_id SERIAL PRIMARY KEY,
  name TEXT UNIQUE
);

CREATE TABLE anime (
  anime_id SERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  alternate_titles TEXT[],
  synopsis TEXT,
  year INT,
  popularity INT,
  rating TEXT,
  score FLOAT,
  studio_id INT REFERENCES studios(studio_id),
);

CREATE TABLE genres (
  genre_id SERIAL PRIMARY KEY,
  name TEXT UNIQUE
);

CREATE TABLE anime_genres (
  anime_id INT REFERENCES anime(anime_id),
  genre_id INT REFERENCES genres(genre_id),
  PRIMARY KEY (anime_id, genre_id)
);

CREATE TABLE users (
  user_id SERIAL PRIMARY KEY,
  username TEXT UNIQUE,
  country TEXT
);

CREATE TABLE user_scores (
  user_id INT REFERENCES users(user_id),
  anime_id INT REFERENCES anime(anime_id),
  score FLOAT,
  timestamp TIMESTAMP,
  PRIMARY KEY (user_id, anime_id)
);
