CREATE TABLE IF NOT EXISTS studios (
    studio_id SERIAL PRIMARY KEY,
    name TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS anime (
    anime_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    synopsis TEXT,
    year INT,
    popularity INT,
    score FLOAT,
    studio_id INT REFERENCES studios(studio_id)
);