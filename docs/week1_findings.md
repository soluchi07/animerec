# Week 1: Data Collection & Preprocessing

## Key Accomplishments

### Data Pipeline Setup
- Successfully ingested full MAL anime catalog
- Flattened and processed anime data from API calls
- Generated clean, normalized datasets:
  - `anime.csv` - Core anime metadata
  - `anime_genres.csv` - Genre relationships
  - `implicit_interactions.csv` - Derived engagement signals
  - `studios.csv` - Studio information

### Data Constraints Identified
- **No real user interaction histories available** - MyAnimeList user data identified as synthetic/unreliable
- This constraint fundamentally shapes our recommendation strategy

### Implicit Signal Design
Created three implicit interaction signals from anime metadata:
1. **Popularity** - Inverse popularity rank (lower rank = higher signal strength)
   - Formula: `1 / (popularity + 1)` 
   - Rationale: Chose popularity over arbitrary "rank" field as it's data-driven
2. **Favorites** - Raw favorite counts as engagement proxy
3. **Score** - User ratings as quality indicator

### Recommender Strategy Decision
**Content-based filtering as core approach** due to:
- Lack of genuine user interaction histories
- Rich anime metadata available (genres, studios, themes, scores)
- Implicit signals can supplement content similarity

This design decision impacts downstream models:
- Content similarity will drive primary recommendations
- Implicit signals provide popularity/quality weighting
- Enables cold-start handling for new users
- Future: Can incorporate real user data when available

### Exploratory Data Analysis
- Began initial EDA on anime dataset
- Identified and resolved schema mismatches
- Validated data quality across generated CSVs

### Tooling Improvements
- Upgraded CSV visualization workflow (VSCode extension better than Rainbow CSV)

## Deliverables
- [x] Full MAL anime catalog ingested  
- [x] Clean, normalized CSVs ready for database loading  
- [x] Implicit interaction signals created  
- [x] EDA initiated with initial insights  
- [x] Clear justification for content-based recommender strategy  

## Next Steps (Week 2)
- Design Postgres schema around processed data
- Bulk-load CSVs into database
- Create indexes for similarity searches and joins
- Validate data integrity post-load