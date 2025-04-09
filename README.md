# Recommendation Systems

## Dataset Preprocessing

Preprocessing pipeline:

1. **Data Download and Loading**
   - downloads the hotels dataset from kaggle

2. **Data Cleaning and Transformation**
   - cleans column names by removing whitespace
   - removes duplicate entries
   - renames columns for consistency (e.g., 'countyName' to 'country')
   - converts star ratings from text to numeric values
   - removes invalid ratings (e.g., "All")
   - drops unnecessary columns (e.g., fax numbers, phone numbers)
   - extracts latitude and longitude from map coordinates

3. **Amenity Extraction**
   - processes hotel facilities text to extract standardized amenities
   - uses fuzzy matching to identify amenities from a predefined vocabulary
   - creates a list of parsed amenities for each hotel

4. **Data Storage**
   - saves raw data to `data/raw/hotels.csv`
   - saves cleaned and processed data to `data/processed/hotels_cleaned.csv`

## Naive Approach

Simple filtering and Ranking:

1. **Filtering Criteria**
   - location (city/country)
   - etc. 

2. **Ranking Method**
   - hotels are ranked based on their star rating
   - within the same star rating, rank my amenity count
