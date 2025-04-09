import os
import re
import pandas as pd
from rapidfuzz import process
from functools import lru_cache
from kagglehub import dataset_download

# Amenity vocabulary
AMENITY_VOCAB = {
    "wifi", "air conditioning", "pool", "gym", "private parking", "parking", "beach", "tv",
    "concierge", "restaurant", "room service", "airport pickup", "laundry", "shuttle",
    "kids play area", "balcony", "free parking", "pet friendly", "spa", "breakfast",
    "security", "atm", "terrace", "ironing service", "television", "office", "valet", "24-hours",
    "currency exchange", "bar", "bike rental", "computer station", "electric car charging station",
    "playground", "meeting rooms", "garden"
}

# Fuzzy matcher cache
@lru_cache(maxsize=None)
def cached_match(phrase):
    match = process.extractOne(phrase, AMENITY_VOCAB, score_cutoff=80)
    return match[0].title() if match else None

# extract amenities from text
def extract_amenities(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    # generate all possible phrases from tokens
    phrases = (
        tokens +
        [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)] +
        [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    )
    # find matches in vocabulary
    found = set()
    for phrase in phrases:
        result = cached_match(phrase)
        if result:
            found.add(result)
    # return list of found amenities
    return list(found)

# download and load raw data
def download_and_load_raw_data():
    print("\nDownloading dataset from Kaggle...")
    dataset_path = dataset_download("raj713335/tbo-hotels-dataset")
    raw_csv_path = os.path.join(dataset_path, "hotels.csv")
    # load raw data
    hotels = pd.read_csv(raw_csv_path, encoding="latin1")
    # clean column names
    return hotels

def clean_hotels_data(hotels):
    print("\nCleaning and transforming data...")
    # clean column names    
    hotels.columns = hotels.columns.str.strip()
    # drop duplicates
    hotels = hotels.drop_duplicates()
    # rename columns
    hotels.rename(columns={'countyName': 'country', 'cityName': 'city'}, inplace=True)
    # convert star ratings to numeric values
    hotels['HotelRating'] = hotels['HotelRating'].replace({
        'OneStar': 1, 'TwoStar': 2, 'ThreeStar': 3, 'FourStar': 4, 'FiveStar': 5
    })
    # drop invalid ratings
    hotels = hotels[hotels['HotelRating'] != "All"]
    # drop unnecessary columns
    hotels.drop(columns=[
        'countyCode', 'FaxNumber', 'PhoneNumber', 'PinCode', 'HotelWebsiteUrl', 'cityCode', 'HotelCode'
    ], inplace=True, errors='ignore')
    if 'Map' in hotels.columns:
        hotels[['Latitude', 'Longitude']] = hotels['Map'].str.split('|', expand=True)
        hotels.drop(columns=['Map'], inplace=True)
    return hotels

# save dataframe to csv
def save_dataframe(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved to {path}")

# main function to run the script
def main():
    # download and load raw data
    hotels = download_and_load_raw_data()
    # save raw data
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
    # save raw data
    print("\nSaving raw data...")
    save_dataframe(hotels, os.path.join(raw_dir, "hotels.csv"))

    # clean and transform data
    hotels_cleaned = clean_hotels_data(hotels)
    # extract amenities
    print("\nExtracting amenities...")
    hotels_cleaned['ParsedAmenities'] = hotels_cleaned['HotelFacilities'].apply(extract_amenities)
    # save cleaned data
    
    print("\nSaving cleaned data...")
    save_dataframe(hotels_cleaned, os.path.join(processed_dir, "hotels_cleaned.csv"))

    print("\nDone! Cleaned data saved.")

if __name__ == "__main__":
    main()
