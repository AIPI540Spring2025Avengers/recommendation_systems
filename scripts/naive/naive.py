import pandas as pd
import os

# this is the naive recommendation system, simple sort by rating and amenities
# depending on world, country, or city.


# load hotels data
def load_hotels_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    processed_dir = os.path.join(data_dir, "processed")
    hotels = pd.read_csv(os.path.join(processed_dir, "hotels_cleaned.csv"), encoding="latin1")
    return hotels

# sort hotels by rating and amenities
def sort_by_rating_and_amenities(df, top_n):
    df = df.copy()  # new dataframe
    # count the number of amenities for each hotel
    df['AmenityCount'] = df['ParsedAmenities'].apply(len)
    # sort by rating and amenities
    sorted_df = df.sort_values(by=['HotelRating', 'AmenityCount'], ascending=[False, False])
    return sorted_df.drop(columns=['AmenityCount']).head(top_n)


# get top 10 hotels globally
def get_top_hotels_global(hotels, top_n=10):
    return sort_by_rating_and_amenities(hotels, top_n)[['HotelName', 'city', 'country', 'HotelRating', 'ParsedAmenities']]

# get top hotels in a city
def get_top_hotels_by_city(hotels, city_name, top_n=5):
    # filter by city
    city_hotels = hotels[hotels['city'].str.lower() == city_name.lower()]
    # if no hotels, return empty dataframe
    if city_hotels.empty:
        return pd.DataFrame()
    return sort_by_rating_and_amenities(city_hotels, top_n)[['HotelName', 'HotelRating', 'city', 'country', 'ParsedAmenities']]

# get top hotels in a country
def get_top_hotels_by_country(hotels, country_name, top_n=5):
    # filter by country 
    country_hotels = hotels[hotels['country'].str.lower() == country_name.lower()]
    # if no hotels, return empty dataframe
    if country_hotels.empty:
        return pd.DataFrame()
    return sort_by_rating_and_amenities(country_hotels, top_n)[['HotelName', 'HotelRating', 'city', 'country', 'ParsedAmenities']]

# main function
def main():
    hotels = load_hotels_data()

    # get top 10 hotels globally
    print("Top 10 Hotels Globally:")
    print(get_top_hotels_global(hotels))

    # get top 5 hotels in a city
    city_query = "Cannes"
    print(f"\nTop Hotels in {city_query}")
    print(get_top_hotels_by_city(hotels, city_query))

    # get top 5 hotels in a country
    country_query = "France"
    print(f"\nTop Hotels in {country_query}")
    print(get_top_hotels_by_country(hotels, country_query))

if __name__ == "__main__":
    main()
