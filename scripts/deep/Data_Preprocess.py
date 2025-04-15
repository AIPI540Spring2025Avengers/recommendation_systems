import pandas as pd
from sklearn.model_selection import train_test_split
import os

def data_preprocess(Train_data):
    """
    Preprocesses hotel recommendation training data.
    
    Args:
        Train_data: Raw pandas DataFrame containing training data
        
    Returns:
        Four DataFrames: train_User_data, test_User_data, train_Hotel_data, test_Hotel_data
    """
    # Drop unnecessary columns
    drop_columns = ['date_time', 'site_name', 'user_id', 'is_package', 'channel', 'cnt', 'srch_destination_id', 'srch_destination_type_id']
    Train_data = Train_data.drop(drop_columns, axis=1)

    # Filter for only booking data (is_booking = 1)
    Train_data = Train_data[Train_data['is_booking'] == 1]

    # Define user feature columns
    user_columns = ['posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 
                    'orig_destination_distance', 'is_mobile', 'srch_ci', 'srch_co', 'srch_adults_cnt', 'srch_children_cnt', 
                    'srch_rm_cnt']

    # Define hotel feature columns
    hotel_columns = ['hotel_continent', 'hotel_country', 'hotel_market', 'hotel_cluster']

    # Split data into user and hotel features
    User_data = Train_data[user_columns]
    Hotel_data = Train_data[hotel_columns]

    # Define categorical columns for encoding
    cat_cols_user = ['posa_continent', 'user_location_country', 'user_location_region', 'user_location_city']
    cat_cols_hotel = ['hotel_continent', 'hotel_country', 'hotel_market']

    # Encode categorical variables
    for col in cat_cols_user:
        User_data[col] = User_data[col].astype('category').cat.codes
    for col in cat_cols_hotel:
        Hotel_data[col] = Hotel_data[col].astype('category').cat.codes

    # Convert date columns to datetime format
    User_data['srch_ci'] = pd.to_datetime(User_data['srch_ci'])
    User_data['srch_co'] = pd.to_datetime(User_data['srch_co'])

    # Calculate reservation duration in days
    User_data['Reservation_Time'] = (User_data['srch_co'] - User_data['srch_ci']).dt.days

    # Drop date columns after creating duration feature
    User_data = User_data.drop(['srch_ci', 'srch_co'], axis=1)

    # Drop rows with missing values
    User_data.dropna(inplace=True)
    Hotel_data.dropna(inplace=True)

    # Ensure user and hotel data have matching indices
    common_index = User_data.index.intersection(Hotel_data.index)
    User_data_common = User_data.loc[common_index]
    Hotel_data_common = Hotel_data.loc[common_index]
    User_data = User_data_common
    Hotel_data = Hotel_data_common

    # Split data into train and test sets
    train_User_data, test_User_data = train_test_split(User_data, test_size=0.1, random_state=99)
    train_Hotel_data, test_Hotel_data = train_test_split(Hotel_data, test_size=0.1, random_state=99)
    
    # Save the hotel data
    processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "processed")
    hotel_data = pd.concat([train_Hotel_data, test_Hotel_data])
    hotel_data.to_csv(os.path.join(processed_data_dir, "Hotel_data.csv"), index=False)
    print("Saved hotel data to Hotel_data.csv")
    
    return train_User_data, test_User_data, train_Hotel_data, test_Hotel_data