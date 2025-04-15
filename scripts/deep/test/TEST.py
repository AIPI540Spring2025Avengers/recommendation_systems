import torch
import pandas as pd
import os
from ..Recommand_Deeplearning import get_topk_similar_hotels, hotel_data_encoder
from ..Deeplearning_Class import DeepUserTower
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout

def encode_categorical(value, categories):
    """Encode a categorical value to its corresponding index."""
    try:
        return categories.index(value)
    except ValueError:
        return 0  # Return 0 for unknown categories

def preprocess_user_data(user_data):
    """Preprocess user data similar to Data_Preprocess.py"""
    # Define categories for encoding
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    countries = ['United Kingdom', 'France', 'Germany', 'Italy', 'Spain', 'United States', 'China', 'Japan']
    regions = ['London', 'Paris', 'Berlin', 'Rome', 'Madrid', 'New York', 'Beijing', 'Tokyo']
    cities = ['London', 'Paris', 'Berlin', 'Rome', 'Madrid', 'New York', 'Beijing', 'Tokyo']
    
    # Encode categorical variables
    encoded_data = {
        'posa_continent': encode_categorical(user_data['posa_continent'], continents),
        'user_location_country': encode_categorical(user_data['user_location_country'], countries),
        'user_location_region': encode_categorical(user_data['user_location_region'], regions),
        'user_location_city': encode_categorical(user_data['user_location_city'], cities),
        'orig_destination_distance': user_data['orig_destination_distance'],
        'is_mobile': user_data['is_mobile'],
        'srch_adults_cnt': user_data['srch_adults_cnt'],
        'srch_children_cnt': user_data['srch_children_cnt'],
        'srch_rm_cnt': user_data['srch_rm_cnt'],
        'Reservation_Time': (pd.to_datetime(user_data['srch_co']) - pd.to_datetime(user_data['srch_ci'])).days
    }
    
    return encoded_data

def get_user_vector(user_data, user_tower, device):
    """Convert user data to a vector using the user tower model."""
    # Preprocess user data
    processed_data = preprocess_user_data(user_data)
    
    # Convert to tensor and move to device
    user_feats = torch.tensor(list(processed_data.values()), dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get user vector
    with torch.no_grad():
        user_vec = user_tower(user_feats).squeeze(0).cpu().numpy()
    
    return user_vec

def main():
    # Set device
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Add necessary classes to safe globals
    torch.serialization.add_safe_globals([
        DeepUserTower,
        Linear,
        BatchNorm1d,
        ReLU,
        Dropout,
        torch.nn.Module,
        torch.nn.functional.normalize
    ])
    
    # Load model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    user_tower = torch.load(
        os.path.join(models_dir, "user_tower.pth"),
        map_location=device,
        weights_only=False
    )
    user_tower.eval()
    
    # Load and encode hotel data
    hotel_data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "Hotel_data.csv"))
    encoded_hotel_data = hotel_data_encoder(hotel_data)
    
    # Example user input data
    user_data = {
        'posa_continent': 'Europe',
        'user_location_country': 'United Kingdom',
        'user_location_region': 'London',
        'user_location_city': 'London',
        'orig_destination_distance': 100,
        'is_mobile': 0,
        'srch_ci': '2024-01-01',
        'srch_co': '2024-01-02',
        'srch_adults_cnt': 2,
        'srch_children_cnt': 0,
        'srch_rm_cnt': 1
    }
    
    # Get user vector
    user_vec = get_user_vector(user_data, user_tower, device)
    
    # Get recommendations
    recommendations = get_topk_similar_hotels(user_vec, encoded_hotel_data, hotel_data, topk=5)
    
    # Print recommendations
    print("\nTop 5 Recommended Hotels:")
    print(recommendations[['hotel_cluster', 'hotel_market', 'hotel_country', 'hotel_continent']])

if __name__ == "__main__":
    main() 