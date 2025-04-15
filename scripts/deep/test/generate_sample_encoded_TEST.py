import pandas as pd
import numpy as np
import torch
import os
from ..Deeplearning_Class import DeepHotelTower
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout

def main():
    # Create sample hotel data
    sample_hotels = pd.DataFrame({
        'hotel_continent': ['Europe', 'Asia', 'North America', 'Europe', 'Asia'],
        'hotel_country': ['France', 'Japan', 'USA', 'Italy', 'China'],
        'hotel_market': ['Paris', 'Tokyo', 'New York', 'Rome', 'Beijing'],
        'hotel_cluster': [1, 2, 3, 4, 5]
    })
    
    # Encode categorical variables
    for col in ['hotel_continent', 'hotel_country', 'hotel_market']:
        sample_hotels[col] = sample_hotels[col].astype('category').cat.codes
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    model_path = os.path.join(models_dir, "hotel_tower.pth")
    
    # Add all necessary classes to safe globals
    torch.serialization.add_safe_globals([
        DeepHotelTower,
        Linear,
        BatchNorm1d,
        ReLU,
        Dropout,
        torch.nn.Module,
        torch.nn.functional.normalize
    ])
    
    # Load the entire model
    hotel_tower = torch.load(model_path, map_location=device)
    hotel_tower.eval()
    hotel_tower.to(device)
    
    # Convert data to tensor
    data_array = sample_hotels.values.astype(np.float32)
    data_tensor = torch.from_numpy(data_array).to(device)
    
    # Generate encoded vectors
    print("Generating encoded hotel data...")
    with torch.no_grad():
        encoded_vectors = hotel_tower(data_tensor).cpu().numpy()
    
    # Convert to DataFrame
    encoded_data = pd.DataFrame(encoded_vectors)
    
    # Save to CSV
    encoded_data.to_csv('sample_encoded_hotel_data.csv', index=False)
    print("Sample encoded hotel data saved to 'sample_encoded_hotel_data.csv'")
    
    # Print sample data for verification
    print("\nSample hotel data:")
    print(sample_hotels)
    print("\nEncoded vectors (first 5 dimensions):")
    print(encoded_data.iloc[:, :5])

if __name__ == "__main__":
    main() 