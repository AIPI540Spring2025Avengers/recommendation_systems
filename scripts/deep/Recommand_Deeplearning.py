import numpy as np
import torch
import pandas as pd
import os

def hotel_data_encoder(hotel_data):
    """
    Encodes hotel data into 32-dimensional vectors using the trained hotel tower model.
    
    Args:
        hotel_data: DataFrame containing hotel features
        
    Returns:
        DataFrame with encoded 32-dimensional vectors for each hotel
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Process data in batches for efficient forward propagation
    batch_size = 256
    encoded_list = []
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    hotel_tower = torch.load(
        os.path.join(models_dir, "hotel_tower.pth"),
        map_location=device,
        weights_only=False
    )
    
    hotel_tower.eval()
    hotel_tower.to(device)
    # Extract numerical values from DataFrame as NumPy array
    data_array = hotel_data.values.astype(np.float32)  # shape = (n, 4)
    n_rows = data_array.shape[0]

    with torch.no_grad():  # Disable gradient calculation to save memory
        for start_idx in range(0, n_rows, batch_size):
            end_idx = start_idx + batch_size
            batch_data = data_array[start_idx:end_idx]  # shape = (batch_size, 4)
            
            # Convert to PyTorch tensor
            batch_tensor = torch.from_numpy(batch_data)
            batch_tensor = batch_tensor.to(device)
            # Forward pass -> output (batch_size, 32)
            batch_output = hotel_tower(batch_tensor)
            
            # Convert back to NumPy and collect
            encoded_list.append(batch_output.cpu().numpy())

    # Concatenate all batch outputs => shape = (n, 32)
    encoded_array = np.concatenate(encoded_list, axis=0)

    # Convert to DataFrame with 32 columns
    encoded_hotel_data = pd.DataFrame(encoded_array)

    # save the encoded hotel data
    # processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
    # encoded_hotel_data.to_csv(os.path.join(processed_data_dir, "encoded_hotel_data.csv"), index=False)
    # print("Saved encoded hotel data to encoded_hotel_data.csv")
    return encoded_hotel_data


def get_topk_similar_hotels(user_vec, encoded_hotel, hotel_data, topk=20):
    """
    Finds the top-k most similar hotels for a given user vector.
    
    Args:
        user_vec: numpy vector of shape [32] representing encoded user preferences
        encoded_hotel: DataFrame or numpy array of shape [M, 32] with encoded hotel features
        hotel_data: DataFrame containing hotel information, with same number of rows as encoded_hotel
        topk: number of most similar hotels to return (default: 20)
        
    Returns:
        DataFrame with topk most similar hotels based on cosine similarity
    """
    # Calculate cosine similarity: cos_sim = (u·h) / (||u||*||h||)
    # Simplified here by using numpy
    hotel_array = encoded_hotel.values  # shape = (M, 32)
    
    dot_products = hotel_array.dot(user_vec)          # shape = (M, )
    similarities = dot_products

    # argsort helps with sorting, selecting top k highest values
    # np.argsort is ascending by default, using negative to get highest similarities first
    topk_idx = np.argsort(-similarities)[:topk]
    
    # Return corresponding rows from hotel_data
    return hotel_data.iloc[topk_idx]