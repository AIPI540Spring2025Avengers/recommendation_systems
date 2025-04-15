import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .Double_Tower import DeepUserTower, DeepHotelTower
from .Dataset_Class import DoubleTowerDataset
import os
def model_evaluation(Selected_Hotel_data, true_hotel_data):
    """
    Evaluates similarity between selected hotel data and true hotel data.
    
    Args:
        Selected_Hotel_data: DataFrame with 20 rows and 32 columns of selected hotel features
        true_hotel_data: DataFrame with 1 row and 32 columns of true hotel features
        
    Returns:
        float: Ratio of hotels with similarity above 0.9 threshold
    """
    # Define the path to the models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_tower = torch.load(os.path.join(models_dir, "user_tower.pth"), weights_only=False)
    hotel_tower = torch.load(os.path.join(models_dir, "hotel_tower.pth"), weights_only=False)

    user_tower.eval()
    hotel_tower.eval()

    # Convert true_hotel_data (1 row) to tensor
    true_hotel_tensor = torch.tensor(true_hotel_data.values, dtype=torch.float32).to(device)
    
    # Process true hotel data through hotel tower
    with torch.no_grad():
        true_hotel_embedding = hotel_tower(true_hotel_tensor)
    
    # Define cosine similarity function
    cos_sim = nn.CosineSimilarity(dim=1)
    
    # Track similarities and threshold count
    similarities = []
    above_threshold_count = 0
    threshold = 0.9
    total_count = len(Selected_Hotel_data)
    
    # Process each selected hotel
    with torch.no_grad():
        # Convert Selected_Hotel_data to tensor
        selected_hotels_tensor = torch.tensor(Selected_Hotel_data.values, dtype=torch.float32).to(device)
        
        # Get embeddings for all selected hotels
        selected_hotels_embeddings = hotel_tower(selected_hotels_tensor)
        
        # Calculate similarity between each selected hotel and the true hotel
        for i in range(total_count):
            similarity = cos_sim(selected_hotels_embeddings[i:i+1], true_hotel_embedding)
            sim_value = similarity.item()
            similarities.append(sim_value)
            
            if sim_value > threshold:
                above_threshold_count += 1
    
    # Calculate ratio of hotels above threshold
    ratio = above_threshold_count / total_count
    
    print(f"Hotels with similarity > {threshold}: {above_threshold_count}/{total_count}")
    print(f"Ratio: {ratio:.4f}")
    
    return ratio