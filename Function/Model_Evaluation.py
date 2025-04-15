import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Double_Tower import DeepUserTower, DeepHotelTower
from Dataset_Class import DoubleTowerDataset

def model_evaluation(test_User_data, test_Hotel_data):
    """
    Evaluates the performance of the trained model using cosine similarity metrics.
    
    Args:
        test_User_data: User features in the test dataset
        test_Hotel_data: Hotel features in the test dataset
        
    Returns:
        float: Average cosine similarity across the test set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_tower = torch.load("user_tower.pth", weights_only=False)
    hotel_tower = torch.load("hotel_tower.pth", weights_only=False)

    user_tower.eval()
    hotel_tower.eval()

    # Create Dataset and DataLoader
    batch_size = 256
    test_dataset = DoubleTowerDataset(test_User_data, test_Hotel_data)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define cosine similarity function
    cos_sim = nn.CosineSimilarity(dim=1)

    total_similarity = 0.0  # For accumulating similarity across all samples
    total_samples = 0       # For counting total samples

    with torch.no_grad():  # Disable gradient calculation for faster inference
        for batch_idx, (user_features, hotel_features) in enumerate(test_dataloader):
            # Move data to the same device as the model
            user_features = user_features.to(device)
            hotel_features = hotel_features.to(device)

            # Forward pass to get user and hotel vectors
            user_output = user_tower(user_features)     # [batch_size, embed_dim]
            hotel_output = hotel_tower(hotel_features)  # [batch_size, embed_dim]

            # Calculate cosine similarity for current batch
            similarity = cos_sim(user_output, hotel_output)  # [batch_size]

            # Accumulate similarity and sample count
            total_similarity += similarity.sum().item()
            total_samples += similarity.size(0)

    # Calculate average similarity across test set
    avg_similarity = total_similarity / total_samples
    print(f"Average cosine similarity on test set: {avg_similarity:.4f}")
    
    return avg_similarity