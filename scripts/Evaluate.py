import torch
import pandas as pd
import numpy as np
import os
from Data_Preprocess import data_preprocess
from Recommand_Deeplearning import get_topk_similar_hotels
from Model_Evaluation import model_evaluation
from Recommand_Deeplearning import hotel_data_encoder

def evaluate_recommendation_system():
    """
    Evaluates the recommendation system on test data:
    1. Gets test data using data_preprocess
    2. Encodes hotel test data with the hotel_tower model
    3. For each user, gets top 20 recommended hotels
    4. Evaluates similarity between recommendations and true hotels
    5. Returns average accuracy
    """
    print("Loading test data...")
    # Get preprocessed test data
    _, test_User_data, _, test_Hotel_data = data_preprocess(pd.read_csv("data/train.csv"))
    
    # Load user model from different possible paths
    model_paths = ["user_tower.pth", "Model/user_tower.pth"]
    user_tower = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for path in model_paths:
        if os.path.exists(path):
            user_tower = torch.load(path, weights_only=False)
            print(f"Loaded user model from: {path}")
            break
    
    if user_tower is None:
        raise FileNotFoundError("Could not find user_tower.pth in any of the expected locations")
    
    user_tower.eval()
    user_tower.to(device)
    
    print(f"Test data loaded: {len(test_User_data)} user records and {len(test_Hotel_data)} hotel records")
    
    # Encode all hotels in test set
    print("Encoding hotel data...")
    encoded_hotel_data = hotel_data_encoder(test_Hotel_data)
    
    # Process each user
    print("Evaluating recommendations...")
    total_accuracy = 0.0
    valid_users = 0
    
    # Process in batches to speed up evaluation
    batch_size = 32
    n_users = len(test_User_data)
    
    with torch.no_grad():
        for start_idx in range(0, n_users, batch_size):
            end_idx = min(start_idx + batch_size, n_users)
            batch_user_data = test_User_data.iloc[start_idx:end_idx]
            batch_user_tensor = torch.tensor(batch_user_data.values, dtype=torch.float32).to(device)
            
            # Get user embeddings
            batch_user_embeddings = user_tower(batch_user_tensor)
            
            for i in range(len(batch_user_data)):
                user_idx = start_idx + i
                user_vec = batch_user_embeddings[i].cpu().numpy()
                
                # Get true hotel data for this user
                true_hotel_data = test_Hotel_data.iloc[[user_idx]]
                
                # Get top-20 recommended hotels based on user vector
                recommended_hotels = get_topk_similar_hotels(user_vec, encoded_hotel_data, test_Hotel_data, topk=20)
                
                try:
                    # Encode true hotel and recommended hotels for evaluation
                    true_encoded = hotel_data_encoder(true_hotel_data)
                    recommended_encoded = hotel_data_encoder(recommended_hotels)
                    
                    # Evaluate similarity
                    accuracy = model_evaluation(recommended_encoded, true_encoded)
                    total_accuracy += accuracy
                    valid_users += 1
                    
                    if (valid_users % 10) == 0:
                        print(f"Processed {valid_users} users. Current average accuracy: {total_accuracy/valid_users:.4f}")
                except Exception as e:
                    print(f"Error processing user {user_idx}: {e}")
                    
                # Limit processing to first 100 users for faster evaluation
                if valid_users >= 100:
                    break
            
            if valid_users >= 100:
                break
    
    # Calculate average accuracy
    if valid_users > 0:
        avg_accuracy = total_accuracy / valid_users
        print(f"\nEvaluation complete. Average accuracy: {avg_accuracy:.4f}")
        print(f"Number of valid users evaluated: {valid_users}")
        return avg_accuracy
    else:
        print("No valid users were evaluated.")
        return 0.0

if __name__ == "__main__":
    evaluate_recommendation_system()
