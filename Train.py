import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Function.Double_Tower import DeepUserTower, DeepHotelTower
from Function.Dataset_Class import DoubleTowerDataset
from matplotlib import pyplot as plt
from Function.Data_Preprocess import data_preprocess

def train_model(Train_data):
    """
    Train the double tower recommendation model.
    
    Args:
        Train_data: DataFrame containing raw training data
        
    Returns:
        Tuple containing trained user_tower and hotel_tower models
    """
    # Create Dataset and DataLoader
    batch_size = 256
    train_User_data, test_User_data, train_Hotel_data, test_Hotel_data = data_preprocess(Train_data)
    dataset = DoubleTowerDataset(train_User_data, train_Hotel_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Use GPU for training if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Initialize models and move to device
    user_tower = DeepUserTower().to(device)
    hotel_tower = DeepHotelTower().to(device)

    # Define cosine similarity and loss function: using MSELoss to make positive sample pairs have similarity close to 1
    cos_sim = nn.CosineSimilarity(dim=1)
    loss_fn = nn.MSELoss()

    # Define optimizer: update parameters of both towers together
    optimizer = optim.Adam(list(user_tower.parameters()) + list(hotel_tower.parameters()), lr=0.0005)

    # Set number of training epochs
    num_epochs = 5

    # For recording average loss per epoch
    epoch_loss_list = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, (user_features, hotel_features) in enumerate(dataloader):
            user_features = user_features.to(device)
            hotel_features = hotel_features.to(device)
            
            # Forward pass
            user_output = user_tower(user_features)
            hotel_output = hotel_tower(hotel_features)
            
            similarity = cos_sim(user_output, hotel_output)
            loss = loss_fn(similarity, torch.ones_like(similarity))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.16f}")
        
        avg_loss = epoch_loss / num_batches
        epoch_loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}")

    torch.save(user_tower, "user_tower.pth")
    torch.save(hotel_tower, "hotel_tower.pth")
    print("Training completed")

    # Plot and save the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs+1), epoch_loss_list, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig('training_loss_curve.png', dpi=300)
    plt.show()
    return user_tower, hotel_tower