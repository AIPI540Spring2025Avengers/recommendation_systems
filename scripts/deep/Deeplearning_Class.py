import torch
import torch.nn as nn

class DeepUserTower(nn.Module):
    """
    Deep neural network for encoding user features into a 32-dimensional embedding space.
    This tower processes user-specific features like location, search parameters, and booking history.
    """
    def __init__(self):
        super(DeepUserTower, self).__init__()
        # Input layer: 10 user features -> 128 dimensions
        self.fc1 = nn.Linear(10, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        
        # Hidden layer 1: 128 -> 512 dimensions with dropout for regularization
        self.fc2 = nn.Linear(128, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)
        
        # Hidden layer 2: 512 -> 256 dimensions
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Hidden layer 3: 256 -> 128 dimensions
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        
        # Hidden layer 4: 128 -> 64 dimensions
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        
        # Output layer: 64 -> 32 dimensions (final embedding)
        self.fc6 = nn.Linear(64, 32)
    
    def forward(self, x):
        """
        Forward pass through the user tower network.
        Applies batch normalization and ReLU activation after each layer.
        Final output is L2 normalized for cosine similarity calculations.
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.fc6(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # L2 normalization for cosine similarity
        return x

class DeepHotelTower(nn.Module):
    """
    Deep neural network for encoding hotel features into a 32-dimensional embedding space.
    This tower processes hotel-specific features like location, amenities, and ratings.
    """
    def __init__(self):
        super(DeepHotelTower, self).__init__()
        # Input layer: 4 hotel features -> 64 dimensions
        self.fc1 = nn.Linear(4, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Hidden layer 1: 64 -> 256 dimensions with dropout for regularization
        self.fc2 = nn.Linear(64, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.1)
        
        # Hidden layer 2: 256 -> 128 dimensions
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Hidden layer 3: 128 -> 64 dimensions
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        # Output layer: 64 -> 32 dimensions (final embedding)
        self.fc5 = nn.Linear(64, 32)
    
    def forward(self, x):
        """
        Forward pass through the hotel tower network.
        Applies batch normalization and ReLU activation after each layer.
        Final output is L2 normalized for cosine similarity calculations.
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.fc5(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # L2 normalization for cosine similarity
        return x 