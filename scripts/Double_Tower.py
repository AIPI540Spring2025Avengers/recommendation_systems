import torch
import torch.nn as nn
import torch.optim as optim

# User Tower: 10-dimensional input, 32-dimensional output, with multiple hidden layers and regularization
class DeepUserTower(nn.Module):
    """
    Deep neural network for encoding user features into an embedding space.
    Takes 10-dimensional user features and outputs a 32-dimensional normalized embedding vector.
    """
    def __init__(self):
        super(DeepUserTower, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(128, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)  # 10% dropout rate
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.fc6 = nn.Linear(64, 32)
    
    def forward(self, x):
        """
        Forward pass through the user tower.
        
        Args:
            x: Input tensor of shape [batch_size, 10]
            
        Returns:
            L2-normalized output tensor of shape [batch_size, 32]
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
        # L2 normalization to make vector length 1, helps with cosine similarity calculations
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

# Hotel Tower: 4-dimensional input, 32-dimensional output, with multiple layers and regularization
class DeepHotelTower(nn.Module):
    """
    Deep neural network for encoding hotel features into an embedding space.
    Takes 4-dimensional hotel features and outputs a 32-dimensional normalized embedding vector.
    """
    def __init__(self):
        super(DeepHotelTower, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(64, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.fc5 = nn.Linear(64, 32)
    
    def forward(self, x):
        """
        Forward pass through the hotel tower.
        
        Args:
            x: Input tensor of shape [batch_size, 4]
            
        Returns:
            L2-normalized output tensor of shape [batch_size, 32]
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
        x = nn.functional.normalize(x, p=2, dim=1)
        return x