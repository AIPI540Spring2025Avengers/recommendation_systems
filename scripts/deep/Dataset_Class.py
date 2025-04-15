import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DoubleTowerDataset(Dataset):
    """
    Dataset class for the double tower recommendation model.
    
    This dataset assumes that user_df and hotel_df have the same number of rows,
    with each row forming a positive sample pair of a user's features and the corresponding hotel features.
    The input data should already be preprocessed, including encoding of categorical variables.
    """
    def __init__(self, user_df: pd.DataFrame, hotel_df: pd.DataFrame):
        # Convert DataFrames to numpy arrays
        self.user_data = user_df.values
        self.hotel_data = hotel_df.values
        assert self.user_data.shape[0] == self.hotel_data.shape[0], "User and hotel data must have the same number of rows"
    
    def __len__(self):
        return self.user_data.shape[0]
    
    def __getitem__(self, idx):
        user_features = self.user_data[idx]      # shape: (10,)
        hotel_features = self.hotel_data[idx]    # shape: (4,)
        # Convert numpy arrays to torch.Tensor
        return torch.tensor(user_features, dtype=torch.float), torch.tensor(hotel_features, dtype=torch.float)