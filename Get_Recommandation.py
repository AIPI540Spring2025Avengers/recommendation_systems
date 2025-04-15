import torch
from Function.Recommand_Deeplearning import hotel_data_encoder, get_topk_similar_hotels
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## First, load the model
user_tower = torch.load("Model/user_tower.pth", weights_only=False)
hotel_tower = torch.load("Model/hotel_tower.pth", weights_only=False)

# Then load the encoded hotel data
encoded_hotel_data = pd.read_csv('encoded_hotel_data.csv')

# Then load the hotel data
hotel_data = pd.read_csv('Hotel_data.csv')

# Then you get the input data from frontend, the data structure is like this:
# user_data = ['posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 
#             'orig_destination_distance', 'is_mobile', 'srch_ci', 'srch_co', 'srch_adults_cnt', 'srch_children_cnt', 
#             'srch_rm_cnt']

# Example input data
input_data = ['Europe', 'United Kingdom', 'London', 'London', 100, 0, '2024-01-01', '2024-01-02', 2, 0, 1]
# Change the input data to the number

# Then use the following code to get the recommandation:

# Feed the input data to the user tower
user_feats = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)  # (1,4)
user_vec = user_tower(user_feats).squeeze(0).cpu().numpy()

# Get the recommanded hotels
top20 = get_topk_similar_hotels(user_vec, encoded_hotel_data, hotel_data, topk=10000)

# Top20 is a dataframe, include the hotel_cluster, hotel_market, hotel_country, hotel_continent of the top20 recommanded hotels
# The order of the hotels is descendingly sorted by the similarity, the first one is the most recommanded one
