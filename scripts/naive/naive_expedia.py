import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# load the data
def load_expedia_data():
    """Load the Expedia dataset from the raw data directory."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    return train_df

# get the most popular hotels
# naive approach - just recommend the most booked hotels overall
def get_most_popular_hotels(hotels, n=5):
    """
    Get the n most popular hotel clusters based on frequency.
    """
    # Drop rows with missing hotel_cluster values
    hotels = hotels.dropna(subset=['hotel_cluster'])
    return hotels['hotel_cluster'].value_counts().head(n).index.tolist()

def mapk(actual, predicted, k=5):
    """
    Computes the Mean Average Precision at k (MAP@k) for a list of users.
    actual: List of lists, where each inner list contains hotel clusters booked by a user
    predicted: List of predicted hotel clusters (same for all users in naive approach)
    """
    def apk(actual, predicted, k):
        if len(predicted) > k:
            predicted = predicted[:k]
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score / min(len(actual), k)

    # Convert single predicted list to list of lists for each user
    predicted = [predicted] * len(actual)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def evaluate_naive_recommender(train_df, k_values=[5]):
    """
    Evaluate the naive recommender using MAP@k metrics.
    Uses training data split for evaluation.
    
    Args:
        train_df: DataFrame containing training data
        k_values: List of k values to evaluate MAP@k (default [5] to match ML.py)
    """
    # Split the training data into train and validation sets
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Get the most popular hotels from the training set
    popular_hotels = get_most_popular_hotels(train_data, n=max(k_values))
    
    # Group actual bookings by user
    actual_bookings = val_data.groupby('user_id')['hotel_cluster'].apply(list).values
    
    # Calculate MAP@k for each k value
    map_scores = {}
    for k in k_values:
        map_scores[f"MAP@{k}"] = mapk(actual_bookings, popular_hotels, k=k)
    
    return map_scores

# main function
def main():
    # Load the data
    print("Loading Expedia datasets...")
    train_df = load_expedia_data()
    
    # Get top 5 most popular hotels
    print("\nTop 5 Most Popular Hotel Clusters")
    popular_hotels = get_most_popular_hotels(train_df, n=5)
    print(popular_hotels)
    
    # Evaluate the naive recommender
    print("\nEvaluating Naive Recommender...")
    map_scores = evaluate_naive_recommender(train_df, k_values=[5])
    print("\nMAP@k Scores:")
    for k, score in map_scores.items():
        print(f"{k}: {score:.4f}")

if __name__ == "__main__":
    main()

# this script was refined using GPT-4o to ensure the MAP@k score is calculated correctly.