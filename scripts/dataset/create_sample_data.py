import pandas as pd
import os

# create a sample dataset for testing the recommendation system in streamlit app
def create_sample_data():
    """Create a sample dataset for testing the recommendation system."""
    # Load the full training data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
    train_df = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))
    
    # Take a random sample of 1000 rows
    sample_df = train_df.sample(n=1000, random_state=42)
    
    # Save the sample data
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
    os.makedirs(sample_dir, exist_ok=True)
    sample_df.to_csv(os.path.join(sample_dir, "sample_train.csv"), index=False)
    
    print(f"Created sample dataset with {len(sample_df)} rows")

if __name__ == "__main__":
    create_sample_data() 