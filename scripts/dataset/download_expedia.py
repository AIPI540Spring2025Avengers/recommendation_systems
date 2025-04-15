import os
from kaggle.api.kaggle_api_extended import KaggleApi

# download the expedia hotel recommendations dataset
def download_expedia_data():
    """
    Download the Expedia hotel recommendations dataset.
    """
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Create directory if it doesn't exist
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Download the dataset
    print("Downloading Expedia hotel recommendations dataset...")
    api.competition_download_files('expedia-hotel-recommendations', path=raw_dir)
    
    # Unzip the downloaded files
    print("Extracting files...")
    os.system(f'unzip -o {os.path.join(raw_dir, "expedia-hotel-recommendations.zip")} -d {raw_dir}')
    
    print("Download complete!")

if __name__ == "__main__":
    download_expedia_data() 