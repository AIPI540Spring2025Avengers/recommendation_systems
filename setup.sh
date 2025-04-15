#!/bin/bash

# Exit on error
set -e

echo "Setting up Hotel Recommendation System..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run dataset scripts
echo "Running dataset processing scripts..."
cd scripts/dataset

echo "Downloading Expedia dataset..."
python download_expedia.py

echo "Creating sample data..."
python create_sample_data.py

cd ../..

# Process data for traditional approach
echo "Processing data for traditional approach..."
cd scripts/traditional
python modeling.py --preprocess-only
cd ../..

# Process data for deep learning approach
echo "Processing data for deep learning approach..."
cd scripts/deep
python Data_Preprocess.py
cd ../..


# Run the Streamlit app
echo "Setup complete! You can now run the Streamlit app with:"
echo "streamlit run main.py"
