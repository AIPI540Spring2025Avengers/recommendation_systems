 # Hotel Recommendation System

A deep learning-based hotel recommendation system that uses a dual-tower (two-tower) neural network architecture to match users with suitable hotels based on their preferences and hotel features.

## Project Overview

This project implements a hotel recommendation system using a dual-tower neural network architecture. The system processes user and hotel data separately through two specialized neural networks (towers), creating embeddings that capture the essence of user preferences and hotel characteristics. These embeddings are then used to compute similarity scores, enabling personalized hotel recommendations.

## Model Architecture

### Double Tower Neural Network

The core of the recommendation system is a Double Tower neural network architecture, which consists of two separate neural networks:

#### User Tower

- **Input**: 10-dimensional vector of user features
- **Output**: 32-dimensional normalized embedding vector
- **Architecture**:
  - Layer 1: Linear(10 → 128) → BatchNorm → ReLU
  - Layer 2: Linear(128 → 512) → BatchNorm → ReLU → Dropout(0.1)
  - Layer 3: Linear(512 → 256) → BatchNorm → ReLU
  - Layer 4: Linear(256 → 128) → BatchNorm → ReLU
  - Layer 5: Linear(128 → 64) → BatchNorm → ReLU
  - Layer 6: Linear(64 → 32)
  - L2 Normalization: Normalizes the output vector to unit length

#### Hotel Tower

- **Input**: 4-dimensional vector of hotel features
- **Output**: 32-dimensional normalized embedding vector
- **Architecture**:
  - Layer 1: Linear(4 → 64) → BatchNorm → ReLU
  - Layer 2: Linear(64 → 256) → BatchNorm → ReLU → Dropout(0.1)
  - Layer 3: Linear(256 → 128) → BatchNorm → ReLU
  - Layer 4: Linear(128 → 64) → BatchNorm → ReLU
  - Layer 5: Linear(64 → 32)
  - L2 Normalization: Normalizes the output vector to unit length

### Key Design Features

1. **Separate Towers**: By processing user and hotel data through separate networks, each tower can specialize in capturing the intrinsic patterns of its domain.

2. **Deep Architecture**: Multiple layers with gradually decreasing widths allow the model to learn hierarchical representations.

3. **Regularization Techniques**:
   - Batch Normalization: Applied after each linear layer to stabilize training
   - Dropout: Applied with a 10% rate in the second layer to prevent overfitting
   - L2 Normalization: Applied to the final embeddings to ensure they lie on a unit hypersphere, making cosine similarity calculations more effective

4. **Embedding Space**: Both towers output 32-dimensional embeddings in the same vector space, allowing direct comparison via cosine similarity.

## Training Process

The model is trained by:

1. Preprocessing raw hotel booking data to extract user and hotel features
2. Encoding categorical variables and handling date information
3. Creating positive sample pairs (user-hotel matches)
4. Training both towers simultaneously with a shared optimizer
5. Using cosine similarity and MSE loss to push matching user-hotel pairs to have high similarity (close to 1)

## Inference and Recommendation

During inference, the system:

1. Encodes a user's features through the User Tower to get a user embedding
2. Encodes multiple hotels through the Hotel Tower to get hotel embeddings
3. Computes cosine similarity between the user embedding and each hotel embedding
4. Returns the top-k hotels with the highest similarity scores as recommendations

## Data Preprocessing

The preprocessing pipeline:

1. Filters only for actual bookings (is_booking = 1)
2. Separates user and hotel features
3. Encodes categorical variables like continent, country, region, and city
4. Converts check-in and check-out dates to calculate reservation duration
5. Handles missing values
6. Splits data into training and testing sets

## Model Evaluation

The model is evaluated by computing the average cosine similarity between user and hotel pairs in the test set. Higher average similarity indicates better model performance.

### Model Evaluation Implementation

The recommendation system includes detailed evaluation mechanisms to assess its performance:

```python
# From scripts/Model_Evaluation.py
def model_evaluation(Selected_Hotel_data, true_hotel_data):
    """
    Evaluates similarity between selected hotel data and true hotel data.
    
    Args:
        Selected_Hotel_data: DataFrame with 20 rows and 32 columns of selected hotel features
        true_hotel_data: DataFrame with 1 row and 32 columns of true hotel features
        
    Returns:
        float: Ratio of hotels with similarity above 0.9 threshold
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_tower = torch.load("user_tower.pth", weights_only=False)
    hotel_tower = torch.load("hotel_tower.pth", weights_only=False)

    user_tower.eval()
    hotel_tower.eval()

    # Convert true_hotel_data (1 row) to tensor
    true_hotel_tensor = torch.tensor(true_hotel_data.values, dtype=torch.float32).to(device)
    
    # Process true hotel data through hotel tower
    with torch.no_grad():
        true_hotel_embedding = hotel_tower(true_hotel_tensor)
    
    # Define cosine similarity function
    cos_sim = nn.CosineSimilarity(dim=1)
    
    # Track similarities and threshold count
    similarities = []
    above_threshold_count = 0
    threshold = 0.9
    total_count = len(Selected_Hotel_data)
    
    # Process each selected hotel
    with torch.no_grad():
        # Convert Selected_Hotel_data to tensor
        selected_hotels_tensor = torch.tensor(Selected_Hotel_data.values, dtype=torch.float32).to(device)
        
        # Get embeddings for all selected hotels
        selected_hotels_embeddings = hotel_tower(selected_hotels_tensor)
        
        # Calculate similarity between each selected hotel and the true hotel
        for i in range(total_count):
            similarity = cos_sim(selected_hotels_embeddings[i:i+1], true_hotel_embedding)
            sim_value = similarity.item()
            similarities.append(sim_value)
            
            if sim_value > threshold:
                above_threshold_count += 1
    
    # Calculate ratio of hotels above threshold
    ratio = above_threshold_count / total_count
    
    print(f"Hotels with similarity > {threshold}: {above_threshold_count}/{total_count}")
    print(f"Ratio: {ratio:.4f}")
    
    return ratio
```

### Full Evaluation Process

The system includes a comprehensive evaluation pipeline that assesses recommendation accuracy across multiple users:

```python
# From scripts/Evaluate.py
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
```

The evaluation metrics focus on the similarity ratio - the proportion of recommended hotels that exceed a similarity threshold of 0.9 with the user's true hotel choice. This provides a clear measurement of how well the recommendation system matches users with relevant hotels.

## Dependencies

- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

### Installation Steps

1. **Clone or download the repository**:
   ```bash
   git clone <repository-url>
   cd hotel_recommendation_system
   ```

2. **Install the package and dependencies**:
   ```bash
   pip install -e .
   ```
   Or install using the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the installation**:
   ```bash
   python -c "from scripts.Double_Tower import DeepUserTower, DeepHotelTower; print('Installation successful!')"
   ```

### Directory Structure

```
hotel_recommendation_system/
├── data/                  # Contains training data
│   └── train.csv         
├── models/                # Pre-trained model files
│   ├── hotel_tower.pth   
│   └── user_tower.pth    
├── scripts/               # Core implementation files
│   ├── Data_Preprocess.py
│   ├── Dataset_Class.py  
│   ├── Double_Tower.py   
│   ├── Model_Evaluation.py
│   └── Recommand_Deeplearning.py
├── Evaluate.py            # Evaluation script
├── Get_Recommandation.py  # Script for generating recommendations
├── Setup.py               # Package setup file
├── Train.py               # Training script
└── README.md              # Project documentation
```

### Troubleshooting

#### Model File Paths

If you encounter errors related to model file paths, ensure that the model files (`hotel_tower.pth` and `user_tower.pth`) are located either in:
- The root directory
- The `models/` directory
- The `Model/` directory

The system will check all these locations when loading models.

#### Data File Not Found

If the system cannot find the training data file, ensure that `train.csv` is located in the `data/` directory. You may need to adjust the path in the scripts if your data is stored in a different location.

#### CUDA/GPU Issues

If you encounter CUDA-related errors:
1. Ensure you have a compatible NVIDIA GPU
2. Install the appropriate CUDA toolkit
3. Install the CUDA-compatible version of PyTorch

If you don't have a GPU, the system will automatically use CPU mode, though training and evaluation will be slower.

## Usage

### Running the Scripts

1. **Evaluate the recommendation system**:
   ```bash
   python Evaluate.py
   ```

2. **Train the model**:
   ```bash
   python Train.py
   ```

3. **Get recommendations**:
   ```bash
   python Get_Recommandation.py
   ```

### Using the API

1. Process the raw data:
   ```python
   from scripts.Data_Preprocess import data_preprocess
   train_User_data, test_User_data, train_Hotel_data, test_Hotel_data = data_preprocess(raw_data)
   ```

2. Train the model:
   ```python
   from Train import train_model
   user_tower, hotel_tower = train_model(raw_data)
   ```

3. Evaluate the model:
   ```python
   from scripts.Model_Evaluation import model_evaluation
   avg_similarity = model_evaluation(test_User_data, test_Hotel_data)
   ```

4. Get hotel recommendations for a user:
   ```python
   from scripts.Recommand_Deeplearning import hotel_data_encoder, get_topk_similar_hotels
   encoded_hotels = hotel_data_encoder(hotel_data)
   recommendations = get_topk_similar_hotels(user_vector, encoded_hotels, hotel_data, topk=20)
   ```

## Future Improvements

Possible enhancements to the recommendation system:
- Incorporate negative sampling techniques
- Add attention mechanisms to better capture feature importance
- Implement more complex architectures like transformers
- Add content-based features from hotel descriptions
- Include collaborative filtering signals


# Hotel Recommendation System Evaluation

## Overview
This document describes the evaluation process of our two-tower recommendation model for hotel recommendations. The model uses a dual-encoder architecture where user features and hotel features are encoded into separate embedding spaces, and recommendations are made based on similarity between these representations.

## Evaluation Process

### Data Preparation
1. The model was trained on a filtered dataset containing only actual bookings (`is_booking == 1`).
2. User features include:
   - Geographical information (continent, country, region, city)
   - Distance metrics (origin-destination distance)
   - Device information (mobile usage)
   - Search parameters (check-in/out dates, number of adults, children, rooms)
3. Hotel features include:
   - Location data (continent, country, market)
   - Hotel cluster information

### Model Architecture
- **User Tower**: Encodes user features into a 32-dimensional embedding
- **Hotel Tower**: Encodes hotel features into a matching 32-dimensional embedding
- Both pre-trained models are loaded from saved files (`user_tower.pth` and `hotel_tower.pth`)

### Evaluation Methodology
1. **Test Dataset Creation**: 10,000 random user-hotel pairs were sampled from the processed dataset
2. **Embedding Generation**:
   - Unique hotel combinations were identified and encoded with the hotel tower
   - Each test user was encoded with the user tower
3. **Similarity Calculation**:
   - For each user, the system computes similarity scores with all encoded hotels
   - Top 20 most similar hotels are retrieved for each user
4. **Performance Metrics**:
   - **Rank Score**: Calculated as 1.0 / log2(1/similarity) for the actual booked hotel if it appears in top 20
   - **Correlation Rate**: Measures how strongly correlated the top 20 recommended hotels are with the user vector
   - A similarity threshold of 0.999999 is used to identify strongly correlated matches

### Results
The evaluation generates two main metrics:
- Individual score for each user based on the rank of their actual booked hotel
- Average correlation rate across all test users (current value: displayed in system output)

This evaluation framework allows us to assess how effectively our recommendation model can identify hotels that match user preferences based on their encoded feature representations.
