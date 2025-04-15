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