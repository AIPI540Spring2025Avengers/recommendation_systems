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

## Usage

1. Process the raw data:
   ```python
   from Function.Data_Preprocess import data_preprocess
   train_User_data, test_User_data, train_Hotel_data, test_Hotel_data = data_preprocess(raw_data)
   ```

2. Train the model:
   ```python
   from Train import train_model
   user_tower, hotel_tower = train_model(raw_data)
   ```

3. Evaluate the model:
   ```python
   from Function.Model_Evaluation import model_evaluation
   avg_similarity = model_evaluation(test_User_data, test_Hotel_data)
   ```

4. Get hotel recommendations for a user:
   ```python
   from Function.Recommand_Deeplearning import hotel_data_encoder, get_topk_similar_hotels
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