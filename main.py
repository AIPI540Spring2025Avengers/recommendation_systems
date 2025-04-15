import streamlit as st
import pandas as pd
from scripts.naive.naive_expedia import get_most_popular_hotels
import os
import numpy as np
import pickle
import plotly.express as px
import sys
import torch
from scripts.deep.Recommand_Deeplearning import get_topk_similar_hotels, hotel_data_encoder
from scripts.deep.Deeplearning_Class import DeepUserTower
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import random

# add directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts", "deep"))

# load processed data
def load_processed_data():
    """Load the processed training data from the processed directory."""
    data_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    train_df = pd.read_csv(os.path.join(data_dir, "sample_train.csv"))
    
    # Convert datetime-related columns to numeric
    datetime_columns = ['stay_dur', 'no_of_days_bet_booking', 'srch_ci', 'srch_co']
    for col in datetime_columns:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    
    # Features used during ML model training
    features = [
        'hotel_continent', 'hotel_country', 'hotel_market',
        'site_name', 'user_id', 'user_location_region',
        'Cin_month', 'srch_destination_id', 'srch_ci',
        'user_location_city'
    ]
    
    return train_df, features

# load sample profiles from sample train data
def load_sample_profiles():
    """Load sample user profiles from processed training data."""
    train_df, _ = load_processed_data()
    
    # Create user-friendly profile descriptions
    profiles = []
    for idx, row in train_df.iterrows():
        # Use the processed data directly
        profile_data = row.copy()
        
        # Create profile description
        profile = {
            'id': idx,
            'user_id': row['user_id'],
            'description': f"User {row['user_id']} - Country {row['user_location_country']} - Booked Cluster {row['hotel_cluster']}",
            'data': profile_data  # Use the processed data directly
        }
        profiles.append(profile)
    
    return profiles

# load xgboost model
def load_xgboost_model():
    """Load the pre-trained XGBoost model from the models directory."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(models_dir, "xgboost_model.pkl")
    
    if not os.path.exists(model_path):
        st.error("XGBoost model not found. Please ensure the model is trained and saved in the models directory.")
        return None
    
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# mapk function for evaluation
def mapk(actual, predicted, k=5):
    """
    Computes the Mean Average Precision at k (MAP@k) for a single user's recommendations.
    actual: List of actual hotel clusters booked by the user
    predicted: List of predicted hotel clusters for the user
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

    return apk(actual, predicted, k)


# Set page config
st.set_page_config(
    page_title="Hotel Recommendation System",
    page_icon="🏨",
    layout="wide"
)

# Title and description
st.title("🏨 Hotel Recommendation System")
st.markdown("""
This system provides hotel recommendations using different approaches:
1. **Naïve Baseline**: Simple popularity-based recommendations
2. **Machine Learning**: XGBoost-based recommendations
3. **Deep Learning**: Neural network-based recommendations 
""")

# Sidebar for profile selection
st.sidebar.title("User Profile Selection")
profiles = load_sample_profiles()

# Get a random profile from the entire dataset
if 'selected_profile' not in st.session_state:
    st.session_state.selected_profile = random.choice([p['description'] for p in profiles])

# Add a button to get a random profile
if st.sidebar.button("Get Random User"):
    st.session_state.selected_profile = random.choice([p['description'] for p in profiles])
    st.sidebar.success("Generated new user profile!")

# Get the selected profile data
selected_profile_data = next((p for p in profiles if p['description'] == st.session_state.selected_profile), None)

# Display selected profile details
if selected_profile_data:
    st.sidebar.markdown("### Profile Details")
    profile_data = selected_profile_data['data']
    
    # Format dates with error handling
    check_in = pd.to_datetime(profile_data['srch_ci'], unit='s') if 'srch_ci' in profile_data else None
    check_out = pd.to_datetime(profile_data['srch_co'], unit='s') if 'srch_co' in profile_data else None
    check_in_str = check_in.strftime('%Y-%m-%d') if check_in is not None else 'Invalid Date'
    check_out_str = check_out.strftime('%Y-%m-%d') if check_out is not None else 'Invalid Date'
    
    st.sidebar.markdown(f"""
    - **User ID**: {profile_data['user_id']}
    - **Location**: Country {profile_data['user_location_country']}, Region {profile_data['user_location_region']}
    - **Check-in**: {check_in_str}
    - **Check-out**: {check_out_str}
    - **Adults**: {profile_data['srch_adults_cnt']}
    - **Children**: {profile_data['srch_children_cnt']}
    - **Rooms**: {profile_data['srch_rm_cnt']}
    - **Mobile Booking**: {'Yes' if profile_data['is_mobile'] else 'No'}
    - **Package Deal**: {'Yes' if profile_data['is_package'] else 'No'}
    """)

# Number input for recommendations
top_n = st.sidebar.number_input(
    "Number of Recommendations",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

# Add a button to generate recommendations
if st.sidebar.button("Generate Recommendations"):
    # Show loading message
    with st.spinner('Loading data and generating recommendations...'):
        # Load processed data
        train_df, features = load_processed_data()
        
        # Create columns for side-by-side comparison
        col1, col2, col3 = st.columns(3)
        
        # Naïve Baseline
        with col1:
            st.header("Naïve Baseline")
        
            
            # Get recommendations and booking counts
            recommendations = get_most_popular_hotels(train_df, n=top_n)
            booking_counts = train_df.dropna(subset=['hotel_cluster'])['hotel_cluster'].value_counts()
            
            # Create a clean DataFrame for display with rank
            results_df = pd.DataFrame({
                'Rank': list(range(1, len(recommendations) + 1)),
                'Hotel Cluster': recommendations,
            })
            
            st.subheader(f"Top {len(recommendations)} Recommended Hotel Clusters")
            st.dataframe(results_df, hide_index=True)
            
            # Show MAP@k evaluation
            st.subheader("Model Evaluation")
            
            # Get user's actual bookings
            user_bookings = train_df[train_df['user_id'] == selected_profile_data['data']['user_id']]['hotel_cluster'].values
            if len(user_bookings) > 0:
                map_score = mapk(user_bookings, recommendations, k=5)
                st.metric(label="MAP@5", value=f"{map_score:.4f}")
                st.info(f"User has booked {len(user_bookings)} hotel clusters: {user_bookings}")
            else:
                st.info("No booking history available for this user.")
        
        # Machine Learning
        with col2:
            st.header("Machine Learning")
      
            
            # Load the pre-trained model
            model = load_xgboost_model()
            if model is None:
                st.error("Failed to load the XGBoost model. Please ensure the model is trained and saved in the models directory.")
                st.stop()
            
            try:
                # Get recommendations for the selected user
                user_data = selected_profile_data['data']
                user_features = pd.DataFrame([user_data[features]])
                user_proba = model.predict_proba(user_features)[0]
                top_clusters = np.argsort(user_proba)[::-1][:top_n]
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'Rank': range(1, top_n + 1),
                    'Hotel Cluster': top_clusters,
                    'Confidence Score': [user_proba[cluster] for cluster in top_clusters]
                })
                
                # Display recommendations
                st.subheader(f"Top {top_n} Recommended Hotel Clusters")
                st.dataframe(results_df, hide_index=True)
                
                # Show MAP@k evaluation
                st.subheader("Model Evaluation")
                
                # Get user's actual bookings
                user_bookings = train_df[train_df['user_id'] == user_data['user_id']]['hotel_cluster'].values
                if len(user_bookings) > 0:
                    map_score = mapk(user_bookings, top_clusters, k=5)
                    st.metric(label="MAP@5", value=f"{map_score:.4f}")
                    st.info(f"User has booked {len(user_bookings)} hotel clusters: {user_bookings}")
                else:
                    st.info("No booking history available for this user.")
                
            except Exception as e:
                st.error(f"Error generating ML recommendations: {str(e)}")

        # Deep Learning
        with col3:
            st.header("Deep Learning")
          
            
            try:
                # Set device
                device = torch.device("cpu")
                
                # Load model
                models_dir = os.path.join(os.path.dirname(__file__), "models")
                user_tower = torch.load(
                    os.path.join(models_dir, "user_tower.pth"),
                    map_location=device,
                    weights_only=False
                )
                user_tower.eval()
                
                # Get training data
                train_df, _ = load_processed_data()
                
                # Extract hotel data from training data
                hotel_columns = ['hotel_continent', 'hotel_country', 'hotel_market', 'hotel_cluster']
                hotel_data = train_df[hotel_columns].drop_duplicates()
                
                # Encode hotel data
                encoded_hotel_data = hotel_data_encoder(hotel_data)
                
                # Get user data from selected profile
                user_data = selected_profile_data['data']
                
                # Convert user data to tensor
                user_feats = torch.tensor([
                    user_data['posa_continent'],
                    user_data['user_location_country'],
                    user_data['user_location_region'],
                    user_data['user_location_city'],
                    user_data['orig_destination_distance'],
                    user_data['is_mobile'],
                    user_data['srch_adults_cnt'],
                    user_data['srch_children_cnt'],
                    user_data['srch_rm_cnt'],
                    (pd.to_datetime(user_data['srch_co']) - pd.to_datetime(user_data['srch_ci'])).days
                ], dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get user vector
                with torch.no_grad():
                    user_vec = user_tower(user_feats).squeeze(0).cpu().numpy()
                
                # Get recommendations
                recommendations = get_topk_similar_hotels(user_vec, encoded_hotel_data, hotel_data, topk=top_n)
                
                # Display recommendations
                st.subheader(f"Top {top_n} Recommended Hotel Clusters")
                results_df = pd.DataFrame({
                    'Rank': range(1, top_n + 1),
                    'Hotel Cluster': recommendations['hotel_cluster'].values,
                })
                st.dataframe(results_df, hide_index=True)
                
                # Show MAP@k evaluation
                st.subheader("Model Evaluation")
                
                # Get user's actual bookings
                user_bookings = train_df[train_df['user_id'] == user_data['user_id']]['hotel_cluster'].values
                if len(user_bookings) > 0:
                    map_score = mapk(user_bookings, recommendations['hotel_cluster'].values, k=5)
                    st.metric(label="MAP@5", value=f"{map_score:.4f}")
                    st.info(f"User has booked {len(user_bookings)} hotel clusters: {user_bookings}")
                else:
                    st.info("No booking history available for this user.")
                
            except Exception as e:
                st.error(f"Error generating deep learning recommendations: {str(e)}")
                st.error("Please ensure the deep learning models are trained and saved in the models directory.")

# Footer
st.markdown("---")


# this app was created using GPT-4o, with serveral prompts to the model to ensure the app works as expected.
# the provided code in the prompt was model implementation snippets from each modeling approach.