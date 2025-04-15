# Hotel Recommendation System

## Overview

Travelers often struggle to find the ideal hotel that matches their unique preferences and constraints. Whether searching for luxury accommodations, budget-friendly options, or hotels offering specific amenities, existing platforms rarely deliver personalized recommendations. This project develops a robust recommendation system tailored to individual traveler interests and needs.

---
## Data Processing Pipeline

### Deep Learning
•⁠  ⁠Filters for actual bookings (⁠ is_booking = 1 ⁠)
•⁠  ⁠Separates user and hotel features
•⁠  ⁠Encodes categorical variables (e.g., continent, country, region, city)
•⁠  ⁠Calculates reservation duration from check-in and check-out dates
•⁠  ⁠Handles missing values
•⁠  ⁠Splits data into training and testing sets

### Traditional Approach
•⁠  ⁠Transforms datetime columns into numerical features (e.g., year, month, day)
•⁠  ⁠Applies winsorization to cap outliers using IQR-based thresholds:  
  ⁠ lower_bound = Q1 - 1.5 * IQR ⁠, ⁠ upper_bound = Q3 + 1.5 * IQR ⁠
•⁠  ⁠Replaces sparse null values with column means
•⁠  ⁠Ensures all features are numeric and suitable for model training

---

## Modeling Approaches 

### Naïve Rule-Based Classifier
•⁠  ⁠Top K hotel recommendations based on popularity
•⁠  ⁠MAP@5: 0.0745

### Traditional Modeling Approach

We use an XGBoost classifier for multiclass hotel recommendation, trained with 5-fold stratified cross-validation to maintain class distribution across folds.

### Model Configuration
•⁠  ⁠⁠ objective="multi:softprob" ⁠: Multiclass classification with probability outputs
•⁠  ⁠⁠ num_class ⁠: Automatically set based on number of unique hotel labels
•⁠  ⁠⁠ eval_metric="mlogloss" ⁠: Multi-class log loss as the evaluation metric
•⁠  ⁠⁠ max_depth=10 ⁠: Limits tree depth to reduce overfitting
•⁠  ⁠⁠ n_jobs=-1 ⁠: Enables parallel processing using all available CPU cores

### Deep Learning Approach


The deep learning approach uses a dual-tower neural network to generate hotel recommendations. The model maps user and hotel features into a shared 32-dimensional embedding space using fully connected layers with ReLU activation, batch normalization, and L2 normalization. Cosine similarity between user and hotel embeddings is computed to rank hotels. Given a user's search profile, the model retrieves the top K most similar hotels based on embedding similarity.

Model Deployed Here: https://huggingface.co/okamdar/hotel-rec/tree/main

---

## Previous Efforts

### Hotel2Vec Embeddings  
Sadeghian et al. developed *Hotel2Vec*, a neural network architecture that learns hotel embeddings by integrating user clicks, hotel attributes, amenities, and geographic information. This approach effectively tackles the cold-start problem by incorporating diverse data sources.  
[Paper: Hotel2Vec – Learning Hotel Embeddings](https://arxiv.org/abs/1910.03943)

### NLP-Based Sentiment Analysis  
Aravani et al. proposed a framework utilizing *BERT-based models* to analyze user reviews, categorizing hotels into "Bad," "Good," or "Excellent" based on sentiment. This method enhances personalized recommendations by understanding user preferences through textual feedback.  
[Paper: Sentiment Analysis for Hotel Recommendation](https://arxiv.org/abs/2408.00716)

### Integration of ChatGPT and Persuasive Technologies  
Remountakis et al. explored the incorporation of *ChatGPT and persuasive techniques* into hotel recommender systems. Their approach aims to generate context-aware, personalized suggestions by analyzing user preferences and online reviews.  
[Paper: ChatGPT in Recommender Systems](https://arxiv.org/abs/2307.14298)

---

## Evaluation Metric

The models were primarily evaluated using mean Average Precision (mAP). mAP was used to measure the model's ability to accurately rank hotels by calculating the average precision across all queries, emphasizing both the correctness and the order of relevant results.

### Comparison MAP@5:
•⁠  ⁠Naive: 0.0745
•⁠  ⁠Traditional ML: 0.4276
•⁠  ⁠Deep Learning: 0.953

---

## User Interface

•⁠  ⁠*Streamlit Web App*: 
  - Interactive web interface for hotel recommendations
  - Supports all three recommendation systems approaches
  - Input is Random Generated User Profile from training data
  - Displays hotel recommendations by rank and respective MAP scores

---

## Setup

⁠ bash
./setup.sh
 ⁠

This script takes care of setting up your virtual environment if it does not already exist, activating it, installing requirements, pulling the dataset (if not already present in the data directory), and pre-processing the data.

## Running the Streamlit application locally

Assuming your virtual environment is setup and activated, and that the requirements are installed from running ⁠ setup.sh ⁠,
you can then run the following to startup a local instance of the Streamlit application.

⁠ bash
python streamlit run main.py
 ⁠

---

## Dataset & License
This repository uses the [Expedia Hotel Dataset](https://www.kaggle.com/c/expedia-hotel-recommendations) licensed under competition rules defined by Kaggle.

---

## *Ethics Statement*  

This project uses publicly available datasets in compliance with their terms of use. We ensure that all data is handled responsibly, avoiding any misuse, unauthorized distribution, or unethical applications. No personally identifiable information (PII) is collected or used. 

---

## Presentation Pitch

View our Pitch [HERE](https://docs.google.com/presentation/d/1f10f97H5Tj7s4oodW_kLxO4mKXoLSJzMlBV520TZrPM/edit?usp=sharing).

---

## Streamlit Application

Access our Streamlit application [HERE]().

# ChatGPT was used to help refine and polish this README for better grammar and flow.