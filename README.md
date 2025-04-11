
# Hotel Recommendation System

## Overview

Travelers often struggle to find the ideal hotel that matches their unique preferences and constraints. Whether searching for luxury accommodations, budget-friendly options, or hotels offering specific amenities, existing platforms rarely deliver personalized recommendations. This project develops a robust recommendation system tailored to individual traveler interests and needs.

## Approaches 

### Naïve Rule-Based Classifier
- Top 10 hotel recommendation based on rating for specific locations

### Traditional Machine Learning
- A hybrid filtering method combines multiple techniques to capture user preferences and rank hotel recommendations effectively.

- **Matrix Factorization:**  
  Utilize techniques like Singular Value Decomposition (SVD) and Alternating Least Squares (ALS).

- **Gradient Boosting:**  
  Implement ranking models using XGBoost and LightGBM.

### Deep Learning 
- For capturing complex patterns in both numeric and textual data, the deep learning approach integrates neural networks with transformer-based models.


**Evaluation Metrics:**  
WIP


## User Interface

- **Streamlit Web App**: 
WIP

## Setup

```bash
./setup.sh
```

This script takes care of setting up your virtual environment if it does not already exist, activating it, installing requirements, pulling the dataset (if not already present in the data directory), pre-processing the data for the traditional model (feature extraction), and traditional model training.

## Running the Streamlit application locally

Assuming your virtual environment is setup and activated, and that the requirements are installed from running `setup.sh`,
you can then run the following to startup a local instance of the Streamlit application.

```bash
python main.py
```

## Dataset & License
This repository uses the [Hotels Dataset](https://www.kaggle.com/datasets/raj713335/tbo-hotels-dataset/data) dataset from Kaggle, licensed under MIT.

## **Ethics Statement**  

This project uses publicly available datasets in compliance with their terms of use. We ensure that all data is handled responsibly, avoiding any misuse, unauthorized distribution, or unethical applications. No personally identifiable information (PII) is collected or used, and we strive to mitigate bias in AI-driven resume analysis. Our goal is to enhance fair and transparent hiring processes while respecting data privacy and ethical AI principles.


## Presentation Link

View our presentation [HERE](https://docs.google.com/presentation/d/1f10f97H5Tj7s4oodW_kLxO4mKXoLSJzMlBV520TZrPM/edit?usp=sharing).

## Streamlit Application

WIP
Access our Streamlit application [HERE]().
