
# Hotel Recommendation System

## Overview

Travelers often struggle to find the ideal hotel that matches their unique preferences and constraints. Whether searching for luxury accommodations, budget-friendly options, or hotels offering specific amenities, existing platforms rarely deliver personalized recommendations. This project develops a robust recommendation system tailored to individual traveler interests and needs.

---

## Data Processing Pipeline

WIP @ruhan-dave
---

## Modeling Approaches 

### Naïve Rule-Based Classifier
- Top 10 hotel recommendation based on rating for specific locations

### Traditional Machine Learning
- A hybrid filtering method combines multiple techniques to capture user preferences and rank hotel recommendations effectively.

- **Matrix Factorization:**  
  Utilize techniques like Singular Value Decomposition (SVD) and Alternating Least Squares (ALS).

- **Gradient Boosting:**  
  Implement ranking models using LightGBM.

### Deep Learning 
WIP
- For capturing complex patterns in both numeric and textual data, the deep learning approach integrates neural networks with transformer-based models. 

---

## Evaluation Metric

The models were primarily evaluated using mean Average Precision (mAP). mAP was used to measure the model’s ability to accurately rank hotels by calculating the average precision across all queries, emphasizing both the correctness and the order of relevant results.

---

## User Interface

- **Streamlit Web App**: 
WIP

---

## Setup

```bash
./setup.sh
```

This script takes care of setting up your virtual environment if it does not already exist, activating it, installing requirements, pulling the dataset (if not already present in the data directory), and pre-processing the data.

## Running the Streamlit application locally

Assuming your virtual environment is setup and activated, and that the requirements are installed from running `setup.sh`,
you can then run the following to startup a local instance of the Streamlit application.

```bash
python main.py
```

---

## Dataset & License
This repository uses the [Expedia Hotel Dataset](https://www.kaggle.com/c/expedia-hotel-recommendations) dataset licensed under competition rules definded by Kaggle.

---

## **Ethics Statement**  

This project uses publicly available datasets in compliance with their terms of use. We ensure that all data is handled responsibly, avoiding any misuse, unauthorized distribution, or unethical applications. No personally identifiable information (PII) is collected or used. 

---

## Presentation Link

View our presentation [HERE](https://docs.google.com/presentation/d/1f10f97H5Tj7s4oodW_kLxO4mKXoLSJzMlBV520TZrPM/edit?usp=sharing).

---

## Streamlit Application

WIP
Access our Streamlit application [HERE]().
