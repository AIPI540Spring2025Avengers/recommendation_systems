#!pip install xgboost

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import os
# %matplotlib inline

import plotly.express as px
# machine learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import pickle

#  destination = pd.read_csv('../input/destinations.csv', nrows=100000)
# Convert date object into relevant attributes
def convert_date_into_days(df):
    df['srch_ci'] = pd.to_datetime(df['srch_ci'])
    df['srch_co'] = pd.to_datetime(df['srch_co'])
    df['date_time'] = pd.to_datetime(df['date_time'])

    df['stay_dur'] = (df['srch_co'] - df['srch_ci']).astype('timedelta64[ns]')
    df['no_of_days_bet_booking'] = (df['srch_ci'] - df['date_time']).astype('timedelta64[ns]')

    # Check-in Month, Year, Day
    df['Cin_day'] = df["srch_ci"].apply(lambda x: x.day)
    df['Cin_month'] = df["srch_ci"].apply(lambda x: x.month)
    df['Cin_year'] = df["srch_ci"].apply(lambda x: x.year)

def understand_df_compact(df):
    # Dimensions of dataset
    print("———————————————————————————————————————————————————————————")
    print("Dimension of the dataset is", df.shape, "\n")

    # Summary of dataset
    print("———————————————————————————————————————————————————————————")
    print("Summary of the dataset is \n", df.describe(), "\n")

    print("———————————————————————————————————————————————————————————")
    print("Number of duplicates:")
    print(df.duplicated().sum())

    # Stats of dataset
    stats = []

    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Type'])
    print("———————————————————————————————————————————————————————————")

    print(f"Statistics of the dataset are \n {stats_df.sort_values('Percentage of missing values', ascending=False)}\n\n")

"""Outliers"""
def winsorize_outliers(df):
    """
    Adjusts outliers in the DataFrame using the winsorizing method based on the IQR.

    Returns:
    pd.DataFrame: DataFrame with outliers adjusted.
    """
    # Iterate through each column in the DataFrame
    for column in df.select_dtypes(include=[np.number]).columns:
        # Calculate Q1 and Q3
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        # Calculate Interquartile Range (IQR)
        IQR = Q3 - Q1

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Winsorizing outliers
        df.loc[:,column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    return df

# Define MAP@k metric
def mapk(actual, predicted, k=5):
    """
    Computes the Mean Average Precision at k (MAP@k) and returns MAP@k score.
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

    return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])

def xgboost_map_classification(data, target_col='hotel_cluster', features=None, k_values=[5], n_splits=5):
    """
    Trains an XGBoost model for multiclass classification using 5-fold cross-validation
    and evaluates it using MAP@k metrics. Plots a large confusion matrix after all folds.
    """
    # Prepare features and target
    if features is None:
        features = [col for col in data.columns if col != target_col]
    x = data[features]
    y = data[target_col]

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Initialize variables to store results
    overall_map_scores = {f"MAP@{k}": [] for k in k_values}
    all_y_test = []
    all_y_pred_top1 = []
    fold = 1

    # Do cross-validation
    for train_index, test_index in skf.split(x, y):
        print(f"Fold {fold}/{n_splits}")
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train XGBoost classifier
        model = XGBClassifier(objective="multi:softprob", num_class=len(np.unique(y)), eval_metric="mlogloss",
                              max_depth=10, random_state=0, n_jobs=-1)
        model.fit(x_train, y_train)

        # Make predictions
        y_proba = model.predict_proba(x_test)  # Probabilities for each class
        y_pred = np.argsort(y_proba, axis=1)[:, ::-1]  # Sort predictions by probability (descending)

        # Calculate MAP@k for each k in k_values
        fold_map_scores = {}
        for k in k_values:
            fold_map_scores[f"MAP@{k}"] = mapk(y_test.values, y_pred, k=k)
            overall_map_scores[f"MAP@{k}"].append(fold_map_scores[f"MAP@{k}"])

        # Print MAP@k scores for the current fold
        for k, score in fold_map_scores.items():
            print(f"Fold {fold} {k}: {score:.4f}")

        # Collect true and predicted labels for the confusion matrix
        all_y_test.extend(y_test.values)
        all_y_pred_top1.extend(y_pred[:, 0])  # Top-1 predictions

        fold += 1

    # Calculate average MAP@k scores across all folds
    avg_map_scores = {k: np.mean(scores) for k, scores in overall_map_scores.items()}
    print("\nAverage MAP@k Scores Across All Folds:")
    for k, score in avg_map_scores.items():
        print(f"{k}: {score:.4f}")

    # Plot the confusion matrix after cross validation
    cm = confusion_matrix(all_y_test, all_y_pred_top1)
    plt.figure(figsize=(20, 16))  # Make the plot large to accommodate many clusters
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", cbar=True)
    plt.title("Confusion Matrix (Aggregated Across All Folds)", fontsize=20)
    plt.xlabel("Predicted Labels", fontsize=16)
    plt.ylabel("True Labels", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    return model, avg_map_scores, y_pred, y_proba


def preprocess_datetime_columns(data):
    """
    Converts all datetime columns in the dataset to numeric (float or int) format.
    :param data: DataFrame containing the dataset.
    :return: DataFrame with datetime columns converted to numeric format.
    """
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]) or data[col].dtype == 'object':
            try:
                # Attempt to parse the column as datetime
                data[col] = pd.to_datetime(data[col], errors='coerce')
                # Convert datetime to numeric (e.g., Unix timestamp)
                data[col] = data[col].map(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
            except Exception:
                # If parsing fails, leave the column as is
                pass
    return data

def xgb_plot_feature_importance(model):
    """
    Plots the feature importance of an XGBoost model.
    :param model: Trained XGBoost model.
    :return: DataFrame of feature importances and a Plotly bar chart.
    """
    # Get feature names and importances
    feature_names = model.get_booster().feature_names  # Get feature names from the model
    feature_importances = model.feature_importances_  # Get feature importances

    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

    # Plot feature importances using Plotly
    fig = px.bar(feature_importance_df, y="importance", x="feature", title="Feature Importance",
                 labels={"importance": "Importance", "feature": "Feature"}, text="importance")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)

    # Show the plot
    fig.show()

    return feature_importance_df

def preprocess_and_save_data():
    """
    Preprocesses the data and saves it to the processed directory without training the model.
    """
    print("----------------------------")
    print("Starting data preprocessing...")

    # get expedia & test csv files as a DataFrame
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    train_df = pd.read_csv(os.path.join(data_dir, "raw", "train.csv"), nrows=10000)
    test_df = pd.read_csv(os.path.join(data_dir, "raw", "test.csv"), nrows=10000)

    # preview the data
    train_df.head()
    convert_date_into_days(train_df)
    convert_date_into_days(test_df)

    # Check the percentage of Nan in dataset
    total = train_df.isnull().sum().sort_values(ascending=False)
    percent = (train_df.isnull().sum()/train_df['hotel_cluster'].count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    # Fill nan with the day which has max occurence
    train_df['Cin_day'] = train_df['Cin_day'].fillna(26.0)
    train_df['Cin_month'] = train_df['Cin_month'].fillna(8.0)
    train_df['Cin_year'] = train_df['Cin_year'].fillna(2014.0)
    train_df['stay_dur'] = train_df['stay_dur'].fillna(1.0)
    train_df['no_of_days_bet_booking'] = train_df['no_of_days_bet_booking'].fillna(0.0)

    # Fill average values in place for nan, fill with mean
    train_df['orig_destination_distance'].fillna(train_df['orig_destination_distance'].mean(), inplace=True)

    # Give detailed overview of the dataset 
    understand_df_compact(train_df)

    # Remove some null values 
    train_df.isnull().sum()
    train_df["srch_ci"].fillna(train_df["srch_ci"].mean(), inplace=True)
    train_df["srch_co"].fillna(train_df["srch_co"].mean(), inplace=True)

    # Winsorize Outliers
    train_df = winsorize_outliers(train_df)
    test_df = winsorize_outliers(test_df)

    # Remove irrelevant features 
    features = [t for t in train_df.columns if t not in ["date_time", "hotel_cluster"]]
    train_df = preprocess_datetime_columns(train_df)
    test_df = preprocess_datetime_columns(test_df)

    # save the processed train data in processed folder
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, "train_processed.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_processed.csv"), index=False)
    
    print("Data preprocessing and saving completed!")
    return train_df, test_df, features

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run data preprocessing and/or model training')
    parser.add_argument('--preprocess-only', action='store_true', help='Only run data preprocessing without training')
    args = parser.parse_args()
    
    if args.preprocess_only:
        preprocess_and_save_data()
    else:
        # Run the full pipeline including training
        train_df, test_df, features = preprocess_and_save_data()
        
        # Model Training 
        xgb, map_scores, y_pred, y_proba = xgboost_map_classification(train_df, target_col='hotel_cluster', features=features, k_values=[5])

        # Plot feature importance map
        feature_importance_df = xgb_plot_feature_importance(xgb)
        # Display top 10 features
        feature_importance_df["feature"][:10].to_list()
        # save model in models folder
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "xgboost_model.pkl"), "wb") as file:
            pickle.dump(xgb, file)

        # Load the model using pickle
        with open(os.path.join(models_dir, "xgboost_model.pkl"), "rb") as file:
            loaded_model = pickle.load(file)