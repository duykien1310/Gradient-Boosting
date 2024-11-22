#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loan Approval Classification using Gradient Boosting

This script performs classification to predict whether a loan should be approved
for individuals based on their features. It uses Gradient Boosting as the machine
learning model and evaluates the model using confusion matrix, recall, precision,
and F1-score.

Requirements:
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn

Usage:
    Ensure that the data files 'train.csv' and 'test.csv' are located in the 'data' folder.
    Run the script:
        python loan_approval_classifier.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
)
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")


def load_data(train_path, test_path):
    """
    Load training and testing data from CSV files.

    Parameters:
        train_path (str): Path to the training data CSV file.
        test_path (str): Path to the testing data CSV file.

    Returns:
        train_df (DataFrame): Training data.
        test_df (DataFrame): Testing data.
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print("Data loaded successfully.")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def preprocess_data(train_df, test_df, target_col):
    """
    Preprocess the training and testing data:
        - Handle missing values
        - Encode categorical variables
        - Scale numerical features

    Parameters:
        train_df (DataFrame): Training data.
        test_df (DataFrame): Testing data.
        target_col (str): Name of the target column.

    Returns:
        X_train (DataFrame): Preprocessed training features.
        y_train (Series): Training labels.
        X_test (DataFrame): Preprocessed testing features.
        y_test (Series): Testing labels (if available).
        label_encoders (dict): Dictionary of label encoders for categorical features.
        scaler (StandardScaler): Fitted scaler object.
    """
    # Separate features and target
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]

    # If test_df has target, separate it
    if target_col in test_df.columns:
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]
    else:
        X_test = test_df.copy()
        y_test = None

    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Handle missing values
    # Numerical columns: impute with mean
    num_imputer = SimpleImputer(strategy="mean")
    X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

    # Categorical columns: impute with most frequent
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    # Encode categorical variables using Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        # Handle unseen labels in test set
        X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
        # If 'Unknown' was added, refit LabelEncoder to include it
        if 'Unknown' in X_test[col].astype(str).values:
            le_classes = le.classes_.tolist()
            if 'Unknown' not in le_classes:
                le_classes.append('Unknown')
                le.classes_ = le_classes
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    # Feature Scaling
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, y_train, X_test, y_test, label_encoders, scaler


def train_model(X, y):
    """
    Train a Gradient Boosting Classifier with default parameters.

    Parameters:
        X (DataFrame): Training features.
        y (Series): Training labels.

    Returns:
        model (GradientBoostingClassifier): Trained model.
    """
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(X, y)
    print("Model training completed.")
    return model


def evaluate_model(model, X, y):
    """
    Evaluate the model using confusion matrix, recall, precision, and F1-score.

    Parameters:
        model (Classifier): Trained model.
        X (DataFrame): Features to predict.
        y (Series): True labels.

    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    recall = recall_score(y, y_pred, average='binary')
    precision = precision_score(y, y_pred, average='binary')
    f1 = f1_score(y, y_pred, average='binary')
    report = classification_report(y, y_pred)

    metrics = {
        "confusion_matrix": cm,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "classification_report": report,
    }

    print("Model Evaluation:")
    print("Confusion Matrix:")
    print(cm)
    print(f"\nRecall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)

    return metrics


def plot_feature_importances(model, feature_names, output_path=None):
    """
    Plot and display feature importances from the trained model.

    Parameters:
        model (Classifier): Trained model with feature_importances_ attribute.
        feature_names (list): List of feature names.
        output_path (str, optional): Path to save the plot image. Defaults to None.
    """
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importances')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()


def hyperparameter_tuning(X, y):
    """
    Perform hyperparameter tuning using GridSearchCV to find the best Gradient Boosting parameters.

    Parameters:
        X (DataFrame): Training features.
        y (Series): Training labels.

    Returns:
        best_model (GradientBoostingClassifier): Best estimator found by GridSearchCV.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
    }

    gb_clf = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=gb_clf,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X, y)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best F1 Score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    return best_model


def main():
    # Define file paths
    DATA_DIR = ''
    TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
    TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

    # Define target column
    TARGET_COL = 'loan_status'

    # Load data
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)

    # Check if target column exists in test data
    test_has_target = TARGET_COL in test_df.columns

    # Preprocess data
    X_train, y_train, X_test, y_test, label_encoders, scaler = preprocess_data(
        train_df, test_df, TARGET_COL
    )

    # If test set does not have labels, split training data into train and validation sets
    if not test_has_target:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
        X_test = X_val
        y_test = y_val
        print("Split training data into training and validation sets.")
    else:
        print("Test data contains target labels. Using them for evaluation.")

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Plot feature importances
    plot_feature_importances(model, X_train.columns.tolist())

    # Optional: Hyperparameter Tuning
    # Uncomment the following lines if you want to perform hyperparameter tuning
    """
    print("Starting hyperparameter tuning...")
    best_model = hyperparameter_tuning(X_train, y_train)
    print("Evaluating the best model from GridSearchCV...")
    best_metrics = evaluate_model(best_model, X_test, y_test)
    plot_feature_importances(best_model, X_train.columns.tolist())
    """

    # Save the trained model (optional)
    """
    import joblib
    model_path = 'gradient_boosting_model.joblib'
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}")
    """

    # Save evaluation metrics to a text file (optional)
    """
    with open('evaluation_metrics.txt', 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(metrics['confusion_matrix']))
        f.write("\n\nRecall: {:.4f}".format(metrics['recall']))
        f.write("\nPrecision: {:.4f}".format(metrics['precision']))
        f.write("\nF1-Score: {:.4f}".format(metrics['f1_score']))
        f.write("\n\nClassification Report:\n")
        f.write(metrics['classification_report'])
    print("Evaluation metrics saved to 'evaluation_metrics.txt'")
    """

    print("Script execution completed.")


if __name__ == '__main__':
    main()
