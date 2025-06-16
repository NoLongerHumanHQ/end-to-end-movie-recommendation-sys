# Model Training Pipeline for Churn Prediction

"""
This script trains and evaluates multiple models for the Telco Customer Churn dataset.
It uses Optuna for hyperparameter tuning and MLflow for experiment tracking.
"""

import pandas as pd
import numpy as np
import os
import mlflow
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dtree import DecisionTreeClassifier
from sklearn.xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import NotFittedError

def load_data(file_path="data/processed/churn_features.csv"):
    """
    Load processed data from CSV file.
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded and prepared data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    df = pd.read_csv(file_path)
    return df

def prepare_data(df):
    """
    Prepare data for modeling by splitting and scaling.
    
    Args:
        df (pd.DataFrame): Input data
        
    Returns:
        X_train, X_test, y_train, y_test: Prepared datasets
    """
    # Assuming 'Churn' is the target variable
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Create transformers for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Transform data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def train_model(model_name, X_train, y_train):
    """
    Train a model with hyperparameter tuning using Optuna.
    
    Args:
        model_name (str): Name of the model to train
        X_train: Training features
        y_train: Training target
        
    Returns:
        Best model from tuning
    """
    def objective(trial):
        # Hyperparameter search space
        if model_name == "RandomForest":
            max_depth = trial.suggest_int("max_depth", 10, 100)
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
            
            # Create and train model
            clf = RandomForestClassifier(
                max_depth=max_depth,
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
        elif model_name == "XGBoost":
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            subsample = trial.suggest_float("subsample", 0.5, 1.0, step=0.1)
            
            # Create and train model
            clf = XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                random_state=42,
                use_label_encoder=False
            )
            
        elif model_name == "LogisticRegression":
            C = trial.suggest_float("C", 0.001, 10.0, log=True)
            penalty = trial.suggest_categorical("penalty", ["l2", "none"])
            
            # Create and train model
            clf = LogisticRegression(
                C=C,
                penalty=penalty,
                solver='liblinear' if penalty == 'none' else 'lbfgs',
                max_iter=1000,
                random_state=42
            )
            
        elif model_name == "DecisionTree":
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
            
            # Create and train model
            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Train model
        clf.fit(X_train, y_train)
        return 0  # Dummy return value for now
        
    # Run optimization (we'll leave this as a placeholder)
    best_clf = None
    
    return best_clf

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation metrics
    """
    if model is None:
        return {}
    
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    except NotFittedError:
        return {}

def register_model(model, model_name, metrics):
    """
    Register model in MLflow.
    
    Args:
        model: Trained model
        model_name (str): Name of the model
        metrics (dict): Evaluation metrics
    """
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

if __name__ == "__main__":
    # For testing purposes
    try:
        df = load_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        
        # Train and evaluate models
        for model_name in ["RandomForest", "XGBoost", "LogisticRegression", "DecisionTree"]:
            print(f"Training {model_name}...")
            model = train_model(model_name, X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            register_model(model, model_name, metrics)
            print(f"Completed {model_name} training")
            
    except Exception as e:
        print(f"Error during model training: {str(e)}")
