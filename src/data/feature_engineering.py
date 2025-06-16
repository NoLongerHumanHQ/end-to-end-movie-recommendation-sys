# Feature Engineering Script for Churn Prediction Pipeline

"""
This script performs feature engineering on the Telco Customer Churn dataset.
It handles data transformations, creates new features, and saves the processed data.
"""

import pandas as pd
import os
# Removed unused import to resolve Pylance errors

def feature_engineering(input_path: str = "data/raw/telco_churn.csv", output_path: str = "data/processed/churn_features.csv") -> None:
    """
    Perform feature engineering on the Telco Churn dataset.
    
    Args:
        input_path (str): Path to input CSV file. Defaults to "data/raw/telco_churn.csv".
        output_path (str): Path to save processed CSV file. Defaults to "data/processed/churn_features.csv".
        
    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If input file is not CSV or data is empty.
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
    
    # Check if input is a CSV file
    if not input_path.endswith('.csv'):
        raise ValueError("Input file must be a CSV file for the Telco Churn dataset.")
    
    try:
        # Load the data
        df = pd.read_csv(input_path)  # type: ignore
        # Basic validation: ensure data is not empty
        if df.empty:
            raise ValueError("The loaded dataset is empty. Please ensure the dataset is properly placed and non-empty.")
        
        # Perform feature engineering steps
        # 1. Handle missing values (drop them for simplicity)
        # Handle missing values (drop them for simplicity)
        df.dropna(inplace=True)  # type: ignore
        
        # 2. Convert tenure to a binned categorical variable
        df['tenure_bin'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60], labels=['0-12', '13-24', '25-36', '37-48', '49+'])  # type: ignore
        
        # 3. One-hot encode categorical variables (example for 'Contract' column)
        # Note: The dataset schema isn't specified, so this assumes columns like 'Contract' exist
        # If the dataset doesn't have categorical columns, this can be adjusted
        categorical_cols = ['Contract', 'PaymentMethod']  # Common columns in churn datasets
        for col in categorical_cols:
            if col in df.columns and df[col].dtype == 'object':
                df = pd.get_dummies(df, columns=[col], prefix=[col])
        
        # 4. Handle other potential feature engineering (example: log transformation for numerical features)
        # For simplicity, I'll apply log transformation to 'MonthlyCharges' if it exists
        numerical_cols = ['MonthlyCharges', 'TotalCharges']  # Example columns
        for col in numerical_cols:
            if col in df.columns:
                # Add small epsilon to avoid log(0) issues
                df[col] = df[col].apply(lambda x: math.log(x + 1e-5) if x > 0 else 0)  # type: ignore
        
        # Save the processed data
        df.to_csv(output_path, index=False)
        print(f"Feature engineering completed. Processed data saved to {output_path}")
        
    except Exception as e:
        raise ValueError(f"Error during feature engineering: {str(e)}") from e

# Import math for log transformation (needed for the example)
import math

if __name__ == "__main__":
    # For testing purposes, can run directly
    feature_engineering()
