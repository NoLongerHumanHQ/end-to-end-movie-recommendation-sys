# Data Ingestion Script for Churn Prediction Pipeline

"""
This script handles the ingestion and validation of the Telco Customer Churn dataset.
It performs basic validation checks to ensure the data is properly formatted and exists.
"""

import pandas as pd
import os
from typing import Optional

def ingest_data(file_path: str = "data/raw/telco_churn.csv") -> Optional[pd.DataFrame]:
    """
    Ingest and validate the Telco Customer Churn dataset.
    
    Args:
        file_path (str): Path to the CSV file. Defaults to "data/raw/telco_churn.csv".
        
    Returns:
        pd.DataFrame: The loaded and processed DataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a CSV or cannot be read properly.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please place the Telco Churn dataset in the data/raw/ directory.")
    
    # Check if it's a CSV file
    if not file_path.endswith('.csv'):
        raise ValueError(f"The file {file_path} is not a CSV file. Expected a CSV file for the Telco Churn dataset.")
    
    # Load the data
    try:
        df = pd.read_csv(file_path)
        # Basic validation: check for non-empty dataframe
        if df.empty:
            raise ValueError("The loaded dataset is empty. Please ensure the dataset is not empty.")
        return df
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {str(e)}") from e

if __name__ == "__main__":
    # For testing purposes, can run directly to see the data
    df = ingest_data()
    print("Data ingestion successful. First few rows:")
    print(df.head())
