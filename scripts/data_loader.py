# src/data_loader.py

import pandas as pd

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
