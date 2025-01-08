import os
import pandas as pd
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
def summarize_data(df):
    """
    Summarize dataset with column details.
    """
    print("Summary of the dataset:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
def identify_column_types(df):
    """
    Identify numeric and categorical columns.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    return numeric_cols, categorical_cols
