import pandas as pd
import numpy as np
def clean_data(data):
    """
    Clean the dataset by handling missing values and outliers.

    Args:
        data (pd.DataFrame): The raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Handling missing values
    data.fillna(method='ffill', inplace=True)  # Forward fill for simplicity

    # Handling outliers
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data[col] = np.clip(data[col], lower_bound, upper_bound)

    return data

def save_clean_data(data, output_path):
    """
    Save the cleaned dataset to a CSV file.

    Args:
        data (pd.DataFrame): Cleaned dataset.
        output_path (str): Path to save the cleaned data.
    """
    data.to_csv(output_path, index=False)
