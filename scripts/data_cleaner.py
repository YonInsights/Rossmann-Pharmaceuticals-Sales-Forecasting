import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def clean_data(data: pd.DataFrame, data_type: str = "train", save_path: str = None) -> pd.DataFrame:
    """
    Cleans the input data by handling missing values, outliers, data types, and consistency checks.
    
    Parameters:
        data (pd.DataFrame): The input data to be cleaned.
        data_type (str): Type of data ('train', 'test', 'store') to handle specific columns differently.
        save_path (str): Path to save the cleaned data (optional).
        
    Returns:
        pd.DataFrame: The cleaned data.
    """
    # Handle Missing Values
    if data_type == "store":
        data['CompetitionDistance'].fillna(data['CompetitionDistance'].median(), inplace=True)
        data['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        data['CompetitionOpenSinceYear'].fillna(0, inplace=True)
        data['Promo2SinceWeek'].fillna(0, inplace=True)
        data['Promo2SinceYear'].fillna(0, inplace=True)
        data['PromoInterval'].fillna('None', inplace=True)

    elif data_type == "train" or data_type == "test":
        # Fill missing 'Open' values in test data with 1 (assume open if missing)
        if 'Open' in data.columns:
            data['Open'].fillna(1, inplace=True)

    # Convert Data Types
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])

    # Encode Categorical Features
    if 'StateHoliday' in data.columns:
        data['StateHoliday'] = data['StateHoliday'].replace({'0': '0', 'a': '1', 'b': '2', 'c': '3'}).astype(int)

    # Detect and Remove Duplicates
    data.drop_duplicates(inplace=True)

    # Standardize Numerical Columns (Optional for modeling phase)
    if data_type == "store" and 'CompetitionDistance' in data.columns:
        scaler = StandardScaler()
        data['CompetitionDistance'] = scaler.fit_transform(data[['CompetitionDistance']])

    # Save the cleaned data if save_path is provided
    if save_path:
        data.to_csv(save_path, index=False)
    
    return data
