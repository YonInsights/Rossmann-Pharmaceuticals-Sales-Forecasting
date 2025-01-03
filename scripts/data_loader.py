import os
import pandas as pd

def load_data(file_name):
    """
    Load data from a CSV file in the data folder.
    
    Parameters:
    file_name (str): The name of the CSV file to load.
    
    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    file_path = os.path.join(data_folder, file_name)
    return pd.read_csv(file_path)
