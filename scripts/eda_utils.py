# eda_utils

def summarize_data(data):
    """
    Summarize the dataset with basic information.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        None
    """
    print("Dataset Overview:")
    print(data.info())
    print("\nDataset Statistics:")
    print(data.describe())
    print("\nMissing Values:")
    print(data.isnull().sum())

def count_unique_values(data, column):
    """
    Count unique values in a specific column.

    Args:
        data (pd.DataFrame): The dataset.
        column (str): Column name.

    Returns:
        pd.Series: Counts of unique values.
    """
    return data[column].value_counts()
