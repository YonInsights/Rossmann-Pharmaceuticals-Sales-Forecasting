# feature_engineering

def create_features(data):
    """
    Create new features for analysis.

    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: Dataset with new features.
    """
    # Example: Add a feature that calculates log transformation of a column
    if 'value' in data.columns:
        data['log_value'] = data['value'].apply(lambda x: np.log(x + 1))
    return data
