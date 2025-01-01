
import pandas as pd
def select_features(data: pd.DataFrame):
    """
    Selects the relevant features and the target variable.
    
    Args:
        data (pd.DataFrame): The dataset.

    Returns:
        tuple: Features (X) and target variable (y).
    """
    features = ['Store', 'DayOfWeek', 'Open', 'Promo', 'CompetitionDistance', 'SchoolHoliday']

    # Ensure that all columns in 'features' are present in the dataset
    features = [col for col in features if col in data.columns]

    # Extracting features and target variable
    X = data[features]
    y = data['Sales']

    return X, y

