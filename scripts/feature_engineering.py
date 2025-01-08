import pandas as pd

def add_date_features(df, date_column):
    """
    Add features related to dates.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['month'] = df[date_column].dt.month
    df['year'] = df[date_column].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def add_rolling_features(df, target_column, window_size=7):
    """
    Add rolling average features.
    """
    df[f'rolling_avg_{window_size}'] = df[target_column].rolling(window=window_size).mean()
    return df
