import pandas as pd

def extract_date_features(df, date_column):
    # Ensure the date_column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Extract day, month, year, and day_of_week
    df['day'] = df[date_column].dt.day
    df['month'] = df[date_column].dt.month
    df['year'] = df[date_column].dt.year
    df['day_of_week'] = df[date_column].dt.dayofweek

    # Create flags for is_weekend and is_weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_weekday'] = ~df['is_weekend']

    # Identify the start_of_month, mid_month, and end_of_month
    df['start_of_month'] = df['day'] <= 10
    df['mid_month'] = (df['day'] > 10) & (df['day'] <= 20)
    df['end_of_month'] = df['day'] > 20

    return df
