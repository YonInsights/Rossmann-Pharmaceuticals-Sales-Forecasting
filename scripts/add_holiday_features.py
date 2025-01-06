import pandas as pd
import numpy as np

# Define holidays
holidays = pd.to_datetime(['2015-12-25', '2015-01-01', '2015-07-04'])  # Example holidays

# Calculate days to the next holiday
def days_to_holiday(date, holidays):
    days_to_next_holiday = (holidays - date).days
    days_to_next_holiday = days_to_next_holiday[days_to_next_holiday >= 0]
    if len(days_to_next_holiday) > 0:
        return days_to_next_holiday.min()
    else:
        return np.nan

# Calculate days after the last holiday
def days_after_holiday(date, holidays):
    days_after_last_holiday = (date - holidays).days
    days_after_last_holiday = days_after_last_holiday[days_after_last_holiday >= 0]
    if len(days_after_last_holiday) > 0:
        return days_after_last_holiday.min()
    else:
        return np.nan

# Apply the functions to the dataset
def add_holiday_features(df):
    df['days_to_holiday'] = df['Date'].apply(lambda x: days_to_holiday(x, holidays))
    df['days_after_holiday'] = df['Date'].apply(lambda x: days_after_holiday(x, holidays))

    # Create flags for specific holidays
    df['is_christmas'] = df['Date'].apply(lambda x: 1 if x in holidays else 0)
    df['is_new_year'] = df['Date'].apply(lambda x: 1 if x in holidays else 0)
    df['is_independence_day'] = df['Date'].apply(lambda x: 1 if x in holidays else 0)
    
    return df
