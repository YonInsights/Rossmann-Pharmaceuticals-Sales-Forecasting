import logging
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def handle_missing_values(df, strategy='mean', constant_value=0):
    """
    Handle missing values in the dataset.
    """
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'constant':
        return df.fillna(constant_value)
    else:
        raise ValueError("Unsupported strategy for missing values.")

def encode_categorical_columns(df):
    """
    Encode categorical columns using Label Encoding.
    """
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def scale_numerical_columns(df, numeric_cols):
    """
    Scale numerical columns using StandardScaler.
    """
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
