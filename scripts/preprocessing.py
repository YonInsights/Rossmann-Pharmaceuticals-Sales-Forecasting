import logging
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def handle_missing_values(df, strategy='mean', constant_value=0):
    """
    Handle missing values in the dataset.
    - strategy='mean': Replace missing values with the column mean (numeric columns only).
    - strategy='constant': Replace missing values with a constant value.
    """
    try:
        if strategy == 'mean':
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # Only apply mean for numeric columns
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    print(f"Skipping non-numeric column: {col}")
        elif strategy == 'constant':
            df.fillna(constant_value, inplace=True)
        else:
            raise ValueError("Unsupported strategy for missing values.")
        print("Missing values handled successfully.")
        return df
    except Exception as e:
        print(f"Error in handle_missing_values: {e}")
        raise
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
