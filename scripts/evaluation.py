import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

def save_model(model, path):
    """
    Save the model to a file.
    """
    joblib.dump(model, path)
    print(f"Model saved at {path}")

def load_model(path):
    """
    Load a model from a file.
    """
    return joblib.load(path)

def calculate_confidence_intervals(predictions, confidence=0.95):
    """
    Estimate confidence intervals for predictions.
    """
    mean = np.mean(predictions)
    std = np.std(predictions)
    interval = std * 1.96  # For 95% confidence
    return mean - interval, mean + interval
