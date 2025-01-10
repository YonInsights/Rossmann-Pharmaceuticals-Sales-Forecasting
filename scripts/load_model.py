import joblib

def load_model(model_path):
    """
    Load a serialized model from the specified file path.

    Parameters:
    model_path (str): The path to the serialized model file.

    Returns:
    model: The loaded model.
    """
    model = joblib.load(model_path)
    return model
