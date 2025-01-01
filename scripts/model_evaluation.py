from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Root Mean Squared Error (RMSE).

    Args:
        model: The trained model.
        X_test (pd.DataFrame): The test data features.
        y_test (pd.Series): The test data labels.

    Returns:
        float: The RMSE value.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return rmse
