from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# model_training.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(X, y):
    """
    Trains a Random Forest model with hyperparameter tuning using GridSearchCV.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        
    Returns:
        model (RandomForestRegressor): Trained model.
        best_params (dict): Best hyperparameters from GridSearchCV.
        rmse (float): Root Mean Squared Error of the model on the training set.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = RandomForestRegressor(random_state=42)

    # Define hyperparameters for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Calculate the RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Best Hyperparameters: {best_params}")
    print(f"RMSE: {rmse}")

    return best_model, best_params, rmse
