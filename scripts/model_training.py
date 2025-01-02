
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import randint
import pandas as pd

def train_model(X, y):
    """
    Trains a Random Forest model with hyperparameter tuning using RandomizedSearchCV.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        
    Returns:
        model (RandomForestRegressor): Trained model.
        best_params (dict): Best hyperparameters from RandomizedSearchCV.
        rmse (float): Root Mean Squared Error of the model on the test set.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = RandomForestRegressor(random_state=42)

    # Define the hyperparameters to sample
    param_dist = {
        'n_estimators': randint(50, 150),
        'max_depth': [10, 20, None],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'bootstrap': [True, False]
    }

    # Use RandomizedSearchCV to sample from the parameter space
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=20, cv=2, n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Calculate the RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Best Hyperparameters: {best_params}")
    print(f"RMSE: {rmse}")

    return best_model, best_params, rmse