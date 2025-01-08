from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

def train_random_forest(X, y, params=None):
    """
    Train a Random Forest model.
    """
    if params is None:
        params = {'n_estimators': 100, 'random_state': 42}
    model = RandomForestRegressor(**params)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model using RMSE.
    """
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"RMSE: {rmse}")
    return rmse
