from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import joblib

# Generate a random regression problem
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)

# Train a RandomForestRegressor model
model = RandomForestRegressor()
model.fit(X, y)

# Serialize the model to a file
joblib.dump(model, 'serialized_model.pkl')
