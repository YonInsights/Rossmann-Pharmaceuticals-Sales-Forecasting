import pytest
from flask import Flask, json

# Assuming your Flask app is named 'app' and is imported from your main application file
from models import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_prediction_endpoint(client):
    # Example payload for the prediction endpoint
    payload = {
        "store": 1,
        "day_of_week": 3,
        "date": "2023-10-10",
        "promo": 1,
        "state_holiday": "0",
        "school_holiday": 0
    }
    
    response = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert isinstance(data['prediction'], float)
