import pytest
from flask import Flask
from flask.testing import FlaskClient
from app import app

@pytest.fixture
def client() -> FlaskClient:
    with app.test_client() as client:
        yield client

def test_predict(client: FlaskClient):
    response = client.post('/predict', json={'feature1': 1, 'feature2': 2})
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'predictions' in json_data
    assert isinstance(json_data['predictions'], list)
