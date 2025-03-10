import os
import json
import pytest
from fastapi.testclient import TestClient
from tests.mock_model import create_mock_model
from src.api.app import app

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_mock_model(tmp_path_factory):
    # Create a temporary directory for test models
    test_model_dir = tmp_path_factory.mktemp("models")
    model_path = os.path.join(test_model_dir, "best_model.joblib")
    
    # Define feature names
    feature_names = [
        'votes', 'cost_for_two', 'location_encoded', 'rest_type_encoded',
        'primary_cuisine_encoded', 'price_range_encoded', 'has_online_delivery',
        'has_table_booking', 'cuisine_popularity', 'location_popularity',
        'location_cuisine_popularity', 'log_cost'
    ]
    
    # Create mock model in the temporary directory
    model = create_mock_model(model_path)
    
    # Create model metadata file
    metadata = {
        "r2_score": 0.9121,
        "model_type": "RandomForestRegressor",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        "feature_names": feature_names,
        "feature_count": len(feature_names)
    }
    metadata_path = os.path.join(test_model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    # Set environment variable for model path
    os.environ["MODEL_PATH"] = str(test_model_dir)
    
    yield
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Restaurant Rating Prediction API"}

def test_predict_rating():
    test_restaurant = {
        "name": "Test Restaurant",
        "location": "Indiranagar",
        "rest_type": "Fine Dining",
        "cuisines": "North Indian",
        "cost_for_two": "2500",
        "online_order": "Yes",
        "book_table": "Yes",
        "votes": 1000
    }
    response = client.post("/predict", json=test_restaurant)
    assert response.status_code == 200
    
    data = response.json()
    assert "restaurant_name" in data
    assert "predicted_rating" in data
    assert data["restaurant_name"] == test_restaurant["name"]
    assert isinstance(data["predicted_rating"], (int, float))
    assert 0 <= data["predicted_rating"] <= 5  # Rating should be between 0 and 5

def test_model_metrics():
    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    
    # Check all required fields are present
    required_fields = ["r2_score", "model_type", "parameters", "feature_names", "feature_count"]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
    
    # Check specific values
    assert data["model_type"] == "RandomForestRegressor"
    assert isinstance(data["r2_score"], (int, float))
    assert isinstance(data["parameters"], dict)
    assert isinstance(data["feature_names"], list)
    assert data["feature_count"] == len(data["feature_names"])
    
    # Check feature names match our expected features
    expected_features = [
        'votes', 'cost_for_two', 'location_encoded', 'rest_type_encoded',
        'primary_cuisine_encoded', 'price_range_encoded', 'has_online_delivery',
        'has_table_booking', 'cuisine_popularity', 'location_popularity',
        'location_cuisine_popularity', 'log_cost'
    ]
    assert sorted(data["feature_names"]) == sorted(expected_features) 