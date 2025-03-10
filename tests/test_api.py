from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

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
    assert "predicted_rating" in response.json()
    assert "restaurant_name" in response.json()

def test_model_metrics():
    response = client.get("/model/metrics")
    assert response.status_code == 200
    assert "r2_score" in response.json()
    assert "model_type" in response.json()
    assert "parameters" in response.json()
    assert "feature_count" in response.json() 