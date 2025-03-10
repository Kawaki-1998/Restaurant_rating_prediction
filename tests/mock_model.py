import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def create_mock_model(model_path):
    """Create a mock model for testing"""
    # Define feature names (must match FeatureEngineer)
    feature_names = [
        'votes', 'cost_for_two', 'location_encoded', 'rest_type_encoded',
        'primary_cuisine_encoded', 'price_range_encoded', 'has_online_delivery',
        'has_table_booking', 'cuisine_popularity', 'location_popularity',
        'location_cuisine_popularity', 'log_cost'
    ]
    
    # Create a simple random forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Create dummy data to fit the model
    X = pd.DataFrame(np.random.rand(100, len(feature_names)), columns=feature_names)
    y = np.random.rand(100)
    
    # Fit the model
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, model_path)
    
    return model 