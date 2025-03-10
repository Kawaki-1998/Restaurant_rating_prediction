import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def create_mock_model():
    # Create a simple random forest model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Create some dummy data and fit the model
    X = np.random.rand(100, 5)  # 5 features
    y = np.random.rand(100)
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, 'models/best_model.joblib') 