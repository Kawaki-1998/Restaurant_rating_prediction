from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict
from src.models.predict import RatingPredictor
import logging
import joblib
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import JSONResponse, HTMLResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Restaurant Rating Prediction API",
    description="API for predicting restaurant ratings based on various features",
    version="1.0.0"
)

# Create directories for static files and templates
static_dir = Path("static")
templates_dir = Path("templates")
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize predictor
predictor = RatingPredictor()

# Load model metadata
model_dir = Path('models')
with open(model_dir / 'model_metadata.json', 'r') as f:
    model_metadata = json.load(f)

class Restaurant(BaseModel):
    """Restaurant data model"""
    name: str
    location: str
    rest_type: str
    cuisines: str
    cost_for_two: str
    online_order: str
    book_table: str
    votes: int

class PredictionResponse(BaseModel):
    """Prediction response model"""
    restaurant_name: str
    predicted_rating: float

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    r2_score: float
    model_type: str
    parameters: Dict
    feature_count: int

class FeatureImportance(BaseModel):
    """Feature importance data"""
    feature_name: str
    importance_score: float

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Restaurant Rating Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_rating(restaurant: Restaurant):
    """
    Predict rating for a single restaurant
    
    Args:
        restaurant: Restaurant data
    
    Returns:
        Predicted rating
    """
    try:
        # Convert restaurant data to dictionary
        restaurant_data = restaurant.dict()
        
        # Make prediction
        prediction = predictor.predict(restaurant_data)
        
        return {
            "restaurant_name": restaurant.name,
            "predicted_rating": prediction
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_ratings_batch(restaurants: List[Restaurant]):
    """
    Predict ratings for multiple restaurants
    
    Args:
        restaurants: List of restaurant data
    
    Returns:
        List of predicted ratings
    """
    try:
        # Convert restaurants data to list of dictionaries
        restaurants_data = [restaurant.dict() for restaurant in restaurants]
        
        # Make predictions
        predictions = predictor.predict_batch(restaurants_data)
        
        return [
            {
                "restaurant_name": restaurant.name,
                "predicted_rating": prediction
            }
            for restaurant, prediction in zip(restaurants, predictions)
        ]
        
    except Exception as e:
        logger.error(f"Error making batch predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """
    Get model performance metrics
    
    Returns:
        Model metrics including R2 score, model type, and parameters
    """
    try:
        return {
            "r2_score": model_metadata["r2_score"],
            "model_type": model_metadata["model_type"],
            "parameters": model_metadata["parameters"],
            "feature_count": len(model_metadata["feature_names"])
        }
    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/feature_importance", response_model=List[FeatureImportance])
async def get_feature_importance():
    """
    Get feature importance scores
    
    Returns:
        List of features and their importance scores
    """
    try:
        feature_importance = []
        for feature_name, importance in zip(model_metadata["feature_names"], 
                                         predictor.model.feature_importances_):
            feature_importance.append({
                "feature_name": feature_name,
                "importance_score": float(importance)
            })
        
        # Sort by importance score in descending order
        feature_importance.sort(key=lambda x: x["importance_score"], reverse=True)
        return feature_importance
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/feature_importance_plot")
async def get_feature_importance_plot():
    """
    Get feature importance visualization
    
    Returns:
        Base64 encoded PNG image of feature importance plot
    """
    try:
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': model_metadata["feature_names"],
            'importance': predictor.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Create plot
        plt.figure(figsize=(12, 6))
        plt.barh(feature_importance.head(10)['feature'], 
                feature_importance.head(10)['importance'])
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()

        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # Encode plot as base64
        plot_base64 = base64.b64encode(buf.getvalue()).decode()
        
        return JSONResponse({
            "image": plot_base64,
            "content_type": "image/png"
        })
        
    except Exception as e:
        logger.error(f"Error generating feature importance plot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Render the dashboard page
    """
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request}
    )

@app.get("/api/dashboard_data")
async def dashboard_data():
    """
    Get all dashboard data in one call
    """
    try:
        # Get model metrics
        metrics = {
            "r2_score": model_metadata["r2_score"],
            "model_type": model_metadata["model_type"],
            "parameters": model_metadata["parameters"],
            "feature_count": len(model_metadata["feature_names"])
        }
        
        # Get feature importance
        feature_importance = []
        for feature_name, importance in zip(model_metadata["feature_names"], 
                                         predictor.model.feature_importances_):
            feature_importance.append({
                "feature_name": feature_name,
                "importance_score": float(importance)
            })
        feature_importance.sort(key=lambda x: x["importance_score"], reverse=True)
        
        # Generate feature importance plot
        plt.figure(figsize=(12, 6))
        plt.barh([f["feature_name"] for f in feature_importance[:10]], 
                [f["importance_score"] for f in feature_importance[:10]])
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode()
        
        return {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "feature_importance_plot": plot_base64
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 