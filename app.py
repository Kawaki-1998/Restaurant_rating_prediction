from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import yaml
from src.utils.logger import logging

app = FastAPI(
    title="Restaurant Rating Prediction",
    description="API for predicting restaurant ratings",
    version="1.0.0"
)

class PredictionInput(BaseModel):
    # Add your input features here
    pass

class PredictionOutput(BaseModel):
    predicted_rating: float

def load_model():
    try:
        with open('src/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        with open(config['model_trainer']['model_path'], 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e

@app.get("/")
def read_root():
    return {"message": "Welcome to Restaurant Rating Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        model = load_model()
        # Add prediction logic here
        prediction = 0.0  # Replace with actual prediction
        return PredictionOutput(predicted_rating=prediction)
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 