import pandas as pd
import joblib
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

class RatingPredictor:
    def __init__(self):
        self.model = None
        self.model_dir = Path(os.getenv('MODEL_PATH', 'models'))
        
    def _load_model(self):
        """Lazy load the model when needed"""
        if self.model is None:
            try:
                model_path = self.model_dir / 'best_model.joblib'
                self.model = joblib.load(model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    
    def predict(self, restaurant_data):
        """
        Make predictions for restaurant rating
        
        Args:
            restaurant_data (dict): Dictionary containing restaurant information
                Required keys:
                - name: restaurant name
                - location: restaurant location
                - rest_type: restaurant type
                - cuisines: comma-separated cuisines
                - cost_for_two: cost for two people
                - online_order: 'Yes' or 'No'
                - book_table: 'Yes' or 'No'
                - votes: number of votes
        
        Returns:
            float: Predicted rating
        """
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([restaurant_data])
            
            # Prepare features
            X = self.feature_engineer.prepare_features(df)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            return round(prediction, 1)
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise e
    
    def predict_batch(self, restaurants_data):
        """
        Make predictions for multiple restaurants
        
        Args:
            restaurants_data (list): List of dictionaries containing restaurant information
        
        Returns:
            list: List of predicted ratings
        """
        try:
            # Convert input to DataFrame
            df = pd.DataFrame(restaurants_data)
            
            # Prepare features
            X = self.feature_engineer.prepare_features(df)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            return [round(pred, 1) for pred in predictions]
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            raise e

    def get_feature_importance(self):
        """Get feature importance from the model"""
        self._load_model()  # Lazy load the model
        return self.model.feature_importances_

def create_sample_prediction():
    """Create a sample prediction to test the model"""
    try:
        predictor = RatingPredictor()
        
        # Sample restaurant data
        sample_restaurant = {
            'name': 'Sample Restaurant',
            'location': 'Banashankari',
            'rest_type': 'Casual Dining',
            'cuisines': 'North Indian, Chinese',
            'cost_for_two': '800',
            'online_order': 'Yes',
            'book_table': 'Yes',
            'votes': 100
        }
        
        # Make prediction
        prediction = predictor.predict(sample_restaurant)
        
        logger.info(f"Sample prediction completed. Predicted rating: {prediction}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error in sample prediction: {str(e)}")
        raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_sample_prediction() 