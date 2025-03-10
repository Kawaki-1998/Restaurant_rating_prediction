import pandas as pd
import joblib
from pathlib import Path
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def prepare_features(self, df):
        """
        Prepare features for prediction
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: Processed features
        """
        try:
            # Convert to numeric
            df['votes'] = pd.to_numeric(df['votes'])
            df['cost_for_two'] = pd.to_numeric(df['cost_for_two'])
            
            # Create binary features
            df['has_online_delivery'] = (df['online_order'] == 'Yes').astype(int)
            df['has_table_booking'] = (df['book_table'] == 'Yes').astype(int)
            
            # Basic encoding for categorical features
            df['location_encoded'] = pd.factorize(df['location'])[0]
            df['rest_type_encoded'] = pd.factorize(df['rest_type'])[0]
            
            # Extract primary cuisine
            df['primary_cuisine'] = df['cuisines'].str.split(',').str[0]
            df['primary_cuisine_encoded'] = pd.factorize(df['primary_cuisine'])[0]
            
            # Create price range feature (handle duplicate values)
            try:
                df['price_range_encoded'] = pd.qcut(df['cost_for_two'], q=5, labels=False, duplicates='drop')
            except ValueError:
                # If we can't create quantiles (e.g., all values are the same), use a single bin
                df['price_range_encoded'] = 0
            
            # Select features for model
            features = [
                'votes', 'cost_for_two', 'location_encoded', 'rest_type_encoded',
                'primary_cuisine_encoded', 'price_range_encoded', 'has_online_delivery',
                'has_table_booking'
            ]
            
            return df[features]
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise e

class RatingPredictor:
    def __init__(self):
        self.model = None
        self.model_dir = Path(os.getenv('MODEL_PATH', 'models'))
        self.feature_engineer = FeatureEngineer()
    
    def _load_model(self):
        """Load the model if not already loaded"""
        if self.model is None:
            try:
                model_path = self.model_dir / 'best_model.joblib'
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded successfully from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {str(e)}")
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
            # Load model if not already loaded
            self._load_model()
            
            # Convert input to DataFrame
            df = pd.DataFrame([restaurant_data])
            
            # Prepare features
            X = self.feature_engineer.prepare_features(df)
            
            # Ensure feature order matches model's expected features
            if hasattr(self.model, 'feature_names_in_'):
                required_features = list(self.model.feature_names_in_)
                missing_features = set(required_features) - set(X.columns)
                if missing_features:
                    # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
                    X = X.copy()
                    # Add missing features with default values
                    for feature in missing_features:
                        X[feature] = 0
                # Reorder columns to match model's expected order
                X = X[required_features]
            
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