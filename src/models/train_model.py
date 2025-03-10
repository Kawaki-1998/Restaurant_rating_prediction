import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import joblib
import logging
from pathlib import Path
from src.preprocessing.feature_engineering import FeatureEngineer
import json

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        # Focus on Random Forest with fine-tuned parameters
        self.models = {
            'random_forest': {
                'model': RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1,
                    criterion='squared_error',
                    bootstrap=True
                ),
                'params': {
                    'n_estimators': [200, 300],
                    'max_depth': [15, 20, 25],
                    'min_samples_split': [5],
                    'min_samples_leaf': [2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            }
        }
        self.best_model = None
        self.best_score = float('-inf')
        
    def prepare_data(self, data_path):
        """Prepare data for model training"""
        try:
            # Read data
            df = pd.read_csv(data_path)
            
            # Prepare features
            X, y = self.feature_engineer.prepare_features(df)
            
            # Split data with a smaller test size for more training data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42
            )
            
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Testing data shape: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise e
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate feature importances
            feature_importance = pd.DataFrame({
                'feature': self.feature_engineer.get_feature_names(),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise e
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate model"""
        try:
            results = {}
            
            for name, model_info in self.models.items():
                logger.info(f"Training {name} with fine-tuned parameters...")
                
                # Perform grid search with stratified k-fold
                grid_search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=3,
                    scoring=['neg_mean_squared_error', 'r2'],
                    refit='r2',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Evaluate model
                evaluation = self.evaluate_model(best_model, X_test, y_test)
                
                results[name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    **evaluation
                }
                
                logger.info(f"{name} Results:")
                logger.info(f"Best Parameters: {grid_search.best_params_}")
                logger.info(f"RMSE: {evaluation['rmse']:.4f}")
                logger.info(f"MAE: {evaluation['mae']:.4f}")
                logger.info(f"R2 Score: {evaluation['r2']:.4f}")
                logger.info("\nTop 5 Important Features:")
                logger.info(evaluation['feature_importance'].head().to_string())
                
                # Update best model
                if evaluation['r2'] > self.best_score:
                    self.best_score = evaluation['r2']
                    self.best_model = best_model
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model training and evaluation: {str(e)}")
            raise e
    
    def save_model(self, model_dir='models'):
        """Save the best model and feature engineering objects"""
        try:
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save best model
            joblib.dump(self.best_model, model_dir / 'best_model.joblib')
            
            # Save feature engineering objects
            joblib.dump(self.feature_engineer, model_dir / 'feature_engineer.joblib')
            
            # Save model metadata
            metadata = {
                'model_type': type(self.best_model).__name__,
                'parameters': self.best_model.get_params(),
                'r2_score': self.best_score,
                'feature_names': self.feature_engineer.get_feature_names()
            }
            
            with open(model_dir / 'model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model, feature engineering objects, and metadata saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise e

def train():
    """Main training function"""
    try:
        logging.basicConfig(level=logging.INFO)
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Prepare data
        logger.info("Preparing data...")
        X_train, X_test, y_train, y_test = trainer.prepare_data('data/raw/zomato.csv')
        
        # Train and evaluate models
        logger.info("Training and evaluating models...")
        results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Save best model
        logger.info("Saving best model...")
        trainer.save_model()
        
        logger.info("Training completed successfully!")
        
        # Return results for analysis
        return results
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    train() 