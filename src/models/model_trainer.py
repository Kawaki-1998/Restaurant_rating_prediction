import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import pickle
import json
import logging
import mlflow
import mlflow.sklearn

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {
            'xgboost': xgb.XGBRegressor(**config['model_params']['xgboost']),
            'lightgbm': lgb.LGBMRegressor(**config['model_params']['lightgbm']),
            'random_forest': RandomForestRegressor(**config['model_params']['random_forest'])
        }
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        
    def train_and_evaluate(self, X, y):
        """Train and evaluate multiple models"""
        logging.info("Starting model training and evaluation")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['model_trainer']['test_size'],
                random_state=self.config['model_trainer']['random_state']
            )
            
            best_score = float('inf')
            
            # Train and evaluate each model
            for model_name, model in self.models.items():
                logging.info(f"Training {model_name}")
                
                # Start MLflow run
                with mlflow.start_run(run_name=model_name):
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Perform cross-validation
                    cv_scores = cross_val_score(
                        model, X, y, cv=5, scoring='neg_mean_squared_error'
                    )
                    cv_rmse = np.sqrt(-cv_scores.mean())
                    
                    # Log metrics with MLflow
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("mae", mae)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("cv_rmse", cv_rmse)
                    
                    # Log model
                    mlflow.sklearn.log_model(model, model_name)
                    
                    # Store metrics
                    self.metrics[model_name] = {
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'cv_rmse': cv_rmse
                    }
                    
                    logging.info(f"{model_name} metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, CV_RMSE={cv_rmse:.4f}")
                    
                    # Update best model if current model is better
                    if rmse < best_score:
                        best_score = rmse
                        self.best_model = model
                        self.best_model_name = model_name
            
            # Save best model
            self.save_model()
            
            # Save metrics
            self.save_metrics()
            
            logging.info(f"Best model: {self.best_model_name} with RMSE: {best_score:.4f}")
            return self.metrics
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise e
    
    def save_model(self):
        """Save the best model to disk"""
        try:
            with open(self.config['model_trainer']['model_path'], 'wb') as f:
                pickle.dump(self.best_model, f)
            logging.info(f"Best model saved to {self.config['model_trainer']['model_path']}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise e
    
    def save_metrics(self):
        """Save metrics to disk"""
        try:
            with open(self.config['model_trainer']['metric_path'], 'w') as f:
                json.dump(self.metrics, f, indent=4)
            logging.info(f"Metrics saved to {self.config['model_trainer']['metric_path']}")
        except Exception as e:
            logging.error(f"Error saving metrics: {str(e)}")
            raise e
    
    def load_model(self):
        """Load model from disk"""
        try:
            with open(self.config['model_trainer']['model_path'], 'rb') as f:
                self.best_model = pickle.load(f)
            logging.info(f"Model loaded from {self.config['model_trainer']['model_path']}")
            return self.best_model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise e 