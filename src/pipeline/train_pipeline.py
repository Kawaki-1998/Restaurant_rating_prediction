import os
import yaml
import pandas as pd
from src.data.data_ingestion import DataIngestion
from src.preprocessing.preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer
import logging
import mlflow

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def train_pipeline():
    try:
        # Set up MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("restaurant_rating_prediction")
        
        # Load configuration
        config = read_yaml('src/config/config.yaml')
        
        # Data Ingestion
        logging.info("Starting data ingestion")
        data_ingestion = DataIngestion()
        data_path = data_ingestion.initiate_data_ingestion()
        
        # Load Data
        df = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully with shape: {df.shape}")
        
        # Data Preprocessing
        logging.info("Starting data preprocessing")
        preprocessor = DataPreprocessor()
        
        # Separate features and target
        X = df.drop('rate', axis=1)
        y = df['rate']
        
        # Preprocess features
        X_processed = preprocessor.fit_transform(X)
        logging.info(f"Data preprocessing completed. Processed shape: {X_processed.shape}")
        
        # Model Training
        logging.info("Starting model training")
        model_trainer = ModelTrainer(config)
        metrics = model_trainer.train_and_evaluate(X_processed, y)
        
        logging.info("Training pipeline completed successfully")
        return metrics
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        metrics = train_pipeline()
        print("\nTraining Metrics:")
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name}:")
            for metric_name, value in model_metrics.items():
                print(f"{metric_name}: {value:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}") 