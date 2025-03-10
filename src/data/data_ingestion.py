import os
import sys
import pandas as pd
import yaml
from src.utils.logger import logging

def read_yaml(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error reading yaml file: {e}")
        raise e

class DataIngestion:
    def __init__(self, config_path='src/config/config.yaml'):
        self.config = read_yaml(config_path)
        self.data_ingestion_config = self.config['data_ingestion']

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            # Read dataset
            df = pd.read_csv(self.data_ingestion_config['raw_data_path'])
            logging.info(f"Dataset read from {self.data_ingestion_config['raw_data_path']}")

            # Create processed data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.data_ingestion_config['processed_data_path']), 
                       exist_ok=True)

            # Save processed data
            df.to_csv(self.data_ingestion_config['processed_data_path'], index=False)
            logging.info(f"Data saved to {self.data_ingestion_config['processed_data_path']}")

            return self.data_ingestion_config['processed_data_path']

        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise e

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion() 