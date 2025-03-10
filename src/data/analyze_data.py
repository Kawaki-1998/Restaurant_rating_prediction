import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_dataset():
    try:
        # Read the dataset
        logger.info("Reading dataset...")
        df = pd.read_csv('data/raw/zomato.csv')
        
        # Basic information
        logger.info("\nDataset Shape:")
        logger.info(f"Number of rows: {df.shape[0]}")
        logger.info(f"Number of columns: {df.shape[1]}")
        
        # Column information
        logger.info("\nColumns in the dataset:")
        for col in df.columns:
            dtype = df[col].dtype
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            logger.info(f"{col}: {dtype}, Missing: {missing}, Unique values: {unique}")
        
        # Sample data
        logger.info("\nFirst few rows of the dataset:")
        print(df.head())
        
        # Basic statistics for numerical columns
        logger.info("\nBasic statistics for numerical columns:")
        print(df.describe())
        
        # Value counts for categorical columns
        logger.info("\nValue counts for categorical columns:")
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            logger.info(f"\nValue counts for {col}:")
            print(df[col].value_counts().head())
            
        # Save analysis results
        with open('data/raw/data_analysis.txt', 'w') as f:
            f.write("Dataset Analysis Results\n")
            f.write("=======================\n\n")
            f.write(f"Number of rows: {df.shape[0]}\n")
            f.write(f"Number of columns: {df.shape[1]}\n\n")
            f.write("Column Information:\n")
            for col in df.columns:
                dtype = df[col].dtype
                missing = df[col].isnull().sum()
                unique = df[col].nunique()
                f.write(f"{col}: {dtype}, Missing: {missing}, Unique values: {unique}\n")
        
        logger.info("\nAnalysis complete. Results saved to data/raw/data_analysis.txt")
        return df
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}")
        raise e

if __name__ == "__main__":
    analyze_dataset() 