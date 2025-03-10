import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

class DataPreprocessor:
    def __init__(self):
        self.numerical_features = None
        self.categorical_features = None
        self.label_encoders = {}
        self.preprocessor = None
        
    def _identify_features(self, df):
        """Identify numerical and categorical features"""
        # Exclude the target variable 'rate'
        features = df.columns.drop('rate') if 'rate' in df.columns else df.columns
        
        self.numerical_features = df[features].select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df[features].select_dtypes(
            include=['object']).columns.tolist()
            
        logging.info(f"Numerical features: {self.numerical_features}")
        logging.info(f"Categorical features: {self.categorical_features}")
        
    def fit_transform(self, df):
        """Fit and transform the data"""
        logging.info("Starting data preprocessing")
        
        try:
            # Identify features
            self._identify_features(df)
            
            # Create preprocessing pipelines
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('label_encoding', self._LabelEncoderTransformer())
            ])
            
            # Combine pipelines
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, self.numerical_features),
                    ('cat', categorical_pipeline, self.categorical_features)
                ])
            
            # Fit and transform
            transformed_array = self.preprocessor.fit_transform(df)
            
            # Create transformed dataframe
            feature_names = (self.numerical_features + 
                           [f"{col}_encoded" for col in self.categorical_features])
            transformed_df = pd.DataFrame(
                transformed_array, 
                columns=feature_names,
                index=df.index
            )
            
            # Add target variable if it exists
            if 'rate' in df.columns:
                transformed_df['rate'] = df['rate']
            
            logging.info("Data preprocessing completed successfully")
            return transformed_df
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise e
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        try:
            transformed_array = self.preprocessor.transform(df)
            feature_names = (self.numerical_features + 
                           [f"{col}_encoded" for col in self.categorical_features])
            transformed_df = pd.DataFrame(
                transformed_array, 
                columns=feature_names,
                index=df.index
            )
            return transformed_df
            
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise e
    
    class _LabelEncoderTransformer:
        """Custom transformer for label encoding"""
        def __init__(self):
            self.label_encoders = {}
            
        def fit(self, X, y=None):
            X = pd.DataFrame(X)
            for column in X.columns:
                self.label_encoders[column] = LabelEncoder()
                self.label_encoders[column].fit(X[column].astype(str))
            return self
            
        def transform(self, X):
            X = pd.DataFrame(X)
            X_encoded = X.copy()
            for column in X.columns:
                X_encoded[column] = self.label_encoders[column].transform(
                    X[column].astype(str))
            return X_encoded 