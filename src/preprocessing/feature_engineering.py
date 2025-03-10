import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.standard_scaler = StandardScaler()
        self.imputers = {
            'numeric': SimpleImputer(strategy='median'),
            'categorical': SimpleImputer(strategy='constant', fill_value='missing')
        }
        
    def clean_rating(self, x):
        """Clean rating values"""
        try:
            if pd.isna(x) or x == 'NEW' or x == '-':
                return np.nan
            return float(str(x).split('/')[0])
        except:
            return np.nan
    
    def clean_cost(self, x):
        """Clean cost values"""
        try:
            if pd.isna(x):
                return np.nan
            return float(''.join(filter(str.isdigit, str(x))))
        except:
            return np.nan
    
    def extract_primary_cuisine(self, x):
        """Extract primary cuisine from cuisine list"""
        if pd.isna(x):
            return 'Unknown'
        return str(x).split(',')[0].strip()
    
    def create_features(self, df):
        """Create new features from existing ones"""
        try:
            # Create a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # For prediction, we don't need the rating
            if 'rate' in df.columns:
                df.loc[:, 'rating'] = df['rate'].apply(self.clean_rating)
                # Remove rows with missing ratings (since this is our target variable)
                df = df.dropna(subset=['rating'])
            
            # Clean cost
            df.loc[:, 'cost_for_two'] = df['approx_cost(for two people)' if 'approx_cost(for two people)' in df.columns else 'cost_for_two'].apply(self.clean_cost)
            
            # Extract primary cuisine
            df.loc[:, 'primary_cuisine'] = df['cuisines'].apply(self.extract_primary_cuisine)
            
            # Create price range categories based on fixed ranges
            cost = df['cost_for_two'].fillna(df['cost_for_two'].median() if len(df) > 1 else 1000)
            df.loc[:, 'price_range'] = pd.cut(
                cost,
                bins=[-np.inf, 500, 1000, 2000, np.inf],
                labels=['Budget', 'Medium', 'High-End', 'Luxury']
            )
            
            # Create binary features
            df.loc[:, 'has_online_delivery'] = (df['online_order'] == 'Yes').astype(int)
            df.loc[:, 'has_table_booking'] = (df['book_table'] == 'Yes').astype(int)
            
            # Create location popularity feature (with default values for new locations)
            if hasattr(self, 'location_counts'):
                df.loc[:, 'location_popularity'] = df['location'].map(self.location_counts).fillna(self.location_counts.median())
            else:
                self.location_counts = df['location'].value_counts()
                df.loc[:, 'location_popularity'] = df['location'].map(self.location_counts)
            
            # Create cuisine popularity feature (with default values for new cuisines)
            if hasattr(self, 'cuisine_counts'):
                df.loc[:, 'cuisine_popularity'] = df['primary_cuisine'].map(self.cuisine_counts).fillna(self.cuisine_counts.median())
            else:
                self.cuisine_counts = df['primary_cuisine'].value_counts()
                df.loc[:, 'cuisine_popularity'] = df['primary_cuisine'].map(self.cuisine_counts)
            
            # Log transform cost_for_two to handle skewness
            df.loc[:, 'log_cost'] = np.log1p(cost)
            
            # Create interaction features
            df.loc[:, 'location_cuisine_popularity'] = df['location_popularity'] * df['cuisine_popularity']
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature creation: {str(e)}")
            raise e
    
    def encode_categorical(self, df, columns):
        """Encode categorical variables"""
        try:
            df = df.copy()
            for col in columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df.loc[:, f"{col}_encoded"] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df.loc[:, f"{col}_encoded"] = self.label_encoders[col].transform(df[col].astype(str))
            return df
        except Exception as e:
            logger.error(f"Error in categorical encoding: {str(e)}")
            raise e
    
    def scale_numeric(self, df, columns):
        """Scale numeric features"""
        try:
            df = df.copy()
            # Convert columns to float64 before scaling
            df[columns] = df[columns].astype('float64')
            scaled_features = self.standard_scaler.fit_transform(df[columns])
            df.loc[:, columns] = scaled_features
            return df
        except Exception as e:
            logger.error(f"Error in numeric scaling: {str(e)}")
            raise e
    
    def handle_missing_values(self, df, numeric_columns, categorical_columns):
        """Handle missing values in the dataset"""
        try:
            df = df.copy()
            # Handle numeric missing values
            df.loc[:, numeric_columns] = self.imputers['numeric'].fit_transform(df[numeric_columns])
            
            # Handle categorical missing values
            df.loc[:, categorical_columns] = self.imputers['categorical'].fit_transform(df[categorical_columns])
            
            return df
        except Exception as e:
            logger.error(f"Error in missing value imputation: {str(e)}")
            raise e
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        try:
            # Create new features
            df = self.create_features(df)
            
            # Define column types
            numeric_columns = [
                'votes', 'cost_for_two', 'location_popularity', 'cuisine_popularity',
                'log_cost', 'location_cuisine_popularity'
            ]
            categorical_columns = ['location', 'rest_type', 'primary_cuisine', 'price_range']
            
            # Handle missing values
            df = self.handle_missing_values(df, numeric_columns, categorical_columns)
            
            # Encode categorical variables
            df = self.encode_categorical(df, categorical_columns)
            
            # Scale numeric features
            df = self.scale_numeric(df, numeric_columns)
            
            # Select features for modeling
            feature_columns = (
                numeric_columns + 
                [f"{col}_encoded" for col in categorical_columns] +
                ['has_online_delivery', 'has_table_booking']
            )
            
            if 'rating' in df.columns:
                return df[feature_columns], df['rating']
            else:
                return df[feature_columns]
            
        except Exception as e:
            logger.error(f"Error in feature preparation: {str(e)}")
            raise e
    
    def get_feature_names(self):
        """Get list of feature names used in the model"""
        numeric_columns = [
            'votes', 'cost_for_two', 'location_popularity', 'cuisine_popularity',
            'log_cost', 'location_cuisine_popularity'
        ]
        categorical_columns = ['location', 'rest_type', 'primary_cuisine', 'price_range']
        binary_columns = ['has_online_delivery', 'has_table_booking']
        
        feature_names = (
            numeric_columns + 
            [f"{col}_encoded" for col in categorical_columns] +
            binary_columns
        )
        return feature_names 