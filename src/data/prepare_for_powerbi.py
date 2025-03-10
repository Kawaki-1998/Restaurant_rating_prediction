import pandas as pd
import numpy as np
from pathlib import Path

def clean_rating(x):
    """Clean rating values"""
    if pd.isna(x) or x == 'NEW' or x == '-':
        return np.nan
    return float(str(x).split('/')[0])

def clean_cost(x):
    """Clean cost values"""
    if pd.isna(x):
        return np.nan
    return float(''.join(filter(str.isdigit, str(x))))

def extract_primary_cuisine(x):
    """Extract primary cuisine from cuisine list"""
    if pd.isna(x):
        return 'Unknown'
    return str(x).split(',')[0].strip()

def prepare_data_for_powerbi():
    """Prepare the Zomato dataset for Power BI visualization"""
    try:
        # Read the dataset
        df = pd.read_csv('data/raw/zomato.csv')
        
        # Clean ratings
        df['rating'] = df['rate'].apply(clean_rating)
        
        # Clean cost
        df['cost_for_two'] = df['approx_cost(for two people)'].apply(clean_cost)
        
        # Extract primary cuisine
        df['primary_cuisine'] = df['cuisines'].apply(extract_primary_cuisine)
        
        # Create price range categories
        df['price_range'] = pd.qcut(df['cost_for_two'].fillna(df['cost_for_two'].median()), 
                                  q=4, 
                                  labels=['Budget', 'Medium', 'High-End', 'Luxury'])
        
        # Calculate average rating by location
        location_ratings = df.groupby('location')['rating'].agg(['mean', 'count']).reset_index()
        location_ratings.columns = ['location', 'avg_rating', 'restaurant_count']
        
        # Calculate cuisine popularity
        cuisine_popularity = df['primary_cuisine'].value_counts().reset_index()
        cuisine_popularity.columns = ['cuisine', 'restaurant_count']
        
        # Create summary tables for Power BI
        summary_tables = {
            'restaurants': df[[
                'name', 'location', 'rating', 'votes', 'cost_for_two',
                'online_order', 'book_table', 'rest_type', 'primary_cuisine',
                'price_range'
            ]],
            'location_summary': location_ratings,
            'cuisine_summary': cuisine_popularity
        }
        
        # Create output directory
        output_dir = Path('dashboard/data')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save files for Power BI
        for name, data in summary_tables.items():
            output_file = output_dir / f'{name}.csv'
            data.to_csv(output_file, index=False)
            print(f"Saved {output_file}")
        
        print("\nData preparation completed successfully!")
        
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        raise e

if __name__ == "__main__":
    prepare_data_for_powerbi() 