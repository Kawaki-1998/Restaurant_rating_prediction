import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from fpdf import FPDF
import json
from datetime import datetime
from src.models.predict import RatingPredictor

class ModelPerformanceReport:
    def __init__(self, model_dir='models', data_path='data/raw/zomato.csv'):
        self.model_dir = Path(model_dir)
        self.data_path = data_path
        self.predictor = RatingPredictor()
        
        # Load metadata
        with open(self.model_dir / 'model_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Set style for plots
        plt.style.use('default')
        
    def create_example_predictions(self):
        """Create example predictions for different restaurant types"""
        example_restaurants = [
            {
                'name': 'Luxury Fine Dining',
                'location': 'Indiranagar',
                'rest_type': 'Fine Dining',
                'cuisines': 'North Indian, Continental',
                'cost_for_two': '2500',
                'online_order': 'No',
                'book_table': 'Yes',
                'votes': 1000
            },
            {
                'name': 'Casual Family Restaurant',
                'location': 'Koramangala',
                'rest_type': 'Casual Dining',
                'cuisines': 'South Indian, Chinese',
                'cost_for_two': '800',
                'online_order': 'Yes',
                'book_table': 'No',
                'votes': 500
            },
            {
                'name': 'Quick Service Restaurant',
                'location': 'BTM',
                'rest_type': 'Quick Bites',
                'cuisines': 'Fast Food, Beverages',
                'cost_for_two': '400',
                'online_order': 'Yes',
                'book_table': 'No',
                'votes': 200
            }
        ]
        
        predictions = []
        for restaurant in example_restaurants:
            pred = self.predictor.predict(restaurant)
            predictions.append({**restaurant, 'predicted_rating': pred})
        
        return predictions
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        feature_importance = pd.DataFrame({
            'feature': self.metadata['feature_names'],
            'importance': self.predictor.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create reports directory if it doesn't exist
        Path('reports').mkdir(exist_ok=True)
        
        # Create feature importance plot
        plt.figure(figsize=(12, 6))
        plt.barh(feature_importance.head(10)['feature'], feature_importance.head(10)['importance'])
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('reports/feature_importance.png')
        plt.close()
        
        return feature_importance
    
    def generate_pdf_report(self):
        """Generate PDF report with model performance analysis"""
        # Create reports directory if it doesn't exist
        Path('reports').mkdir(exist_ok=True)
        
        # Initialize PDF with Unicode support
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Restaurant Rating Prediction Model Performance Report', ln=True, align='C')
        pdf.ln(10)
        
        # Model Information
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '1. Model Information', ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, f"""
Model Type: {self.metadata['model_type']}
R2 Score: {self.metadata['r2_score']:.4f}
Number of Features: {len(self.metadata['feature_names'])}
        """)
        pdf.ln(5)
        
        # Feature Importance Analysis
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '2. Feature Importance Analysis', ln=True)
        feature_importance = self.analyze_feature_importance()
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, 'Top 5 Most Important Features:', ln=True)
        for _, row in feature_importance.head().iterrows():
            pdf.cell(0, 10, f"- {row['feature']}: {row['importance']:.4f}", ln=True)
        
        # Add feature importance plot
        pdf.image('reports/feature_importance.png', x=10, w=190)
        pdf.ln(10)
        
        # Example Predictions
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '3. Example Predictions', ln=True)
        pdf.set_font('Arial', '', 12)
        
        examples = self.create_example_predictions()
        for restaurant in examples:
            pdf.multi_cell(0, 10, f"""
Restaurant: {restaurant['name']}
Location: {restaurant['location']}
Type: {restaurant['rest_type']}
Cuisines: {restaurant['cuisines']}
Cost for Two: Rs. {restaurant['cost_for_two']}
Online Order: {restaurant['online_order']}
Table Booking: {restaurant['book_table']}
Votes: {restaurant['votes']}
Predicted Rating: {restaurant['predicted_rating']:.1f}/5.0
            """.encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(5)
        
        # Model Parameters
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '4. Model Parameters', ln=True)
        pdf.set_font('Arial', '', 12)
        for param, value in self.metadata['parameters'].items():
            pdf.multi_cell(0, 10, f"{param}: {value}")
        
        # Save report
        report_path = f'reports/model_performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        pdf.output(report_path)
        return report_path

def generate_report():
    """Generate the model performance report"""
    report_generator = ModelPerformanceReport()
    report_path = report_generator.generate_pdf_report()
    print(f"Report generated successfully: {report_path}")

if __name__ == "__main__":
    generate_report() 