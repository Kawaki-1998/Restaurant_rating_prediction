import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories if they don't exist
os.makedirs('docs/images/dashboard', exist_ok=True)
os.makedirs('docs/images/visualizations', exist_ok=True)

# Enhanced sample data for better visualizations
restaurants_data = {
    'name': ['Restaurant A', 'Restaurant B', 'Restaurant C', 'Restaurant D', 'Restaurant E', 
             'Restaurant F', 'Restaurant G', 'Restaurant H', 'Restaurant I', 'Restaurant J'],
    'rating': [4.5, 3.8, 4.2, 3.9, 4.7, 3.5, 4.1, 4.3, 3.7, 4.4],
    'cost': [1200, 800, 1500, 600, 2000, 900, 1300, 1800, 700, 1600],
    'cuisine': ['North Indian', 'Chinese', 'Italian', 'South Indian', 'Continental',
                'North Indian', 'Chinese', 'Italian', 'South Indian', 'Continental'],
    'location': ['Indiranagar', 'Koramangala', 'HSR Layout', 'Whitefield', 'MG Road',
                 'JP Nagar', 'Marathahalli', 'Electronic City', 'BTM Layout', 'Jayanagar'],
    'service_rating': [4.2, 3.5, 4.0, 3.8, 4.5, 3.3, 4.2, 4.1, 3.6, 4.3],
    'online_order': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(restaurants_data)

# Set style for better visualizations
plt.style.use('default')

# 1. Rating Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='rating', bins=20, kde=True)
plt.title('Restaurant Rating Distribution', fontsize=14, pad=20)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('docs/images/visualizations/rating_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Top Cuisines Analysis
plt.figure(figsize=(12, 6))
cuisine_avg = df.groupby('cuisine')['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False)
sns.barplot(x=cuisine_avg.index, y=cuisine_avg['mean'])
plt.title('Average Rating by Cuisine Type', fontsize=14, pad=20)
plt.xticks(rotation=45)
plt.xlabel('Cuisine', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.savefig('docs/images/visualizations/cuisine_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Location Analysis
plt.figure(figsize=(12, 6))
location_avg = df.groupby('location')['rating'].mean().sort_values(ascending=False)
sns.barplot(x=location_avg.index, y=location_avg.values)
plt.title('Average Rating by Location', fontsize=14, pad=20)
plt.xticks(rotation=45)
plt.xlabel('Location', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.savefig('docs/images/visualizations/location_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Service Analysis
plt.figure(figsize=(12, 6))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Service Rating Distribution
sns.histplot(data=df, x='service_rating', bins=20, kde=True, ax=ax1)
ax1.set_title('Service Rating Distribution', fontsize=12)
ax1.set_xlabel('Service Rating', fontsize=10)
ax1.set_ylabel('Count', fontsize=10)

# Online Order Analysis
online_order_avg = df.groupby('online_order')['rating'].mean()
sns.barplot(x=online_order_avg.index, y=online_order_avg.values, ax=ax2)
ax2.set_title('Average Rating by Online Order Availability', fontsize=12)
ax2.set_xlabel('Online Order Available', fontsize=10)
ax2.set_ylabel('Average Rating', fontsize=10)

plt.tight_layout()
plt.savefig('docs/images/visualizations/service_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Cost vs Rating Analysis
plt.figure(figsize=(12, 6))
sns.regplot(data=df, x='cost', y='rating', scatter_kws={'alpha':0.5})
plt.title('Cost vs Rating Relationship', fontsize=14, pad=20)
plt.xlabel('Cost for Two (Rs.)', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.savefig('docs/images/visualizations/cost_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Feature Importance
plt.figure(figsize=(12, 6))
feature_importance = {
    'Service Rating': 0.30,
    'Cost': 0.25,
    'Location': 0.20,
    'Cuisine': 0.15,
    'Online Order': 0.10
}
plt.barh(list(feature_importance.keys()), list(feature_importance.values()))
plt.title('Feature Importance in Rating Prediction', fontsize=14, pad=20)
plt.xlabel('Importance Score', fontsize=12)
plt.gca().invert_yaxis()
plt.savefig('docs/images/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Dashboard Overview
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Rating Distribution', 'Top Cuisines', 'Location Analysis', 'Cost vs Rating')
)

# Rating Distribution
fig.add_trace(go.Histogram(x=df['rating'], nbinsx=20, name='Rating'), row=1, col=1)

# Top Cuisines
cuisine_avg = df.groupby('cuisine')['rating'].mean().sort_values(ascending=False)
fig.add_trace(go.Bar(x=cuisine_avg.index, y=cuisine_avg.values, name='Cuisine'), row=1, col=2)

# Location Analysis
location_avg = df.groupby('location')['rating'].mean().sort_values(ascending=False)
fig.add_trace(go.Bar(x=location_avg.index, y=location_avg.values, name='Location'), row=2, col=1)

# Cost vs Rating
fig.add_trace(go.Scatter(x=df['cost'], y=df['rating'], mode='markers', name='Cost'), row=2, col=2)

fig.update_layout(height=800, width=1200, title_text="Restaurant Analysis Dashboard", showlegend=False)
fig.write_image('docs/images/dashboard/dashboard_overview.png')

print("All visualizations have been generated successfully!") 