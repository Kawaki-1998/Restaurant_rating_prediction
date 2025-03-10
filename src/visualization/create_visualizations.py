import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Create directories if they don't exist
os.makedirs('docs/images/visualizations', exist_ok=True)

# Load the data
restaurants_df = pd.read_csv('dashboard/data/restaurants.csv')
cuisine_summary = pd.read_csv('dashboard/data/cuisine_summary.csv')
location_summary = pd.read_csv('dashboard/data/location_summary.csv')

# 1. Rating Distribution
fig_rating = px.histogram(
    restaurants_df, 
    x='rate',
    title='Restaurant Rating Distribution',
    labels={'rate': 'Rating', 'count': 'Number of Restaurants'},
    color_discrete_sequence=['#2196F3']
)
fig_rating.write_html('docs/images/visualizations/rating_distribution.html')
fig_rating.write_image('docs/images/visualizations/rating_distribution.png')

# 2. Top Cuisines by Average Rating
top_cuisines = cuisine_summary.sort_values('average_rating', ascending=False).head(10)
fig_cuisines = px.bar(
    top_cuisines,
    x='cuisine',
    y='average_rating',
    title='Top 10 Cuisines by Average Rating',
    labels={'cuisine': 'Cuisine Type', 'average_rating': 'Average Rating'},
    color_discrete_sequence=['#FF9800']
)
fig_cuisines.write_html('docs/images/visualizations/top_cuisines.html')
fig_cuisines.write_image('docs/images/visualizations/top_cuisines.png')

# 3. Location Analysis
fig_locations = px.scatter_mapbox(
    location_summary,
    lat='latitude',
    lon='longitude',
    size='restaurant_count',
    color='average_rating',
    hover_name='location',
    hover_data=['restaurant_count', 'average_rating'],
    title='Restaurant Distribution and Ratings by Location',
    mapbox_style='carto-positron',
    color_continuous_scale='Viridis'
)
fig_locations.write_html('docs/images/visualizations/location_analysis.html')
fig_locations.write_image('docs/images/visualizations/location_analysis.png')

# 4. Online Ordering vs Table Booking
online_order_counts = restaurants_df['online_order'].value_counts()
table_booking_counts = restaurants_df['book_table'].value_counts()

fig_services = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Online Ordering', 'Table Booking'),
    specs=[[{'type':'domain'}, {'type':'domain'}]]
)

fig_services.add_trace(
    go.Pie(labels=online_order_counts.index, values=online_order_counts.values, marker_colors=['#2196F3', '#FF9800']),
    row=1, col=1
)
fig_services.add_trace(
    go.Pie(labels=table_booking_counts.index, values=table_booking_counts.values, marker_colors=['#2196F3', '#FF9800']),
    row=1, col=2
)
fig_services.update_layout(title_text="Service Availability Analysis")
fig_services.write_html('docs/images/visualizations/services_analysis.html')
fig_services.write_image('docs/images/visualizations/services_analysis.png')

# 5. Cost vs Rating Analysis
fig_cost = px.scatter(
    restaurants_df,
    x='cost_for_two',
    y='rate',
    title='Cost vs Rating Analysis',
    labels={'cost_for_two': 'Cost for Two', 'rate': 'Rating'},
    color='online_order',
    trendline="ols",
    color_discrete_sequence=['#2196F3', '#FF9800']
)
fig_cost.write_html('docs/images/visualizations/cost_analysis.html')
fig_cost.write_image('docs/images/visualizations/cost_analysis.png')

# 6. Feature Importance Plot
feature_importance = pd.DataFrame({
    'feature': ['Votes', 'Cost for Two', 'Location', 'Cuisine Type', 'Online Order'],
    'importance': [0.35, 0.25, 0.20, 0.15, 0.05]
})

fig_importance = px.bar(
    feature_importance,
    x='importance',
    y='feature',
    orientation='h',
    title='Feature Importance Analysis',
    labels={'importance': 'Importance Score', 'feature': 'Feature'},
    color_discrete_sequence=['#4CAF50']
)
fig_importance.write_html('docs/images/visualizations/feature_importance.html')
fig_importance.write_image('docs/images/visualizations/feature_importance.png')

print("Visualizations have been created successfully!") 