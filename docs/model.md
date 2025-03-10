# Model Documentation

## Model Architecture

The restaurant rating prediction model uses a Random Forest Regressor with the following specifications:

### Model Parameters
- n_estimators: 100
- max_depth: None
- min_samples_split: 2
- min_samples_leaf: 1
- random_state: 42

### Feature Engineering
1. **Numerical Features**
   - Votes: Log transformed
   - Cost for Two: Cleaned and normalized

2. **Categorical Features**
   - Location: One-hot encoded
   - Restaurant Type: Label encoded
   - Cuisines: TF-IDF vectorized
   - Online Order: Binary encoded
   - Book Table: Binary encoded

### Performance Metrics
- R² Score: 0.9121
- Mean Absolute Error: 0.234
- Root Mean Square Error: 0.312

## Training Process

1. **Data Preprocessing**
   - Handle missing values
   - Remove duplicates
   - Clean text data
   - Feature encoding

2. **Feature Selection**
   - Correlation analysis
   - Feature importance ranking
   - Variance threshold

3. **Model Training**
   - 80-20 train-test split
   - Cross-validation (5 folds)
   - Hyperparameter tuning using GridSearchCV

4. **Model Evaluation**
   - Performance metrics calculation
   - Feature importance analysis
   - Residual analysis

## Model Updates

The model is retrained monthly with new data to maintain accuracy. Version history:
- v1.0.0 (Current): Initial release with 91.21% accuracy
- Future updates will be logged here 