# 🍽️ Restaurant Rating Prediction

A machine learning project that predicts restaurant ratings based on various features using restaurant data. This project includes exploratory data analysis (EDA), model development, and a web-based dashboard for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-orange.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/Kawaki-1998/Restaurant_rating_prediction/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Kawaki-1998/Restaurant_rating_prediction/actions/workflows/ci-cd.yml)

## 📊 Project Overview

This project analyzes restaurant data to predict restaurant ratings. It uses machine learning to understand what factors influence a restaurant's rating and provides a user-friendly interface for making predictions.

![Dashboard Overview](docs/images/dashboard/dashboard_overview.png)

### Key Features

- 🔍 **Exploratory Data Analysis**: Comprehensive analysis of restaurant data
- 🤖 **Machine Learning Model**: Random Forest model for accurate rating predictions
- 🌐 **Interactive Web Dashboard**: Real-time prediction interface
- 🐳 **Docker Support**: Easy deployment using containers
- 📈 **Feature Importance Analysis**: Understanding key factors affecting ratings

## 📊 Data Analysis Insights

### Rating Distribution
![Rating Distribution](docs/images/visualizations/rating_distribution.png)

*The distribution of restaurant ratings shows a balanced spread across different rating levels, with most restaurants falling in the 3.8-4.7 range.*

### Cuisine Analysis
![Cuisine Analysis](docs/images/visualizations/cuisine_analysis.png)

*Analysis shows the diversity of cuisine types in our dataset, with North Indian and South Indian cuisines being prominent.*

### Location Analysis
![Location Analysis](docs/images/visualizations/location_analysis.png)

*Geographic distribution of restaurants and their average ratings across different locations, showing both restaurant density and rating patterns.*

### Cost vs Rating Analysis
![Cost Analysis](docs/images/visualizations/cost_analysis.png)

*Relationship between restaurant cost and ratings, helping understand price-quality correlations.*

## 📈 Model Performance

![Model Performance](docs/images/dashboard/model_performance.png)

Our Random Forest model shows strong performance in predicting restaurant ratings:
- **Features Used**: Location, Cuisine Type, Cost for Two, Votes
- **Model Type**: Random Forest Regressor
- **Evaluation**: Strong correlation between predicted and actual ratings
- **Use Case**: Accurate rating predictions for new restaurants

## 🎯 Prediction Interface

![Prediction Interface](docs/images/dashboard/prediction_interface.png)

Our intuitive prediction interface provides a user-friendly form to predict restaurant ratings:

### Input Features
- **Restaurant Name**: Name of the establishment
- **Location**: Area or neighborhood location
- **Restaurant Type**: Category of the restaurant (e.g., Casual Dining, Fine Dining)
- **Cuisines**: Types of cuisine served
- **Cost for Two**: Average cost for two people
- **Online Order**: Whether online ordering is available (Yes/No)
- **Table Booking**: Whether table booking is available (Yes/No)
- **Votes**: Number of customer votes/reviews

### Features
- Real-time prediction using our trained Random Forest model
- Instant feedback with predicted rating score
- Clean and responsive design
- Form validation for accurate inputs
- Clear display of prediction results

The interface is designed to be intuitive and easy to use, making it simple for users to get accurate rating predictions based on restaurant characteristics.

## 🛠️ Technologies Used

- **Python**: Core programming language
- **scikit-learn**: Machine learning library
- **FastAPI**: Web API framework
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization

## 📁 Project Structure

```
Restaurant_rating_prediction/
├── data/
│   ├── raw/            # Raw data files
│   └── processed/      # Processed data files
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks
├── src/
│   ├── api/           # FastAPI application
│   ├── data/          # Data processing scripts
│   ├── models/        # Model training scripts
│   └── visualization/ # Visualization scripts
├── static/            # Static files for web dashboard
├── templates/         # HTML templates
├── tests/            # Unit tests
└── requirements.txt  # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kawaki-1998/Restaurant_rating_prediction.git
   cd Restaurant_rating_prediction
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the application locally**
   ```bash
   python -m uvicorn src.api.app:app --reload
   ```

### API Documentation

Access the API documentation at `http://localhost:8000/docs` for:
- Single prediction endpoint
- Batch prediction endpoint
- Model metrics endpoint

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Check out our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contact

Abhishek Ashok Hippargi - abhishekashokhippargi@gmail.com

Project Link: [https://github.com/Kawaki-1998/Restaurant_rating_prediction](https://github.com/Kawaki-1998/Restaurant_rating_prediction)

## 🙏 Acknowledgments

- Contributors and maintainers of the libraries used
- Everyone who helps improve this project 

## 📚 Documentation

- [Model Architecture and Training](docs/model.md)
- [API Documentation](http://localhost:8000/docs)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🚀 CI/CD Pipeline

This project uses GitHub Actions for Continuous Integration and Deployment:
- Automated testing
- Code quality checks
- Automated deployment to Render
- Coverage reporting 