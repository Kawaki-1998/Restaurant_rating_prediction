# 🍽️ Restaurant Rating Prediction

A machine learning project that predicts restaurant ratings based on various features using the Zomato dataset. This project includes exploratory data analysis (EDA), model development, and a web-based dashboard for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.0-orange.svg)
![Docker](https://img.shields.io/badge/Docker-27.5.1-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/Kawaki-1998/Restaurant_rating_prediction/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Kawaki-1998/Restaurant_rating_prediction/actions/workflows/ci-cd.yml)

## 📊 Project Overview

This project analyzes restaurant data from Zomato to predict restaurant ratings. It uses machine learning to understand what factors influence a restaurant's rating and provides a user-friendly interface for making predictions.

![Dashboard Overview](docs/images/dashboard.png)

### Key Features

- 🔍 **Exploratory Data Analysis**: Comprehensive analysis of restaurant data
- 🤖 **Machine Learning Model**: Random Forest model with 91.21% accuracy (R² score)
- 🌐 **Interactive Web Dashboard**: Real-time prediction interface
- 🐳 **Docker Support**: Easy deployment using containers
- 📈 **Feature Importance Analysis**: Understanding key factors affecting ratings

## 📊 Data Analysis Insights

### Rating Distribution
![Rating Distribution](docs/images/visualizations/rating_distribution.png)
*Distribution of restaurant ratings shows a normal distribution with a slight right skew*

### Top Cuisines Analysis
![Top Cuisines](docs/images/visualizations/top_cuisines.png)
*Analysis of the top-performing cuisines by average rating*

### Location Analysis
![Location Analysis](docs/images/visualizations/location_analysis.png)
*Geographic distribution of restaurants and their ratings*

### Service Analysis
![Service Analysis](docs/images/visualizations/services_analysis.png)
*Comparison of online ordering and table booking availability*

### Cost vs Rating Analysis
![Cost Analysis](docs/images/visualizations/cost_analysis.png)
*Relationship between restaurant cost and ratings*

### Feature Importance
![Feature Importance](docs/images/visualizations/feature_importance.png)
*Key factors influencing restaurant ratings*

## 🛠️ Technologies Used

- **Python**: Core programming language
- **scikit-learn**: Machine learning library
- **FastAPI**: Web API framework
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Docker**: Containerization

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
│   └── analysis/      # Analysis scripts
├── static/            # Static files for web dashboard
├── templates/         # HTML templates
├── tests/             # Unit tests
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
└── requirements.txt   # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional)

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

3. **Run the application**
   ```bash
   python -m uvicorn src.api.app:app --reload
   ```

### Using Docker

1. **Build and run with Docker**
   ```bash
   docker build -t restaurant-rating-app .
   docker run -p 8000:8000 restaurant-rating-app
   ```

2. **Or using Docker Compose**
   ```bash
   docker-compose up --build
   ```

## 🌐 Using the Web Dashboard

1. Access the dashboard at `http://localhost:8000/dashboard`
2. View model metrics and feature importance
3. Make real-time predictions using the form

## 📊 Model Performance

- **Model Type**: Random Forest Regressor
- **R² Score**: 0.9121 (91.21% accuracy)
- **Key Features**: Location, Cuisine Type, Cost for Two, Online Ordering

## 🔍 Feature Importance

Top factors affecting restaurant ratings:
1. Votes
2. Cost for Two
3. Location
4. Cuisine Type
5. Online Ordering Availability

## 📝 API Documentation

Access the API documentation at `http://localhost:8000/docs` for:
- Single prediction endpoint
- Batch prediction endpoint
- Model metrics endpoint
- Feature importance visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contact

Abhishek Ashok Hippargi - [@ahipp01998](https://twitter.com/ahipp01998)

Project Link: [https://github.com/Kawaki-1998/Restaurant_rating_prediction](https://github.com/Kawaki-1998/Restaurant_rating_prediction)

## 🙏 Acknowledgments

- Zomato for providing the dataset
- Contributors and maintainers of the libraries used
- Anyone who helps improve this project 

## 📸 Screenshots

### Dashboard
![Dashboard](docs/images/dashboard_full.png)

### Model Metrics
![Metrics](docs/images/metrics.png)

### Prediction Interface
![Prediction](docs/images/prediction.png)

## 📚 Documentation

- [Model Architecture and Training](docs/model.md)
- [API Documentation](http://localhost:8000/docs)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🚀 CI/CD Pipeline

This project uses GitHub Actions for Continuous Integration and Deployment:
- Automated testing
- Code quality checks
- Docker image building and testing
- Coverage reporting

## 📈 Model Performance Visualization

![Model Performance](docs/images/model_performance.png) 