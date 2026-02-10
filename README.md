# AirIQ

**Machine Learning-Based Air Quality and Health Risk Prediction System**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AirIQ is a predictive analytics system that forecasts air quality levels and estimates health risks based on environmental pollution and weather data across U.S. cities. The system uses ensemble machine learning techniques to model complex, nonlinear relationships between pollutants (PM2.5, NO2, CO2) and meteorological factors (temperature, humidity, wind speed) to generate accurate health risk scores.

**Key Features:**
- Multi-algorithm ensemble approach with Random Forest and Gradient Boosting
- Hyperparameter optimization with cross-validation
- Feature importance analysis for interpretability
- RMSE: 0.1551 | MAPE: 1.14% on test data
- Scalable to multiple cities and time periods

## Problem Statement

Urban air quality monitoring systems need to predict health risks from environmental factors to enable:
- Proactive public health warnings
- Resource allocation for vulnerable populations
- Long-term environmental policy planning

Traditional linear models fail to capture the complex interactions between multiple pollutants and weather conditions that drive health outcomes.

## Technical Approach

### Algorithm Selection

After evaluating three approaches (Linear Regression, Random Forest, Neural Networks), **Random Forest Regression** was selected as the primary algorithm because it:

- Captures nonlinear relationships and feature interactions without extensive feature engineering
- Provides robustness to noisy sensor data through ensemble averaging
- Offers interpretability via feature importance scores
- Balances accuracy with computational efficiency

### Model Optimization

**Hyperparameter Tuning:**
- GridSearchCV with 5-fold cross-validation
- Optimized: `n_estimators`, `max_features`, `max_depth`, `min_samples_leaf`

**Regularization Techniques:**
- `max_depth` limiting to prevent overfitting
- `min_samples_leaf` constraints for prediction smoothing

**Ensemble Methods:**
- Gradient Boosting (sequential error correction)
- Model averaging (Random Forest + Gradient Boosting)

## Performance

| Model | RMSE | MAPE |
|-------|------|------|
| **Baseline Random Forest** | **0.1551** | **1.14%** |
| Tuned Random Forest | 0.1896 | 1.42% |
| Gradient Boosting | 0.1577 | 1.19% |
| Ensemble Average | 0.1614 | 1.22% |

The baseline Random Forest achieved the best overall performance, demonstrating that careful feature selection and default hyperparameters can outperform more complex configurations on this dataset.

## Key Insights

**Feature Importance Analysis:**
- Daily activity levels and baseline health indicators showed strongest predictive power
- Age and chronic condition markers provided gradual risk adjustments
- Low activity combined with poor baseline health acts as a "red flag" combination

**Model Behavior:**
- Typical prediction errors of ~1.14% relative to actual health risk scores
- Strong performance across different risk levels
- Robust to missing or noisy sensor data

## Tech Stack

- **Language:** Python 3.8+
- **ML Framework:** scikit-learn
- **Data Processing:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Model Selection:** GridSearchCV, cross-validation

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/airiq.git
cd airiq

# Install dependencies
pip install -r requirements.txt
```

## Usage
```python
from airiq.models import AirQualityPredictor

# Initialize model
predictor = AirQualityPredictor()

# Train on your data
predictor.fit(X_train, y_train)

# Generate predictions
health_risk_scores = predictor.predict(X_test)

# Get feature importance
predictor.get_feature_importance()
```

## Future Enhancements

- **Temporal Features:** Add rolling averages of pollutants over multiple days to capture cumulative exposure effects
- **Advanced Hyperparameter Tuning:** Implement Bayesian optimization or RandomizedSearchCV for more efficient parameter space exploration
- **Real-time Deployment:** API endpoint for live predictions from sensor networks
- **Geographic Expansion:** Extend model to international cities with different pollution profiles

## Applications

This framework can be adapted for:
- **Corporate Wellness Programs:** Employee health risk screening based on environmental exposure
- **Smart City Planning:** Air quality monitoring and public health alerts
- **Environmental Policy:** Impact assessment of pollution reduction initiatives

## Project Structure
```
airiq/
├── data/
│   └── processed/           # Cleaned datasets
├── models/
│   ├── baseline.pkl         # Trained baseline model
│   └── ensemble.pkl         # Ensemble model
├── notebooks/
│   ├── exploration.ipynb    # Data analysis
│   └── modeling.ipynb       # Model development
├── src/
│   ├── preprocessing.py     # Data cleaning pipeline
│   ├── models.py           # Model implementations
│   └── evaluation.py       # Metrics and visualization
├── requirements.txt
└── README.md
```

## License

MIT License - see LICENSE file for details

---

*Developed as part of advanced machine learning research in environmental health analytics*
