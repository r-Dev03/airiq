# AirIQ

**Machine Learning-Based Air Quality and Health Risk Prediction**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AirIQ is a machine learning project that predicts health risk scores based on air quality and environmental data. Using ensemble methods with Random Forest and Gradient Boosting, it demonstrates model optimization through hyperparameter tuning and regularization techniques.

**Key Results:**
- Baseline Random Forest: RMSE 0.1551, MAPE 1.14%
- Tuned Random Forest: RMSE 0.1896, MAPE 1.42%
- Gradient Boosting: RMSE 0.1577, MAPE 1.19%
- Ensemble Average: RMSE 0.1614, MAPE 1.22%

## Problem Statement

Urban air quality affects public health, but predicting health risk from environmental factors requires analyzing multiple variables simultaneously. This project demonstrates how machine learning can predict health risk scores from air quality metrics and weather conditions.

## Tech Stack

- **Python 3.8+**
- **scikit-learn** - Random Forest, Gradient Boosting, GridSearchCV
- **pandas** - Data loading and manipulation
- **openpyxl** - Excel file reading

## Data Features

**Input Features (14 total):**
- **Air Quality:** PM2.5, NO2, CO2
- **Weather:** Temperature, humidity, wind speed, pressure, cloud cover, visibility, solar radiation, UV index
- **Temporal:** Month, day of week, weekend flag

**Target Variable:**
- Health risk score (continuous value)

## Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version
```

### Option 1: Standard Python Setup
```bash
# Clone repository
git clone https://github.com/yourusername/airiq.git
cd airiq

# Install dependencies
pip install pandas scikit-learn openpyxl
```

### Option 2: Nix Development Environment
```bash
# If you use Nix
nix develop

# Dependencies are automatically available
```

**Required packages:**
- pandas
- scikit-learn
- openpyxl (for reading Excel files)

## Usage

### Running the Model
```bash
# Make sure DQN1 Dataset.xlsx is in the same directory
python random_forest.py
```

### Expected Output
```
Number of training samples: 2611
Number of test samples: 653

=== Baseline Random Forest ===
RMSE: 0.1551
MAPE: 1.14%

=== Tuned Random Forest (Optimization + Regularization) ===
Best hyperparameters: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 300}
RMSE: 0.1896
MAPE: 1.42%

=== Gradient Boosting Model ===
RMSE: 0.1577
MAPE: 1.19%

=== Averaged Ensemble (Tuned RF + GB) ===
RMSE: 0.1614
MAPE: 1.22%
```

## Technical Approach

### 1. Baseline Model

Simple Random Forest with default hyperparameters:
- 200 estimators
- No depth limit
- Establishes performance baseline

### 2. Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation:
```python
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
```

**Optimization Goal:** Minimize RMSE through systematic hyperparameter search

### 3. Regularization

Constrain model complexity to prevent overfitting:
- `max_depth`: Limit tree depth
- `min_samples_leaf`: Require minimum samples per leaf node

### 4. Ensemble Learning

Two-model ensemble:
- Tuned Random Forest (optimized via GridSearch)
- Histogram-based Gradient Boosting
- **Combination:** Simple averaging (0.5 * RF + 0.5 * GB)

## Model Comparison

| Model | RMSE | MAPE | Notes |
|-------|------|------|-------|
| Baseline RF | 0.1551 | 1.14% | Best single model |
| Tuned RF | 0.1896 | 1.42% | More complex, slightly overfit |
| Gradient Boosting | 0.1577 | 1.19% | Sequential error correction |
| Ensemble Average | 0.1614 | 1.22% | Balanced approach |

**Key Insight:** Baseline Random Forest achieved the best performance, suggesting the default hyperparameters were well-suited for this dataset. The tuned model's higher error indicates overfitting despite regularization attempts.

## Project Structure
```
airiq/
├── random_forest.py         # Main training and evaluation script
├── DQN1 Dataset.xlsx        # Input data
├── DQN1 Case Study.docx     # Project documentation
├── report.md                # Project report
├── flake.nix                # Nix development environment
├── flake.lock               # Nix lock file
├── README.md
└── LICENSE
```

## Code Overview

The implementation is straightforward:
```python
# Load data
data = pd.read_excel("DQN1 Dataset.xlsx")

# Split features/target
X = data[feature_cols]
y = data[target_col]

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train baseline model
baseline_rf = RandomForestRegressor(n_estimators=200)
baseline_rf.fit(X_train, y_train)

# Optimize with GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Train gradient boosting
gb_model = HistGradientBoostingRegressor()
gb_model.fit(X_train, y_train)

# Ensemble predictions
y_pred_ensemble = 0.5 * y_pred_rf + 0.5 * y_pred_gb
```

## Evaluation Metrics

**RMSE (Root Mean Squared Error):**
- Measures average prediction error
- Lower is better
- Penalizes large errors more heavily

**MAPE (Mean Absolute Percentage Error):**
- Percentage-based error metric
- Easier to interpret (e.g., "1.14% average error")
- Lower is better

## Limitations

**Current Constraints:**
- Fixed 80/20 train/test split (no k-fold CV for final evaluation)
- Simple ensemble averaging (no weighted or stacking approaches)
- No feature engineering beyond provided dataset
- No time-series considerations (treats data as i.i.d.)
- Excel-only data input

**Model Limitations:**
- Tuned model performed worse than baseline (overfitting)
- No cross-validation for gradient boosting hyperparameters
- Fixed ensemble weights (0.5/0.5) without optimization

## Future Enhancements

**Model Improvements:**
- Feature engineering (interactions, polynomials, lag features)
- More sophisticated ensemble methods (stacking, boosting)
- Deep learning approaches (neural networks)
- Time-series modeling if temporal patterns exist

**Technical Improvements:**
- Cross-validation for final model evaluation
- Hyperparameter tuning for gradient boosting
- Automated model selection
- Feature importance analysis visualization

**Production Features:**
- CSV/JSON data input support
- Model persistence (save/load trained models)
- API endpoint for predictions
- Visualization dashboard for predictions and feature importance

## Dataset

The project uses `DQN1 Dataset.xlsx` containing air quality and environmental measurements with corresponding health risk scores.

## License

MIT License - see LICENSE file for details

---

*A demonstration of machine learning model optimization, regularization, and ensemble methods for environmental health risk prediction.*
