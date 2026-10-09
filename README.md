# AirIQ

**Predicting health risk scores from air quality and weather data with ensemble machine learning.**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)

## Results

| Model | RMSE | MAPE |
|-------|------|------|
| Naive baseline (always predicts the training mean) | 0.6742 | 5.70% |
| **Random Forest** | **0.1551** | **1.14%** |
| Random Forest, tuned with GridSearchCV | 0.1896 | 1.42% |
| Gradient Boosting | 0.1577 | 1.19% |
| Averaged ensemble (tuned RF + GB) | 0.1614 | 1.22% |

The Random Forest cuts MAPE by **80% relative to the naive baseline** (RMSE by 77%), with an R² of **0.95** on held-out data. The baseline and R² were computed separately on the same train/test split.

## Highlights

- Models health risk scores from 14 air quality, weather, and calendar features with a Random Forest regressor
- Measures every model against a naive mean baseline, so the error numbers have context
- Benchmarks GridSearchCV tuning (5-fold cross-validation), gradient boosting, and an averaged ensemble to select the final model

## How It Works

**Features (14):** PM2.5, NO2, and CO2; temperature, humidity, wind speed, pressure, cloud cover, visibility, solar radiation, and UV index; month, day of week, and a weekend flag.

**Pipeline:** load the data, split it 80/20 into train and test sets, then train and compare four models:
1. **Baseline Random Forest:** 200 trees with scikit-learn defaults
2. **Tuned Random Forest:** GridSearchCV over `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features`, with 5-fold cross-validation
3. **Gradient Boosting:** `HistGradientBoostingRegressor`
4. **Ensemble:** the average of the tuned Random Forest and Gradient Boosting predictions

### Why the baseline matters
Health risk scores in this dataset only range from about 8.5 to 11.5, so even a model that always predicts the average scores 5.70% MAPE. On its own, a 1.14% MAPE says little. Compared against that baseline, it represents an 80% reduction in error.

### Why tuning made the model worse
The default Random Forest considers all 14 features at every split (`max_features=1.0`). The search grid only tried `"sqrt"` and `"log2"`, which both limit each split to 3 of the 14 features, so the search could never reproduce the baseline's configuration. GridSearchCV chose the least-constrained depth and leaf settings available, and the result still trailed the default. The gap most likely comes from the search space, not overfitting.

## Getting Started

```bash
git clone https://github.com/r-Dev03/airiq.git
cd airiq
pip install pandas "scikit-learn>=1.4" openpyxl
python random_forest.py
```

Or, with Nix: `nix develop`, then `python random_forest.py`.

The script prints the train/test sizes, the best hyperparameters found, and RMSE and MAPE for each model.

## Tech Stack

Python · scikit-learn · pandas · openpyxl

## Known Limitations

- **Single train/test split.** Final scores come from one 80/20 split rather than cross-validated scores.
- **Narrow tuning grid.** The grid didn't include `max_features=1.0`, so the search couldn't reproduce the default model's configuration.
- **Small, partly synthetic dataset.** The 1,000-row dataset contains physically impossible values (such as negative precipitation), and `month` is constant, so results may not carry over to real sensor data.
