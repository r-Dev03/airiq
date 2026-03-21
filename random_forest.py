import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def evaluate_model(y_true, y_pred):
    """Return RMSE and MAPE (%) for given true and predicted values."""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    return rmse, mape


def main():
    # 1. Load the dataset
    data = pd.read_excel("dataset.xlsx")

    # 2. Define features and target
    feature_cols = [
        "pm2.5",
        "no2",
        "co2",
        "temp",
        "humidity",
        "windspeed",
        "pressure",
        "cloudcover",
        "visibility",
        "solarradiation",
        "uvindex",
        "month",
        "dayOfWeek",
        "isWeekend",
    ]
    target_col = "healthRiskScore"

    X = data[feature_cols]
    y = data[target_col]

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # BASELINE RANDOM FOREST 
    baseline_rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    baseline_rf.fit(X_train, y_train)

    y_pred_baseline = baseline_rf.predict(X_test)
    baseline_rmse, baseline_mape = evaluate_model(y_test, y_pred_baseline)

    # OPTIMIZED / REGULARIZED RANDOM FOREST 
    #   - Optimization: GridSearchCV over hyperparameters with 5-fold CV
    #   - Regularization: constrain model complexity with max_depth and min_samples_leaf
    rf_for_search = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    grid_search = GridSearchCV(
        estimator=rf_for_search,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    y_pred_tuned = best_rf.predict(X_test)
    tuned_rmse, tuned_mape = evaluate_model(y_test, y_pred_tuned)

    # ENSEMBLE LEARNING
    #   1) Gradient boosting model
    #   2) Simple averaged ensemble (tuned RF + gradient boosting)

    gb_model = HistGradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=10,
        random_state=42,
    )
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    gb_rmse, gb_mape = evaluate_model(y_test, y_pred_gb)

    # Simple averaged ensemble of tuned RF and gradient boosting
    y_pred_ensemble = 0.5 * y_pred_tuned + 0.5 * y_pred_gb
    ensemble_rmse, ensemble_mape = evaluate_model(y_test, y_pred_ensemble)

    print("Number of training samples:", len(X_train))
    print("Number of test samples:", len(X_test))
    print()

    print("=== Baseline Random Forest ===")
    print(f"RMSE: {baseline_rmse:.4f}")
    print(f"MAPE: {baseline_mape:.2f}%")
    print()

    print("=== Tuned Random Forest (Optimization + Regularization) ===")
    print("Best hyperparameters:", grid_search.best_params_)
    print(f"RMSE: {tuned_rmse:.4f}")
    print(f"MAPE: {tuned_mape:.2f}%")
    print()

    print("=== Gradient Boosting Model ===")
    print(f"RMSE: {gb_rmse:.4f}")
    print(f"MAPE: {gb_mape:.2f}%")
    print()

    print("=== Averaged Ensemble (Tuned RF + GB) ===")
    print(f"RMSE: {ensemble_rmse:.4f}")
    print(f"MAPE: {ensemble_mape:.2f}%")

    return (
        baseline_rmse,
        baseline_mape,
        tuned_rmse,
        tuned_mape,
        gb_rmse,
        gb_mape,
        ensemble_rmse,
        ensemble_mape,
    )


if __name__ == "__main__":
    main()

