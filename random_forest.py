
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def main():
    data = pd.read_excel("DQN1 Dataset.xlsx")
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

    # Target: health risk score from the dataset
    target_col = "healthRiskScore"

    X = data[feature_cols]
    y = data[target_col]

    # 3. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Create and train the Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(X_train, y_train)

    # 5. Predict on the test set
    y_pred = rf_model.predict(X_test)

    # 6. Simple sanity-check output
    print("Number of training samples:", len(X_train))
    print("Number of test samples:", len(X_test))
    print("First 10 predictions:", y_pred[:10])


if __name__ == "__main__":
    main()

