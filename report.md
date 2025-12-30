# Three Suitable AI Algorithms
For this project, three different algorithms were considered: linear regression, random forest regression, and a simple feed‑forward neural network. These are all supervised learning methods that can predict continuous values like air quality indices or health risk scores from pollution and weather data.

## Linear Regression
Linear regression is the most straightforward option. It assumes there is a mostly linear relationship between inputs such as PM2.5, NO2, CO2, temperature, humidity, and wind speed, and the target value we want to predict. Because of that assumption, it is fast to train, easy to understand, and its coefficients can be interpreted directly, which is helpful when explaining which variables have the strongest influence.

## Random Forest Regression
Random forest regression is a more flexible approach. It builds many decision trees on different random subsets of the data and then averages their predictions. This lets the model capture nonlinear relationships and interactions, for example when certain combinations of temperature and pollution levels drive risk more than any single variable by itself. It also tends to be more robust than a single decision tree and usually handles noisy sensor readings reasonably well.

## Feed-Forward Neural Network
The feed‑forward neural network is the most flexible of the three. It uses multiple layers of connected “neurons” to learn complex patterns in the data. Given historical pollution and weather conditions, it can learn how certain combinations and ranges of these inputs relate to higher or lower health risk scores. It can also be extended later to incorporate more features or time‑windowed inputs for forecasting tasks.

*Random Forest Regression* was chosen as the main algorithm. It strikes a good balance between accuracy, robustness, and interpretability for this air quality and health risk prediction problem.

# Relation to the problem
The main goal is to forecast air quality levels and estimate health risks from pollution and weather data across different U.S. cities. Random forest regression directly supports this by learning a mapping from features such as PM2.5, NO2, CO2, temperature, humidity, and wind speed to continuous targets like air quality indices or health risk scores. Because it can model nonlinear relationships and interactions among these features, it is better suited than a simple linear model to capturing how environmental variables jointly influence health outcomes.

The assignment also emphasizes using regression metrics such as RMSE and MAPE to evaluate how well the models perform over time and across locations. Random forest models fit naturally into this evaluation setup, so it is straightforward to compare different model settings and pick the version that minimizes prediction error.

# Strengths
Random forest regression has several strengths that make it a practical choice here.

  - It can capture complex, nonlinear relationships between variables without requiring a lot of manual feature engineering. This matters because health risk may only spike under specific combinations of pollution and weather conditions.

  - It is fairly robust to noisy and imperfect data, which is common with real‑world sensor readings. Averaging predictions across many trees reduces the impact of outliers and occasional bad measurements.

Another useful benefit is that random forests can produce feature importance scores. Those scores help highlight which pollutants or weather variables are most strongly associated with higher risk, which is valuable information for public health planning and communication.

# Limitations
At the same time, random forest regression is not perfect, and there are a few limitations to keep in mind.

  - Training a large random forest can be computationally expensive, especially as the dataset grows to include more cities, longer time spans, or additional features. If the model needs to be updated frequently with new data, training time and memory use can become an issue.

  - Random forests are less transparent than a simple linear model. Even though tools like feature importance and partial dependence plots help, it is still harder to explain an individual prediction from an ensemble of trees to non‑technical stakeholders.

When the patterns in the data change a lot over time—like after new air‑quality rules, or big shifts in traffic and industry—a random forest model can start to lose accuracy. To keep it trustworthy, the model needs to be checked regularly and retrained on newer data, which lines up with the idea of continuously updating and re-calibrating the system as conditions evolve.

