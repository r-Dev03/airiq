# Three Suitable AI Algorithms - Section 1B
For this project, three different algorithms were considered: linear regression, random forest regression, and a simple feed‑forward neural network. These are all supervised learning methods that can predict continuous values like air quality indices or health risk scores from pollution and weather data (Geron, 2019).

## Linear Regression
Linear regression is the most straightforward option. It assumes there is a mostly linear relationship between inputs such as PM2.5, NO2, CO2, temperature, humidity, and wind speed, and the target value we want to predict (Geron, 2019). Because of that assumption, it is fast to train, easy to understand, and its coefficients can be interpreted directly, which is helpful when explaining which variables have the strongest influence (Geron, 2019).

## Random Forest Regression
Random forest regression is a more flexible approach. It builds many decision trees on different random subsets of the data and then averages their predictions, which is a standard ensemble strategy for random forests (Pedregosa et al., 2011; scikit‑learn, n.d.-a). This lets the model capture nonlinear relationships and interactions—for example, when certain combinations of temperature and pollution levels drive risk more than any single variable by itself—and helps reduce overfitting compared with a single tree (Pedregosa et al., 2011). It also tends to be more robust than an individual decision tree and usually handles noisy sensor readings reasonably well, which is why random forests are widely used on tabular environmental data (scikit‑learn, n.d.-b).

## Feed-Forward Neural Network
The feed‑forward neural network is the most flexible of the three. It uses multiple layers of connected “neurons” to learn complex nonlinear patterns in the data (Goodfellow et al., 2016). Given historical pollution and weather conditions, it can learn how certain combinations and ranges of these inputs relate to higher or lower health risk scores, taking advantage of its capacity as a universal function approximator (Goodfellow et al., 2016). It can also be extended later to incorporate more features or time‑windowed inputs for forecasting tasks, which is a common way deep feed‑forward networks are applied to time‑related prediction problems (Goodfellow et al., 2016).

*Random Forest Regression* was chosen as the main algorithm. It strikes a good balance between accuracy, robustness, and interpretability for this air quality and health risk prediction problem.

## Relation to the problem
The main goal is to forecast air quality levels and estimate health risks from pollution and weather data across different U.S. cities. Random forest regression directly supports this by learning a mapping from features such as PM2.5, NO2, CO2, temperature, humidity, and wind speed to continuous targets like air quality indices or health risk scores. Because it can model nonlinear relationships and interactions among these features, it is better suited than a simple linear model to capturing how environmental variables jointly influence health outcomes.

The assignment also emphasizes using regression metrics such as RMSE and MAPE to evaluate how well the models perform over time and across locations. Random forest models fit naturally into this evaluation setup, so it is straightforward to compare different model settings and pick the version that minimizes prediction error.

## Strengths
Random forest regression has several strengths that make it a practical choice here.

  - It can capture complex, nonlinear relationships between variables without requiring a lot of manual feature engineering. This matters because health risk may only spike under specific combinations of pollution and weather conditions.

  - It is fairly robust to noisy and imperfect data, which is common with real‑world sensor readings. Averaging predictions across many trees reduces the impact of outliers and occasional bad measurements.

Another useful benefit is that random forests can produce feature importance scores. Those scores help highlight which pollutants or weather variables are most strongly associated with higher risk, which is valuable information for public health planning and communication.

## Limitations
At the same time, random forest regression is not perfect, and there are a few limitations to keep in mind.

  - Training a large random forest can be computationally expensive, especially as the dataset grows to include more cities, longer time spans, or additional features. If the model needs to be updated frequently with new data, training time and memory use can become an issue.

  - Random forests are less transparent than a simple linear model. Even though tools like feature importance and partial dependence plots help, it is still harder to explain an individual prediction from an ensemble of trees to non‑technical stakeholders.

When the patterns in the data change a lot over time—like after new air‑quality rules, or big shifts in traffic and industry—a random forest model can start to lose accuracy. To keep it trustworthy, the model needs to be checked regularly and retrained on newer data, which lines up with the idea of continuously updating and re-calibrating the system as conditions evolve.

# Evaluation Metrics - Section 1D
To see how well the random forest model is performing, two evaluation metrics were used: root mean squared error (RMSE) and mean absolute percentage error (MAPE). RMSE tells us, on average, how far the predictions are from the true healthRiskScore values, using the same units as the original score, while MAPE summarizes the typical error as a percentage of the true value so it is easier to interpret across different ranges.


## Analysis of results
When evaluated on the test set, the random forest model achieved an RMSE of about 0.1551 and a MAPE of roughly 1.14%. This means the predicted health risk scores are, on average, very close to the actual scores, with typical errors of only about 1.14% relative to the true values. These results suggest that the model is capturing the main relationships between pollution, weather conditions, and health risk, although some error remains, particularly on days when the risk is unusually high or low.

## Area for improvement
One thing that could make this model better is doing a more careful round of tuning instead of mostly sticking with the default settings. Trying different numbers of trees, changing how deep the trees can grow, and adjusting how many samples are needed in each leaf could help lower both RMSE and MAPE. Another improvement would be to include some simple time‑based features, like averages of key pollutants over the last few days, so the model can better reflect the effect of ongoing exposure rather than looking at only a single day at a time.

# Evaluation Metrics - Section 2C
To evaluate the optimized models, the same two metrics from Task 1 were used: root mean squared error (RMSE) and mean absolute percentage error (MAPE). RMSE shows, on average, how far the predictions are from the true healthRiskScore values, while MAPE expresses that error as a percentage of the actual score, which makes it easier to compare performance across different risk levels.

On the test set, the baseline random forest had an RMSE of 0.1551 and a MAPE of 1.14%. The tuned random forest, even after hyperparameter search and added regularization, ended up with a higher RMSE of 0.1896 and a MAPE of 1.42%, so it did not improve on the original model for this dataset, which can happen when tuning increases variance more than it reduces bias (Geron, 2019). The gradient boosting model performed slightly worse than the baseline on RMSE (0.1577) but slightly better on MAPE (1.19%), and the averaged ensemble of the tuned random forest and gradient boosting landed in between, with an RMSE of 0.1614 and a MAPE of 1.22%. Taken together, these results show that the original random forest still gives the best overall accuracy here, but experimenting with boosting and simple ensembling provides alternative models that come close and could be useful if the data or business priorities change (for example, if percentage error becomes more important than absolute error) (Pedregosa et al., 2011).
