from src.preprocessing import load_data, preprocess
from src.train import train_model, test_model
from src.baseline import baseline_model
from src.evaluate import (
    calculate_metrics,
    residual_analysis,
    cross_validation
)
# Load data
df = load_data(
    "data/housing_dataset_200_ml_fundamentals.json")
print("\n - PREPROCESSING -")
df = preprocess(df)
# train the Linear Regression Model
(
    model,
    X,
    y,
    X_train,
    X_test,
    y_train,
    y_test
) = train_model(df)
# Train the Base Line model
print("\n - BASELINE MODEL - ")

baseline_prediction = baseline_model(y_train)

baseline_predictions = [
    baseline_prediction
] * len(y_test)

calculate_metrics(
    y_test,
    baseline_predictions
)
# Evaluate Linear Regression
print("\n - LINEAR REGRESSION - ")
y_pred = test_model(
    model,
    X_test
)

calculate_metrics(
    y_test,
    y_pred
)

residual_analysis(
    X_test,
    y_test,
    y_pred
)
# Cross Validation
print("\n - CROSS VALIDATION - ")
cross_validation(
    model,
    X,
    y
)