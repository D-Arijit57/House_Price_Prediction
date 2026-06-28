from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

def calculate_metrics(y_test,y_pred):
    """
    Calculate regression evaluation metrics
    and return them as a dictionary.
    """
    metrics = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "RMSE": root_mean_squared_error(y_test, y_pred),
    "R²": r2_score(y_test, y_pred)
    }

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics
   
    
def residual_analysis(X_test,y_test,y_pred):
    """
    Analyze residuals and identify
    the sample with the largest prediction error.
    """
     # Residual analysis : 
    residuals = y_test - y_pred
    print("\n")
    print("Residual Analysis: " , residuals.describe())
    
    # find the largest error (its index)
    largest_error_idx = abs(residuals).idxmax()
    print("\nLargest Prediction Error",largest_error_idx)

    print("\nHouse Features:")
    print(X_test.loc[largest_error_idx])
    print("\nActual Price:")
    print(y_test.loc[largest_error_idx])
    print("\nPredicted Price:")
    print(y_pred[X_test.index.get_loc(largest_error_idx)])


    
# cross validation 
def cross_validation(model,X, y):
    """
    Evaluate the model using 5-fold cross validation.
    Returns the R² score for each fold.
    """

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2"
    )

    for i in range(len(scores)):
        print(f"Fold {i+1}: {scores[i]:.4f}")

    print(f"\nMean R²:,{scores.mean():.4f}")
    print(f"Std Dev:,{scores.std():.4f}")
    
    return scores
    