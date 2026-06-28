import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split,cross_val_score)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

# just because correlation is low, you shouldn't remove the column
# only remove the features/cols when it contributes to model performance or
# It has no logical relationship AND/OR
data = pd.read_json("housing_dataset_200_ml_fundamentals.json")
df = pd.DataFrame(data)
print(df)


# Preprocessing the data 
def preprocessing(df):
    # Information : 
    print(df.info())
    # Statiscal Summary :
    print(df.describe())
    
    # Count the missing values 
    print("Missing values in the dataset: ",df.isnull().sum())
    
    # to show the rows which are missing the values in a feature 
    print(df[df["area_sqft"].isnull()])
    print(df['area_sqft'].corr(df['bedrooms']))
    print(df['area_sqft'].corr(df['bathrooms']))
    print(df['area_sqft'].corr(df['price_lakh']))
    # Keep in mind that all these three features have a positive linear correlation with area_sqft
    missing_price_mean = df[df["area_sqft"].isnull()]['price_lakh'].mean()
    missing_bathrooms_mean = df[df["area_sqft"].isnull()]['bathrooms'].mean()
    missing_bedrooms_mean = df[df["area_sqft"].isnull()]['bedrooms'].mean()
    print(missing_price_mean, missing_bathrooms_mean, missing_bedrooms_mean)
    
    # for the time being just handle the missing values by filling mean, median etc.
    # we would work on those to improve the model prediction later
    df['area_sqft'] = df['area_sqft'].fillna(df['area_sqft'].mean())
    df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].mean())
    df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].mean())
    
    # verify if there's any value missing after filling 
    # print(df.isnull().sum())
    
    # df = df.dropna() # df.dropna() -> creates a new data frame, it doesn't directly modify the data frame
    # print("After dropping the Missing values in the dataset: ",pd.isnull(df).sum()
    
def train_model(df):
    # Features for the prediction 
    X = df[[
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "house_age",
        "garage_spaces"
    ]]
    # Target for the prediction 
    y = df["price_lakh"]
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
    )
        
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    model = LinearRegression()
    model.fit(X_train, y_train) 
    print("Model Trained Sucessfully!")
    
     # Linear Regression Co-efficients 
     # which feature is contributing to the model the most
     # weights learned by the model 
    print("Usefulness of each feature : ")
    for feature, coef in zip(X.columns, model.coef_):
        print(feature, coef) 
    
    return (model,X,y,X_train,X_test,y_train,y_test)
    
def test_model(model, X_test,y_test):
    y_pred = model.predict(X_test)
    
    # Actual values
    print("Actual values : ")
    print(y_test[:5])
    print("Predictied Values: ")
    # the first 5 values
    print(y_pred[:5])


    return y_pred
# base line model 
def baseline_model(y_train, y_test):

    baseline_prediction = y_train.mean()

    baseline_predictions = [
        baseline_prediction
    ] * len(y_test)

    mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    mse = mean_squared_error(
        y_test,
        baseline_predictions
    )

    rmse = root_mean_squared_error(
        y_test,
        baseline_predictions
    )

    r2 = r2_score(
        y_test,
        baseline_predictions
    )

    print("\n===== BASELINE MODEL =====")
    print("Mean House Price:", baseline_prediction)
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R²  :", r2)
      
def calculate_errors(X_test,y_test,y_pred):
    # Calculating Mean Absolute Error
    mae = mean_absolute_error(y_test,y_pred)
    print("MAE : ", mae)
    
    # Calculating Mean Squared Error 
    # helpful when you want to choose between two models producing same MAE 
    mse = mean_squared_error(y_test,y_pred)
    print("MSE : ",mse)
    
    # Calculating Root Mean Squared Error
    rmse = root_mean_squared_error(y_test,y_pred)
    print("RMSE : ", rmse)
    
    # Calculating R^2 score : this is not a error metric 
    r2 = r2_score(y_test,y_pred)
    print("R^2 : ",r2)
    
    # Residual analysis : 
    errors = pd.Series(np.array(y_test) - y_pred,index=y_test.index)
    print("\n")
    print("Residual Analysis: " , errors.describe())
    
    # find the largest error (its index)
    largest_error_idx = abs(errors).idxmax()
    print(largest_error_idx)

    print(X_test.loc[largest_error_idx])
    print(y_test.loc[largest_error_idx])
    
# cross validation 
def cross_validation(X, y):

    scores = cross_val_score(
        LinearRegression(),
        X,
        y,
        cv=5,
        scoring="r2"

    )

    print("\n===== CROSS VALIDATION =====")

    for i in range(len(scores)):
        print("Fold", i+1, ":", scores[i])

    print("\nMean R²:",scores.mean())
    print("Std Dev:",scores.std())
    
preprocessing(df)
    
(
    model,
    X,
    y,
    X_train,
    X_test,
    y_train,
    y_test
) = train_model(df)

baseline_model(y_train,y_test)

cross_validation(
    X,
    y
)

y_pred = test_model(
    model,
    X_test,
    y_test
)

calculate_errors(
    X_test,
    y_test,
    y_pred
)