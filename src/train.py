from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Train 
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

# Predict
def test_model(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred  
