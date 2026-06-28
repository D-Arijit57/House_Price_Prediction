import pandas as pd

# Load the data 
def load_data(file_path):
    data = pd.read_json(file_path)
    df = pd.DataFrame(data)
    return df

# Preprocessing the data 
def preprocess(df):
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
    
    return df
