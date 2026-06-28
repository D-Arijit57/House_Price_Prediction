markdown_content = """# 🏠 House Price Predictor

A Machine Learning project that predicts house prices using Linear Regression while demonstrating the complete supervised learning workflow, from data preprocessing to model evaluation.

---

## 📖 Project Overview

The objective of this project is to predict the price of a house based on several numerical features, including:

* **Area** (Square Feet)
* **Number of Bedrooms**
* **Number of Bathrooms**
* **House Age**
* **Garage Spaces**

Instead of only training a model, this project focuses on understanding the complete Machine Learning pipeline, including preprocessing, evaluation, baseline comparison, and cross-validation.

## 📊 Dataset

The dataset contains 200 housing records with the following features. 

*Note: Some records intentionally contain missing values to demonstrate preprocessing techniques.*

| Feature | Description |
| :--- | :--- |
| `house_id` | Unique identifier |
| `area_sqft` | Area of the house |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `house_age` | Age of the house (years) |
| `garage_spaces` | Number of garage spaces |
| `price_lakh` | **Target variable** (House Price in Lakhs) |

## 🗂️ Project Structure

```text
HousePricePredictor/
│
├── data/
│   └── housing_dataset_200_ml_fundamentals.json
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── baseline.py
│   └── evaluate.py
│
├── main.py
├── requirements.txt
└── README.md
⚙️ Machine Learning Pipeline
```
1. Data Loading
Load JSON dataset using Pandas.

Inspect dataset structure.

Verify data types.

2. Exploratory Data Analysis (EDA)
Performed basic data exploration including:

Statistical summary

Missing value analysis

Correlation analysis

Feature relationships

3. Data Preprocessing
Missing values are handled using mean imputation for numerical features.

Features used for training: Area, Bedrooms, Bathrooms, House Age, Garage Spaces.

Target: House Price.

4. Train/Test Split
The dataset is divided into:

80% Training Data

20% Testing Data

This allows the model to learn from historical data while being evaluated on unseen samples.

5. Model Training
Model used: Linear Regression.

The model learns the relationship between the input features and the target house price.

6. Baseline Model
A simple baseline model is implemented that predicts the mean house price for every test sample.

This serves as a benchmark to verify that the Linear Regression model performs significantly better than a naive prediction strategy.

7. Model Evaluation
The trained model is evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Coefficient of Determination (R²)

8. Residual Analysis
Residuals are analyzed to:

Understand prediction errors.

Identify the largest prediction error.

Inspect houses that are difficult to predict.

9. Cross Validation
5-Fold Cross Validation is performed to evaluate how well the model generalizes across different train-test splits.

This provides a more reliable estimate of model performance than relying on a single train-test split.

📈 Results
Linear Regression
MAE: ≈ 5.28 Lakhs

RMSE: ≈ 6.73 Lakhs

R²: ≈ 0.956

Baseline Model
The baseline model predicts the average house price for every sample. The Linear Regression model significantly outperforms the baseline across all evaluation metrics.

🛠️ Technologies Used
Python

Pandas

NumPy

Scikit-learn

🧠 Key Learnings
Through this project, I learned:

Data preprocessing & Missing value handling

Feature selection

Train/Test Split & Cross Validation

Linear Regression & Baseline Models

Regression Evaluation Metrics

Residual Analysis

Modular Machine Learning project structure

🚀 Future Improvements
Possible enhancements include:

Predictive Imputation for missing values

Feature Scaling

Ridge and Lasso Regression

Residual Visualization

Hyperparameter Tuning

Larger real-world housing dataset
