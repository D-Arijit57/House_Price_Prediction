# 🏠 House Price Predictor

A comprehensive Machine Learning project that predicts house prices using Linear Regression, demonstrating the complete supervised learning workflow from data preprocessing to model evaluation.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Results](#-results)
- [Technologies Used](#-technologies-used)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [Getting Started](#-getting-started)

---

## 📖 Project Overview

This project aims to predict house prices based on several numerical features while emphasizing the complete Machine Learning pipeline, including:

- **Data preprocessing & missing value handling**
- **Model training & evaluation**
- **Baseline comparison**
- **Cross-validation**

### Target Features

The model predicts house prices based on:

- **Area** (Square Feet)
- **Number of Bedrooms**
- **Number of Bathrooms**
- **House Age** (years)
- **Garage Spaces**

---

## 📊 Dataset

**Size:** 200 housing records

**Note:** Some records intentionally contain missing values to demonstrate preprocessing techniques.

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `house_id` | Integer | Unique identifier |
| `area_sqft` | Float | Area of the house in square feet |
| `bedrooms` | Float | Number of bedrooms |
| `bathrooms` | Float | Number of bathrooms |
| `house_age` | Float | Age of the house in years |
| `garage_spaces` | Float | Number of garage spaces |
| `price_lakh` | Float | **Target variable** - House Price in Lakhs |

---

## 🗂️ Project Structure

```
House_Price_Prediction/
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
```

---

## ⚙️ Machine Learning Pipeline

### 1. Data Loading
- Load JSON dataset using Pandas
- Inspect dataset structure
- Verify data types and dimensions

### 2. Exploratory Data Analysis (EDA)
- Statistical summary (mean, median, standard deviation)
- Missing value analysis
- Correlation analysis
- Feature relationship visualization

### 3. Data Preprocessing
- **Missing Value Handling:** Mean imputation for numerical features
- **Features Used:** Area, Bedrooms, Bathrooms, House Age, Garage Spaces
- **Target Variable:** House Price (in Lakhs)

### 4. Train/Test Split
- **Training Data:** 80%
- **Testing Data:** 20%
- Enables model learning from historical data and evaluation on unseen samples

### 5. Model Training
- **Algorithm:** Linear Regression
- Learns the relationship between input features and target house price

### 6. Baseline Model
- Simple baseline predicting the mean house price for every sample
- Serves as a benchmark to validate Linear Regression performance

### 7. Model Evaluation
Metrics used:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Coefficient of Determination (R²)**

### 8. Residual Analysis
- Understand prediction errors
- Identify largest prediction deviations
- Analyze difficult-to-predict samples

### 9. Cross Validation
- **5-Fold Cross Validation** for robust generalization assessment
- More reliable than single train-test split

---

## 📈 Results

### Linear Regression Model Performance

| Metric | Value |
|--------|-------|
| **MAE** | ≈ 5.28 Lakhs |
| **RMSE** | ≈ 6.73 Lakhs |
| **R²** | ≈ 0.956 |

### Model vs Baseline

The Linear Regression model **significantly outperforms** the baseline model across all evaluation metrics, demonstrating the effectiveness of the approach.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning library

---

## 🧠 Key Learnings

Through this project, I gained hands-on experience with:

- ✅ Data preprocessing & missing value handling strategies
- ✅ Feature selection and engineering
- ✅ Train/Test split methodology
- ✅ Cross-validation techniques
- ✅ Linear Regression model implementation
- ✅ Baseline model comparison
- ✅ Regression evaluation metrics
- ✅ Residual analysis and interpretation
- ✅ Modular Machine Learning project structure

---

## 🚀 Future Improvements

Potential enhancements to expand this project:

- 🔄 **Predictive Imputation** for more sophisticated missing value handling
- 📊 **Feature Scaling** (Standardization/Normalization)
- 📈 **Advanced Models:** Ridge and Lasso Regression
- 📉 **Residual Visualization** plots and diagnostics
- 🎯 **Hyperparameter Tuning** for model optimization
- 💾 **Larger Dataset:** Real-world housing data with more features
- 🤖 **Ensemble Methods:** Random Forest, Gradient Boosting
- 📱 **Model Deployment:** Create API or web interface

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Project

```bash
python main.py
```

This will execute the complete pipeline from data loading to model evaluation.

---

**Author:** Arijit Das  
**Last Updated:** 2026  
**License:** MIT (if applicable)
