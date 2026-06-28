markdown_content = """# 🏠 House Price Predictor

A Machine Learning project that predicts house prices using Linear Regression while demonstrating the complete supervised learning workflow, from data preprocessing to model evaluation.

> **Note:** This project was built to strengthen my understanding of Machine Learning fundamentals before moving into more advanced AI Engineering topics such as Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), and AI-powered applications.

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
