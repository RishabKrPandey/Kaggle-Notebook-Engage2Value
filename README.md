# Engage-2: Value From Clicks to Conversions

This repository contains the solution for the **Engage-2: Value from Clicks to Conversions** Kaggle competition. The project focuses on predicting the `purchaseValue` of website sessions using historical session data, user demographics, and traffic source information.

## 📋 Project Overview
Predicting conversion value is a key challenge in digital marketing. This notebook implements a complete machine learning pipeline—from data cleaning and feature engineering to model ensemble and hyperparameter tuning—to estimate the revenue potential of individual web sessions.

### Key Objectives:
* Perform exploratory data analysis (EDA) on high-dimensional web traffic data.
* Build a robust preprocessing pipeline to handle missing values and categorical encoding.
* Train and optimize Gradient Boosting models for regression.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** * `Pandas` & `NumPy` for data manipulation.
    * `Scikit-Learn` for preprocessing pipelines and cross-validation.
    * `LightGBM` & `XGBoost` for high-performance gradient boosting.
    * `Matplotlib` & `Seaborn` for data visualization.

## 📊 Data Pipeline
The dataset includes 52 features related to user sessions. The notebook follows these steps:

1.  **Data Cleaning:** Dropping columns with excessive missing values (e.g., `trafficSource.adContent`) and imputing missing numerical/categorical data.
2.  **Feature Transformation:** * **Numerical:** Standardized using `StandardScaler`.
    * **Categorical:** Encoded using `OneHotEncoder` and `LabelEncoder`.
3.  **Pipeline Construction:** Utilizing `ColumnTransformer` to ensure data consistency between training and testing sets.

## 🤖 Modeling Approach
The solution explores several regression algorithms to find the best fit for the skewed target variable (`purchaseValue`):

* **XGBoost Regressor**
* **LightGBM Regressor** (Best performing with Tweedie objective)
* **Gradient Boosting Regressor**
* **Hyperparameter Tuning:** Performed via `RandomizedSearchCV` to optimize learning rates, depth, and leaf counts.

## 🚀 Getting Started

### Prerequisites
Ensure you have the following installed:
```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
