# Term Deposit Subscription Prediction — Portuguese Bank

This project predicts whether a bank customer will subscribe to a term deposit based on their profile and past marketing interactions. It uses a real-world dataset from a Portuguese bank marketing campaign and applies end-to-end machine learning techniques, including preprocessing, SMOTE, XGBoost, hyperparameter tuning, SHAP explainability, and deployment preparation.

---

## Project Files

| File Name                     | Description                                                 |
|------------------------------|-------------------------------------------------------------|
| Portuguese_Bank_Final.ipynb  | Complete Jupyter notebook with full EDA, modeling, SHAP     |
| xgboost_pipeline.pkl         | Final model pipeline including preprocessing + classifier   |
| bank-additional-full.csv     | Dataset used for analysis and training                      |
| requirements.txt             | List of required Python packages to run this project        |
| streamlit_app.py             | Optional Streamlit app to deploy the model                  |
| README.md                    | Project overview and instructions                           |

---

## Features

- Exploratory Data Analysis (EDA)
- Handling missing values, feature cleaning, encoding, and scaling
- SMOTE for class imbalance
- Modeling with Logistic Regression, Decision Tree, Random Forest, XGBoost
- Hyperparameter tuning using GridSearchCV
- Model explainability using SHAP
- Pipeline creation for deployment
- Streamlit-compatible model

---

## How to Run This Project

1. Clone this repository:

```bash
git clone https://github.com/VaibhavSonawane01/portuguese-bank-term-deposit-prediction-app.git
cd portuguese-bank-term-deposit-prediction-app
