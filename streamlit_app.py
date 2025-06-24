import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# Load the trained model and scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize SHAP explainer
explainer = shap.Explainer(model)

# Streamlit UI
st.title("Term Deposit Subscription Prediction")
st.markdown("Fill in the customer details below to predict whether they will subscribe to a term deposit.")

# Input widgets
age = st.slider("Age", 18, 95, 30)
job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                           'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                       'illiterate', 'professional.course', 'university.degree'])
contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
campaign = st.number_input("Number of Contacts During Campaign", min_value=1, max_value=50, value=1)
poutcome = st.selectbox("Previous Campaign Outcome", ['failure', 'nonexistent', 'success'])
previous = st.number_input("Number of Contacts Before This Campaign", min_value=0, max_value=100, value=0)

# Manual label encoding
job_map = {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4,
           'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9, 'unemployed': 10}
marital_map = {'married': 0, 'single': 1, 'divorced': 2}
education_map = {'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3,
                 'illiterate': 4, 'professional.course': 5, 'university.degree': 6}
contact_map = {'cellular': 0, 'telephone': 1}
month_map = {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
             'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}
poutcome_map = {'failure': 0, 'nonexistent': 1, 'success': 2}

# Create DataFrame with all 19 features (even if some are not collected from user)
input_df = pd.DataFrame({
    'age': [age],
    'job': [job_map[job]],
    'marital': [marital_map[marital]],
    'education': [education_map[education]],
    'default': [0],                      # Not collected → assume 'no'
    'housing': [0],                      # Not collected → assume 'no'
    'loan': [0],                         # Not collected → assume 'no'
    'contact': [contact_map[contact]],
    'month': [month_map[month]],
    'day_of_week': [0],                 # Not collected → assume Monday
    'campaign': [campaign],
    'pdays': [999],                     # Not contacted before
    'previous': [previous],
    'poutcome': [poutcome_map[poutcome]],
    'emp.var.rate': [1.1],              # Use common default
    'cons.price.idx': [93.2],
    'cons.conf.idx': [-36.4],
    'euribor3m': [4.857],
    'nr.employed': [5191.0]
})

# Reorder columns to match model training
expected_order = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                  'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous',
                  'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                  'euribor3m', 'nr.employed']
input_df = input_df[expected_order]

# Scale features
input_scaled = scaler.transform(input_df)

# Predict and show results
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.success(f"Prediction: The customer is likely to SUBSCRIBE (Confidence: {round(proba * 100, 2)}%)")
    else:
        st.warning(f"Prediction: The customer is NOT likely to subscribe (Confidence: {round((1 - proba) * 100, 2)}%)")

    # SHAP Explanation
    st.subheader("Model Explanation (SHAP)")
    shap_values = explainer(input_scaled)
   
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.pyplot()
