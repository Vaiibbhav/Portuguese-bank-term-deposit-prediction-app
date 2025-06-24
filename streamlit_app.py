import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# Load trained model and explainer
model = joblib.load("xgboost_model.pkl")

# Initialize SHAP
explainer = shap.Explainer(model)

# UI Title
st.title("Term Deposit Subscription Prediction App")
st.markdown("Enter customer details below to predict if they will subscribe to a term deposit.")

# Collect input from user
age = st.slider("Age", 18, 95, 30)
job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                           'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                        'illiterate', 'professional.course', 'university.degree'])
contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
campaign = st.number_input("Number of Contacts During Campaign", min_value=1, max_value=50, value=1)
poutcome = st.selectbox("Previous Campaign Outcome", ['failure', 'nonexistent', 'success'])
previous = st.number_input("Number of Contacts Before This Campaign", min_value=0, max_value=100, value=0)

# Encode categorical features (manually mapping to model encoding)
job_map = {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4,
           'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9, 'unemployed': 10}
marital_map = {'married': 0, 'single': 1, 'divorced': 2}
education_map = {'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3,
                 'illiterate': 4, 'professional.course': 5, 'university.degree': 6}
contact_map = {'cellular': 0, 'telephone': 1}
month_map = {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
             'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}
poutcome_map = {'failure': 0, 'nonexistent': 1, 'success': 2}

# Create dataframe for prediction
input_df = pd.DataFrame({
    'age': [age],
    'job': [job_map[job]],
    'marital': [marital_map[marital]],
    'education': [education_map[education]],
    'contact': [contact_map[contact]],
    'month': [month_map[month]],
    'campaign': [campaign],
    'poutcome': [poutcome_map[poutcome]],
    'previous': [previous]
})

# Predict and show result
if st.button("Predict"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.success(f"✅ The customer is likely to SUBSCRIBE. (Confidence: {round(proba * 100, 2)}%)")
    else:
        st.warning(f"❌ The customer is NOT likely to subscribe. (Confidence: {round((1 - proba) * 100, 2)}%)")

    # SHAP Explanation
    st.subheader("SHAP Explanation")
    shap_values = explainer(input_df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.waterfall(shap_values[0], max_display=9)
    st.pyplot()
