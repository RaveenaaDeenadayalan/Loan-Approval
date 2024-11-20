import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

# Load pre-trained models and preprocessors
scaler = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')
dt_model = joblib.load('dt_model.pkl')
encoders = joblib.load('encoders.pkl')

# Function to preprocess the data before prediction
def preprocess_input_data(user_input):
    # Convert the user input into a DataFrame
    input_data = pd.DataFrame([user_input])

    # Ensure column names match the model's expectations
    input_data.columns = [col.strip() for col in input_data.columns]

    # Apply encoding to categorical variables using encoders
    for column in input_data.select_dtypes(include=['object']).columns:
        if column in encoders:
            input_data[column] = encoders[column].fit_transform(input_data[column])

    # Apply scaling to numeric variables
    input_data_scaled = scaler.fit_transform(input_data)

    # Apply feature selection
    input_data_selected = selector.transform(input_data_scaled)

    return input_data_selected

# Streamlit user interface
st.title('Loan Approval Prediction')

# Collecting user input via form
with st.form(key='loan_form'):
    no_of_dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, value=0)
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    income_annum = st.number_input('Annual Income (INR)', min_value=100000, max_value=100000000, value=500000)
    loan_amount = st.number_input('Loan Amount', min_value=1000, max_value=1000000000, value=200000)
    loan_term = st.number_input("Loan Term (years)", min_value=1, value=15, step=1)
    cibil_score = st.number_input('CIBIL Score', min_value=300, max_value=900, value=650)
    residential_assets_value = st.number_input('Residential Assets Value', min_value=0, max_value=500000000, value=1000000)
    commercial_assets_value = st.number_input('Commercial Assets Value', min_value=0, max_value=500000000, value=500000)
    luxury_assets_value = st.number_input('Luxury Assets Value', min_value=0, max_value=500000000, value=200000)
    bank_asset_value = st.number_input('Bank Asset Value', min_value=0, max_value=500000000, value=500000)

    submit_button = st.form_submit_button('Predict Loan Approval')

# Handle prediction after user submits the form
if submit_button:
    # Prepare the user input
    user_input = {
        'no_of_dependents': no_of_dependents,
        'education': education,
        'self_employed': self_employed,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }

    # Preprocess the input data
    processed_input = preprocess_input_data(user_input)

    # Make prediction using the trained model
    prediction = dt_model.predict(processed_input)

    # Display the prediction result
    if prediction == 1:
        st.success('Loan Approved')
    else:
        st.error('Loan Not Approved')
