import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier

# Load pre-trained model and necessary files
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
selector = joblib.load('selector.pkl')
model = joblib.load('best_model.pkl')  # Assuming DecisionTree was the best model

# Title of the webpage
st.title("Loan Approval Prediction")

# Instructions
st.markdown("""
    This web application allows you to predict whether a loan will be approved based on various input features.
    Please enter the details below to make a prediction.
""")

# User input form for loan prediction
with st.form(key="loan_form"):
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
    education = st.selectbox("Education Level", ["Not Graduate", "Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    income_annum = st.number_input("Annual Income (in INR)", min_value=10000, max_value=1000000000, step=1000)
    loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=1000000000, step=1000)
    loan_term = st.number_input("Loan Term (in months)", min_value=12, max_value=480, step=12)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
    residential_assets_value = st.number_input("Residential Assets Value (in INR)", min_value=1000, max_value=1000000000, step=1000)
    commercial_assets_value = st.number_input("Commercial Assets Value (in INR)", min_value=1000, max_value=1000000000, step=1000)
    luxury_assets_value = st.number_input("Luxury Assets Value (in INR)", min_value=1000, max_value=1000000000, step=1000)
    bank_asset_value = st.number_input("Bank Asset Value (in INR)", min_value=1000, max_value=1000000000, step=1000)

    # Submit button
    submit_button = st.form_submit_button(label="Submit")

# Prediction logic when the form is submitted
if submit_button:
    # Prepare user input data as a dictionary
    user_input = {
        'education': education,
        'self_employed': self_employed,
        'no_of_dependents': no_of_dependents,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }

    # Convert the user input into a DataFrame
    input_df = pd.DataFrame([user_input])

    # Separate categorical and numerical columns
    cat_columns = ['education', 'self_employed']
    num_columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
                   'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
                   'luxury_assets_value', 'bank_asset_value']
    
    # Label encode the categorical columns
    for column in cat_columns:
        if column in label_encoders:
            input_df[column] = label_encoders[column].transform(input_df[column])

    # Standardize numerical features
    input_df_num = input_df[num_columns]
    input_df_scaled = scaler.transform(input_df_num)

    # Replace the scaled values back into the DataFrame
    input_df[num_columns] = input_df_scaled

    # Apply feature selection (SelectKBest)
    input_selected = selector.transform(input_df)

    # Make prediction using the loaded model
    prediction = model.predict(input_selected)

    # Show the result
    if prediction == 1:
        st.success("The loan is approved.")
    else:
        st.error("The loan is not approved.")
