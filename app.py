import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.feature_selection import SelectKBest, f_classif # type: ignore
import joblib # type: ignore

st.title("Loan Approval Prediction")

# Load the trained Decision Tree model and feature selector
dt_model = joblib.load('dt_model.pkl')
selector = joblib.load('selector.pkl')  # Ensure the selector was saved during training

# Function to preprocess new data and make predictions
def preprocess_new_data(new_data, selector):
    # Label encode categorical columns
    cat_new_data = new_data.select_dtypes(include=['object'])
    le = LabelEncoder()
    for column in cat_new_data.columns:
        new_data[column] = le.fit_transform(new_data[column])

    # Standardize the numerical features
    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)

    # Select top features based on the selector used in training
    new_data_selected = selector.transform(new_data_scaled)  # Use the loaded selector
    
    return new_data_selected

# Inputs for prediction
income = st.number_input("Annual Income (INR)", min_value=100000, value=500000, step=10000)
loan_amount = st.number_input("Loan Amount (INR)", min_value=100000, value=1000000, step=50000)
loan_term = st.number_input("Loan Term (years)", min_value=1, value=15, step=1)
cibil_score = st.number_input("CIBIL Score", min_value=300, value=650, step=10)
residential_assets_value = st.number_input("Residential Assets Value (INR)", min_value=0, value=2000000, step=50000)
commercial_assets_value = st.number_input("Commercial Assets Value (INR)", min_value=0, value=1500000, step=50000)
luxury_assets_value = st.number_input("Luxury Assets Value (INR)", min_value=0, value=500000, step=10000)
bank_asset_value = st.number_input("Bank Assets Value (INR)", min_value=0, value=1000000, step=50000)
num_dependents = st.number_input("Number of Dependents", min_value=0, value=2, step=1)
education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", options=["Yes", "No"])

# Create new data for prediction based on user input
new_data = pd.DataFrame({
    'loan_id': [1],  # Add the loan_id column here as well
    'income_annum': [income],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term],
    'cibil_score': [cibil_score],
    'residential_assets_value': [residential_assets_value],
    'commercial_assets_value': [commercial_assets_value],
    'luxury_assets_value': [luxury_assets_value],
    'bank_asset_value': [bank_asset_value],
    'no_of_dependents': [num_dependents],
    'education': [education],
    'self_employed': [self_employed]
})


# Preprocess the new data (apply encoding, scaling, and feature selection)
new_data_selected = preprocess_new_data(new_data, selector)

# Predict loan status on the button click
if st.button("Predict"):
    # Make prediction
    new_prediction = dt_model.predict(new_data_selected)
    
    if new_prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
