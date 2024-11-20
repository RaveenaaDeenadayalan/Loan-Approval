import streamlit as st
import pandas as pd
import joblib
import numpy as np

def load_models():
    """
    Load saved models, encoders, and scalers
    """
    try:
        label_encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')
        selector = joblib.load('selector.pkl')
        best_model = joblib.load('best_model.pkl')
        loan_status_encoder = joblib.load('loan_status_encoder.pkl')
        
        return label_encoders, scaler, selector, best_model, loan_status_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

def preprocess_input(input_data, label_encoders, scaler):
    """
    Preprocess input data for prediction
    """
    # Create a copy of input data to avoid modifying original
    new_data = input_data.copy()
    
    # Encode categorical variables
    cat_columns = ['education', 'self_employed']
    for column in cat_columns:
        new_data[column] = label_encoders[column].transform(new_data[column])
    
    # Scale numerical features
    num_columns = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
        'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
        'luxury_assets_value', 'bank_asset_value'
    ]
    new_data_scaled = scaler.transform(new_data[num_columns])
    scaled_df = pd.DataFrame(new_data_scaled, columns=num_columns)
    
    # Combine categorical and scaled numerical features
    processed_data = pd.concat([new_data[cat_columns], scaled_df], axis=1)
    
    return processed_data

def predict_loan_status(processed_data, selector, best_model, loan_status_encoder):
    """
    Make loan status prediction
    """
    # Select features
    data_selected = selector.transform(processed_data)
    
    # Predict
    prediction = best_model.predict(data_selected)
    
    # Convert prediction back to original labels
    status = loan_status_encoder.inverse_transform(prediction)[0]
    
    # Get prediction probability
    prediction_proba = best_model.predict_proba(data_selected)[0]
    approval_prob = prediction_proba[1] * 100 if status == 'Approved' else prediction_proba[0] * 100
    
    return status, approval_prob

def main():
    st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")
    
    # Load models
    label_encoders, scaler, selector, best_model, loan_status_encoder = load_models()
    
    if not all([label_encoders, scaler, selector, best_model, loan_status_encoder]):
        st.error("Failed to load models. Please check your model files.")
        return
    
    # Title and description
    st.title("🏦 Loan Approval Prediction")
    st.write("Enter loan application details to predict approval status")
    
    # Input form
    with st.form("loan_application"):
        col1, col2 = st.columns(2)
        
        with col1:
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            income_annum = st.number_input("Annual Income", min_value=0, value=0)
        
        with col2:
            loan_amount = st.number_input("Loan Amount", min_value=0, value=0)
            loan_term = st.number_input("Loan Term (Months)", min_value=1, max_value=360, value=12)
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
        
        residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=0)
        commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=0)
        luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=0)
        bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=0)
        
        submitted = st.form_submit_button("Predict Loan Status")
        
        if submitted:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'no_of_dependents': [no_of_dependents],
                'education': [education],
                'self_employed': [self_employed],
                'income_annum': [income_annum],
                'loan_amount': [loan_amount],
                'loan_term': [loan_term],
                'cibil_score': [cibil_score],
                'residential_assets_value': [residential_assets_value],
                'commercial_assets_value': [commercial_assets_value],
                'luxury_assets_value': [luxury_assets_value],
                'bank_asset_value': [bank_asset_value]
            })
            
            # Preprocess input
            processed_data = preprocess_input(input_data, label_encoders, scaler)
            
            # Predict
            status, approval_prob = predict_loan_status(
                processed_data, selector, best_model, loan_status_encoder
            )
            
            # Display results
            if status == 'Approved':
                st.success(f"🎉 Loan Approved! Probability of Approval: {approval_prob:.2f}%")
            else:
                st.error(f"❌ Loan Rejected. Probability of Approval: {approval_prob:.2f}%")
            
            # # Additional insights
            # st.subheader("Prediction Insights")
            # st.write("Factors considered:")
            # st.write(f"- Education: {education}")
            # st.write(f"- Self Employment: {self_employed}")
            # st.write(f"- Annual Income: ₹{income_annum:,}")
            # st.write(f"- CIBIL Score: {cibil_score}")

if __name__ == "__main__":
    main()