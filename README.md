# Loan Approval Prediction 🏦💰

## Overview
This is one of the projects for my graduate class. Loan approval is a crucial process in the banking and finance industry. This application provides a simple interface for predicting loan approval based on input parameters such as income, loan amount, CIBIL score, and more. The backend is powered by machine learning models trained on historical data.

## Features
- **Batch Prediction**: Predicts loan statuses for multiple applicants from a dataset.
- **Custom Prediction**: Allows users to input loan details through an interactive interface to get predictions instantly.
- **Data Preprocessing**: Automates label encoding, scaling, and feature selection to ensure data compatibility with models.
- **Machine Learning Models**:
  - Decision Tree Classifier (primary model).
  - Logistic Regression and Support Vector Machine for additional comparisons.
- **Interactive Visualization**: Displays batch predictions using Streamlit.

- ## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - **Data Manipulation**: Pandas, NumPy
  - **Machine Learning**: Scikit-learn
  - **Web Framework**: Streamlit
- **Feature Engineering**:
  - StandardScaler for feature scaling
  - LabelEncoder for categorical encoding
  - SelectKBest for feature selection
- **Model**: DecisionTreeClassifier

- ## Dataset
- **Source**: Kaggle
- **Attributes**:
  - `no_of_dependents`: Number of dependents for the borrower.
  - `income_annum`: Annual income of the borrower.
  - `loan_amount`: Loan amount requested.
  - `loan_term`: Loan term in months.
  - `cibil_score`: Borrower’s credit score.
  - `Asset values`: residential, commercial, luxury, bank assets
  - `education`: Education level of the borrower.
  - `self_employed`: Employment status.

- ## Usage
- Link - https://mis546-loan-approval-prediction.streamlit.app/
- Input loan and borrower details via the interactive form.
