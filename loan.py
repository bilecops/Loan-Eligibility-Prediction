import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load your trained model using pickle
with open("loan_eligibility_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

# Function to encode categorical variables
def encode(data):
    cat_col = ['Term', 'Home Ownership', 'Purpose']
    le = LabelEncoder()
    for col in cat_col:
        if col in data.columns:
            data[col] = le.fit_transform(data[col])
    return data

# Streamlit UI elements
st.title("Loan Eligibility Prediction")
st.write("Upload a CSV file containing loan data for eligibility prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type='csv')

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        
        # Display dataset
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        # Preprocessing the data
        data = encode(data)

        # Check if all required columns are present
        required_columns = [
            'Current Loan Amount', 'Term', 'Credit Score', 'Annual Income',
            'Years in current job', 'Home Ownership', 'Purpose', 'Monthly Debt',
            'Years of Credit History', 'Number of Open Accounts',
            'Number of Credit Problems', 'Current Credit Balance',
            'Maximum Open Credit', 'Bankruptcies', 'Tax Liens'
        ]
        if not all(col in data.columns for col in required_columns):
            missing_cols = set(required_columns) - set(data.columns)
            st.error(f"Missing required columns: {missing_cols}")
        else:
            # Separate features
            X = data[required_columns]

            # Make predictions using the trained model
            predictions = model.predict(X)

            # Map predictions to human-readable labels
            prediction_labels = ["Not Eligible" if pred == 0 else "Eligible" for pred in predictions]

            # Display the predictions
            st.write("Predictions:")
            st.write(prediction_labels)

            # Add predictions to the dataset
            result_df = data.copy()
            result_df['Predicted Loan Status'] = prediction_labels
            st.write("Predictions added to the dataset:")
            st.dataframe(result_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.write("Made by Orange Cat")