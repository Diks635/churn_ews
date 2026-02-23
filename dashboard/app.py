import streamlit as st
import pandas as pd
import joblib
import sys
import os
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.feature_engineering import create_features

APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(APP_DIR, "..", "models", "churn_model.pkl")
COLUMNS_PATH = os.path.join(APP_DIR, "..", "models", "model_columns.pkl")
NUM_COLS_PATH = os.path.join(APP_DIR, "..", "models", "numeric_columns.pkl")
CAT_COLS_PATH = os.path.join(APP_DIR, "..", "models", "categorical_columns.pkl")

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)
numeric_cols = joblib.load(NUM_COLS_PATH)
categorical_cols = joblib.load(CAT_COLS_PATH)

st.title("Customer Churn Prediction")
st.write("Enter customer details below:")

tenure = st.number_input("Tenure (months)", min_value=0, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=900.0)

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

customerID = st.number_input("Customer ID", value=0)
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])

input_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "PaperlessBilling": paperless_billing,
    "MultipleLines": multiple_lines,
    "Partner": partner,
    "Dependents": dependents,
    "Contract": contract_type,
    "PaymentMethod": payment_method,
    "customerID": customerID,
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "PhoneService": 1 if phone_service == "Yes" else 0
}])

input_data = create_features(input_data)

for col in numeric_cols:
    if col in input_data.columns:
        input_data[col] = pd.to_numeric(input_data[col], errors="coerce").fillna(0.0)

for col in categorical_cols:
    if col in input_data.columns:
        input_data[col] = input_data[col].astype(str)

for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_columns]

st.write("Input Data Types Before Prediction:")
st.write(input_data.dtypes)
st.write("Input Data Preview:")
st.write(input_data)

if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.success(f"Prediction: {'Yes, will churn' if prediction == 1 else 'No, will not churn'}")
        st.info(f"Churn Probability: {round(probability * 100, 2)}%")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

st.header("Churn Analysis Charts")
st.write("Analyze churn patterns in your dataset (example)")

DATA_PATH = os.path.join(BASE_DIR, "data", "Churn.csv")
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df = create_features(df)

    fig1 = px.histogram(df, x="Churn", title="Churn Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="Contract", color="Churn", barmode="group",
                        title="Churn by Contract Type")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                  title="Monthly Charges vs Churn")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(df, x="Churn", y="tenure", color="Churn",
                  title="Tenure vs Churn")
    st.plotly_chart(fig4, use_container_width=True)

else:
    st.warning("Customer dataset for analysis not found. Place customer_data.csv in the data folder.")
 