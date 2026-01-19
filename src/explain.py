import shap
import joblib
import pandas as pd

model = joblib.load("models/churn_model.pkl")

def explain_prediction(input_df):
    explainer = shap.Explainer(model.named_steps['model'])
    shap_values = explainer(
        model.named_steps['preprocessor'].transform(input_df)
    )
    return shap_values.values
