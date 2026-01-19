import joblib
import pandas as pd

# IMPORT FEATURE ENGINEERING
from src.feature_engineering import create_features

# Load trained pipeline
model = joblib.load("models/churn_model.pkl")

# Load original dataset
df = pd.read_csv("data/churn.csv")

# Take one row
sample = df.iloc[[0]].copy()

# Drop target
if "Churn" in sample.columns:
    sample = sample.drop("Churn", axis=1)

# APPLY FEATURE ENGINEERING (🔥 IMPORTANT 🔥)
sample = create_features(sample)

# Predict
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("Prediction:", "Churn" if prediction == 1 else "No Churn")
print("Churn Probability:", round(float(probability), 2))


