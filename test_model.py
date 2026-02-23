import joblib
import pandas as pd


from src.feature_engineering import create_features

model = joblib.load("models/churn_model.pkl")

df = pd.read_csv("data/churn.csv")

sample = df.iloc[[0]].copy()

if "Churn" in sample.columns:
    sample = sample.drop("Churn", axis=1)

sample = create_features(sample)

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("Prediction:", "Churn" if prediction == 1 else "No Churn")
print("Churn Probability:", round(float(probability), 2))


