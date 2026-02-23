from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath("src"))
from feature_engineering import create_features

app = FastAPI()

model = joblib.load("models/churn_model.pkl")
num_cols = joblib.load("models/numeric_columns.pkl")
cat_cols = joblib.load("models/categorical_columns.pkl")


@app.post("/predict")
def predict_churn(data: dict):
    try:
        df = pd.DataFrame([data])

        
        for col in num_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        
        for col in cat_cols:
            if col not in df.columns:
                df[col] = "No"
            df[col] = df[col].astype(str).fillna("No")

        
        df = create_features(df)

        
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "Churn": "Yes" if prediction == 1 else "No",
            "Churn_Probability": round(float(probability), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
