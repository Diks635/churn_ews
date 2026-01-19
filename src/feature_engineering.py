import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for churn prediction.
    Ensures all numeric columns are numeric and derived features are consistent.
    """

    # ---------------- NUMERIC COLUMNS ----------------
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric safely, replace errors with 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ---------------- FEATURE ENGINEERING ----------------
    # Long-term customer (tenure > 12 months)
    if "tenure" in df.columns:
        df["LongTermCustomer"] = (df["tenure"] > 12).astype(int)

    # Average charge per month (handle tenure=0 safely)
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["AvgChargePerMonth"] = df.apply(
            lambda row: row["TotalCharges"] / row["tenure"] if row["tenure"] > 0 else 0.0,
            axis=1
        )

    # High monthly charges compared to mean
    if "MonthlyCharges" in df.columns:
        mean_charge = df["MonthlyCharges"].mean()
        df["HighCharges"] = (df["MonthlyCharges"] > mean_charge).astype(int)

    # ---------------- ENSURE NUMERIC TYPES ----------------
    derived_cols = ["LongTermCustomer", "HighCharges", "AvgChargePerMonth"]
    for col in derived_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df
