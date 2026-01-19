import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from feature_engineering import create_features  # make sure this is correct path

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("data/churn.csv")

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
df = create_features(df)

# --------------------------------------------------
# Split features and target
# --------------------------------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

# --------------------------------------------------
# Identify numeric and categorical columns
# --------------------------------------------------
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# --------------------------------------------------
# Preprocessor
# --------------------------------------------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# --------------------------------------------------
# Model
# --------------------------------------------------
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    eval_metric='logloss'
)

# --------------------------------------------------
# Pipeline
# --------------------------------------------------
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# --------------------------------------------------
# Train/test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Fit pipeline
# --------------------------------------------------
pipeline.fit(X_train, y_train)

# --------------------------------------------------
# Save artifacts
# --------------------------------------------------
joblib.dump(pipeline, "models/churn_model.pkl")            # full pipeline
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")  # original columns
joblib.dump(num_cols, "models/numeric_columns.pkl")         # numeric columns
joblib.dump(cat_cols, "models/categorical_columns.pkl")     # categorical columns

print("✅ Advanced churn model trained and artifacts saved successfully")
