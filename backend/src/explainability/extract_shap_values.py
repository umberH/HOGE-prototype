import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.compose import ColumnTransformer

# ============================================================
# 1. Load model & raw dataset
# ============================================================

MODEL_FILE = "models/loan_xgb_monotonic.joblib"
DATA_FILE = "data/raw/df1_loan.csv"

print("Loading model...")
pipeline = joblib.load(MODEL_FILE)
preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]
print("Model loaded ✔")

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# Keep an ID to link back to applications / KG
if "Loan_ID" in df.columns:
    app_ids = df["Loan_ID"].copy()
else:
    app_ids = pd.Series(range(len(df)), name="row_id")

# ============================================================
# 2. Clean data EXACTLY as in training
# ============================================================

# Drop index / ID columns from feature set
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Clean Total_Income: "$5,849.00" -> 5849.0
df["Total_Income"] = (df["Total_Income"].astype(str)
                      .str.replace("$", "", regex=False)
                      .str.replace(",", "", regex=False))
df["Total_Income"] = pd.to_numeric(df["Total_Income"], errors="coerce")

# Map Loan_Status to numeric
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Drop Loan_ID from features (we kept it separately as app_ids)
df = df.drop(columns=["Loan_ID"], errors="ignore")

# Numeric conversion for the same columns as training
numeric_cols = [
    "LoanAmount",
    "ApplicantIncome",
    "CoapplicantIncome",
    "Loan_Amount_Term",
    "Credit_History",
    "Total_Income",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Recreate DTI (must match training logic)
df["DTI"] = df["LoanAmount"] / (df["ApplicantIncome"] + df["CoapplicantIncome"] + 1)

# Split features / target
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

print(f"X shape (raw features): {X.shape}")

# ============================================================
# 3. Transform features with the pipeline preprocessor
# ============================================================

print("Transforming features through preprocessor...")
X_transformed = preprocessor.transform(X)
print(f"X_transformed shape: {X_transformed.shape}")


# ============================================================
# 4. Get transformed feature names
# ============================================================

def get_feature_names(column_transformer: ColumnTransformer):
    output_features = []
    for name, transformer, cols in column_transformer.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps"):
            # e.g. Pipeline(num -> imputer+scaler, cat -> imputer+onehot)
            if "onehot" in transformer.named_steps:
                ohe = transformer.named_steps["onehot"]
                output_features.extend(ohe.get_feature_names_out(cols))
            else:
                output_features.extend(cols)
        else:
            output_features.extend(cols)
    return np.array(output_features)

feature_names = get_feature_names(preprocessor)
print("\nTransformed feature names:")
print(feature_names)


# ============================================================
# 5. Compute SHAP values
#    (This assumes your env has xgboost/shap combo working, since
#     you already managed to generate the plots.)
# ============================================================

print("\nBuilding TreeExplainer and computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# For binary classification, shap_values is (n_samples, n_features)
print(f"shap_values shape: {np.array(shap_values).shape}")


# ============================================================
# 6. Build a WIDE SHAP dataset (one row per application)
# ============================================================

print("\nBuilding wide SHAP dataframe...")
shap_wide = pd.DataFrame(shap_values, columns=feature_names)

# Add metadata: application id, prediction, probability
preds = pipeline.predict(X)
probs = pipeline.predict_proba(X)[:, 1]

shap_wide.insert(0, "application_id", app_ids.values)
shap_wide.insert(1, "prediction", preds)
shap_wide.insert(2, "approval_probability", probs)

shap_wide.to_csv("data/processed/shap_wide.csv", index=False)
print("Saved shap_wide.csv ✔")


# ============================================================
# 7. Build a LONG SHAP dataset (one row per app-feature)
#    This is the best shape for Neo4j / KG ingestion.
# ============================================================

print("Building long SHAP dataframe for KG...")

records = []
shap_values = np.array(shap_values)
abs_shap = np.abs(shap_values)

for i in range(shap_values.shape[0]):
    app_id = app_ids.iloc[i]
    pred = int(preds[i])
    prob = float(probs[i])

    # ranking features by absolute impact for this application
    order = np.argsort(-abs_shap[i])
    rank_map = {feature_names[j]: int(k + 1) for k, j in enumerate(order)}

    for j, feat in enumerate(feature_names):
        val = float(shap_values[i, j])
        records.append(
            {
                "application_id": app_id,
                "feature": feat,
                "shap_value": val,
                "abs_shap": float(abs_shap[i, j]),
                "direction": "supports_approval" if val > 0 else "supports_decline",
                "rank": rank_map[feat],
                "prediction": pred,
                "approval_probability": prob,
            }
        )

shap_long = pd.DataFrame(records)
shap_long.to_csv("data/processed/shap_long.csv", index=False)
print("Saved shap_long.csv ✔")

print("\nDone. You now have:")
print("  - shap_wide.csv  (per-application feature attributions)")
print("  - shap_long.csv  (per-application, per-feature – ready for Neo4j)")
