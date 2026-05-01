import pandas as pd
import joblib
import os


# ============================================================
# 1. LOAD TRAINED MODEL SAFELY
# ============================================================

MODEL_FILE = "models/loan_xgb_monotonic.joblib"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"❌ Model file '{MODEL_FILE}' not found.\n"
        f"Please run the training script first to create it."
    )

model = joblib.load(MODEL_FILE)
print("Model loaded ✔")


# ============================================================
# 2. LOAD NEW APPLICATION DATA
# ============================================================

# Change filename here:

DATA_FILE = "data/raw/new_applications.csv"
# DATA_FILE = "extreme_applications.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"❌ Input file '{DATA_FILE}' not found.\n"
        f"Please provide a valid CSV file."
    )

new = pd.read_csv(DATA_FILE)
print(f"Loaded dataset: {DATA_FILE} ✔")


# ============================================================
# 3. SAFE NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "LoanAmount",
    "ApplicantIncome",
    "CoapplicantIncome",
    "Loan_Amount_Term",
    "Credit_History",
    "Total_Income",
]

for col in numeric_columns:
    if col in new.columns:
        new[col] = pd.to_numeric(new[col], errors="coerce")


# ============================================================
# 4. FEATURE ENGINEERING (MUST MATCH TRAINING)
# ============================================================

new["DTI"] = new["LoanAmount"] / (
    new["ApplicantIncome"] + new["CoapplicantIncome"] + 1
)

new["DTI"] = pd.to_numeric(new["DTI"], errors="coerce")


# ============================================================
# 5. PREDICT WITH XGBOOST MONOTONIC MODEL
# ============================================================

preds = model.predict(new)
probs = model.predict_proba(new)[:, 1]

new["Predicted_Status"] = preds
new["Approval_Probability"] = probs


# ============================================================
# 6. SAVE SCORING OUTPUT
# ============================================================

output_file = "data/processed/scored_applications_xgb.csv"
new.to_csv(output_file, index=False)

print("\nScoring complete ✔")
print(f"Results saved to: {output_file}\n")
print(new.head())
