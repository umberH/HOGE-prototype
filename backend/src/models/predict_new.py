import pandas as pd
import joblib
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import centralized path configuration
from backend.src.config import get_paths

# Get centralized paths
paths = get_paths()


# ============================================================
# 1. LOAD TRAINED MODEL SAFELY
# ============================================================

MODEL_FILE = paths.MODEL_XGB_MONOTONIC

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model file '{MODEL_FILE}' not found.\n"
        f"Please run the training script first to create it."
    )

model = joblib.load(MODEL_FILE)
print(f"Model loaded from {MODEL_FILE} [OK]")


# ============================================================
# 2. LOAD NEW APPLICATION DATA
# ============================================================

# Change filename here if you want to use a different file:

DATA_FILE = paths.RAW_DATA_NEW_APPLICATIONS
# DATA_FILE = paths.RAW_DATA_DIR / "extreme_applications.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Input file '{DATA_FILE}' not found.\n"
        f"Please provide a valid CSV file."
    )

new = pd.read_csv(DATA_FILE)
print(f"Loaded dataset: {DATA_FILE} [OK]")


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

output_file = paths.SCORED_APPLICATIONS
new.to_csv(output_file, index=False)

print("\nScoring complete ✔")
print(f"Results saved to: {output_file}\n")
print(new.head())
