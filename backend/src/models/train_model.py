import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
import joblib
import datetime
import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.provenance.enhanced_provenance import ProvenanceTracker
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False
    print("Warning: Provenance tracking not available")


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

df = pd.read_csv("data/raw/df1_loan.csv")

# Drop generated index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Clean Total_Income ("$5,849.00" → 5849.0)
df["Total_Income"] = (df["Total_Income"].astype(str)
                      .str.replace("$", "", regex=False)
                      .str.replace(",", "", regex=False))
df["Total_Income"] = pd.to_numeric(df["Total_Income"], errors="coerce")

# Convert Loan_Status Y/N → 1/0
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Remove Loan_ID (not useful)
df = df.drop(columns=["Loan_ID"], errors="ignore")


# ============================================================
# 2. ENSURE NUMERIC COLUMNS
# ============================================================

num_cols = [
    "LoanAmount",
    "ApplicantIncome",
    "CoapplicantIncome",
    "Loan_Amount_Term",
    "Credit_History",
    "Total_Income",
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 3. FEATURE ENGINEERING — DTI
# ============================================================

df["DTI"] = df["LoanAmount"] / (df["ApplicantIncome"] + df["CoapplicantIncome"] + 1)
df["DTI"] = pd.to_numeric(df["DTI"], errors="coerce")


# ============================================================
# 4. DEFINE FEATURES & TARGET
# ============================================================

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns


# ============================================================
# 5. PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# ============================================================
# 6. BUILD MONOTONIC CONSTRAINTS
# ============================================================

mono_constraints = []

for col in numeric_features:
    if col == "ApplicantIncome":
        mono_constraints.append(1)
    elif col == "CoapplicantIncome":
        mono_constraints.append(1)
    elif col == "LoanAmount":
        mono_constraints.append(-1)
    elif col == "DTI":
        mono_constraints.append(-1)
    else:
        mono_constraints.append(0)

mono_string = "(" + ",".join(map(str, mono_constraints)) + ")"


# ============================================================
# 7. SETUP XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=350,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    monotone_constraints=mono_string,
    n_jobs=-1
)


# ============================================================
# 8. PIPELINE = PREPROCESSING + MODEL
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("Training XGBoost model...")
pipeline.fit(X_train, y_train)
print("Training complete ✔")


# ============================================================
# 11. EVALUATE MODEL
# ============================================================

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))


# ============================================================
# 12. SAVE MODEL
# ============================================================

joblib.dump(pipeline, "models/loan_xgb_monotonic.joblib")
print("\nModel saved as loan_xgb_monotonic.joblib ✔")


# ============================================================
# 13. SAVE MODEL PROVENANCE
# ============================================================

if PROVENANCE_AVAILABLE:
    print("\nCapturing model provenance...")

    # Calculate detailed metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Initialize tracker
    tracker = ProvenanceTracker()

    # Add model metadata
    model_metadata = {
        "model_path": "models/loan_xgb_monotonic.joblib",
        "model_type": "XGBoost",
        "model_class": "XGBClassifier",
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(df),
        "feature_count": len(X.columns),
        "features": list(X.columns),
        "target": "Loan_Status",
        "hyperparameters": {
            "n_estimators": 350,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            "n_jobs": -1
        },
        "monotonic_constraints": {
            "ApplicantIncome": +1,
            "CoapplicantIncome": +1,
            "LoanAmount": -1,
            "DTI": -1,
            "other": 0
        },
        "monotonic_constraints_string": mono_string,
        "performance_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        },
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": 42,
        "test_split_ratio": 0.2,
        "stratified": True
    }

    tracker.add_model_provenance(model_metadata)

    # Save provenance to JSON
    provenance_output = tracker.export_provenance()
    with open("models/model_provenance.json", "w") as f:
        json.dump(provenance_output, f, indent=2, default=str)

    print("Model provenance saved to models/model_provenance.json ✔")
else:
    print("\nSkipping provenance capture (module not available)")
