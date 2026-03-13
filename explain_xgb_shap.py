import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Load model & dataset
# ============================================================

MODEL_FILE = "loan_xgb_monotonic.joblib"
DATA_FILE = "df1_loan.csv"

print("Loading model...")
pipeline = joblib.load(MODEL_FILE)
print("Model loaded ✔")

# Load dataset
df = pd.read_csv(DATA_FILE)
df = df.drop(columns=["Unnamed: 0", "Loan_ID"], errors="ignore")

# Clean Total Income
df["Total_Income"] = df["Total_Income"].astype(str).replace({"$": ""}, regex=True)
df["Total_Income"] = pd.to_numeric(df["Total_Income"], errors="coerce")

# Convert Loan_Status
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Numeric cleanup
num_cols = [
    "LoanAmount", "ApplicantIncome", "CoapplicantIncome",
    "Loan_Amount_Term", "Credit_History", "Total_Income"
]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Add DTI
df["DTI"] = df["LoanAmount"] / (df["ApplicantIncome"] + df["CoapplicantIncome"] + 1)

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]


# ============================================================
# 2. Extract preprocessor & booster
# ============================================================

print("Extracting components...")
preprocessor = pipeline.named_steps["preprocess"]
xgb_model = pipeline.named_steps["model"]   # XGBClassifier
booster = xgb_model.get_booster()          # raw XGBoost booster


# ============================================================
# 3. Transform X using pipeline’s preprocessing
# ============================================================

print("Transforming features...")
X_transformed = preprocessor.transform(X)


# ============================================================
# 4. Extract transformed feature names
# ============================================================

def get_feature_names(preprocessor):
    output = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps"):
            if "onehot" in trans.named_steps:
                ohe = trans.named_steps["onehot"]
                output.extend(ohe.get_feature_names_out(cols))
            else:
                output.extend(cols)
        else:
            output.extend(cols)
    return np.array(output)

feature_names = get_feature_names(preprocessor)

print("\nFinal feature names:")
print(feature_names)


# ============================================================
# 5. Use SHAP TreeExplainer **directly on booster**
# ============================================================

print("\nBuilding TreeExplainer...")
explainer = shap.TreeExplainer(booster)

print("Computing SHAP values...")
shap_values = explainer.shap_values(X_transformed)
print("SHAP computed ✔")


# ============================================================
# 6. Summary Plot (Beeswarm)
# ============================================================

plt.figure()
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
plt.savefig("shap_summary_plot.png", bbox_inches="tight")
plt.close()
print("Saved: shap_summary_plot.png ✔")


# ============================================================
# 7. Bar Plot (Importance)
# ============================================================

plt.figure()
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names,
                  plot_type="bar", show=False)
plt.savefig("shap_feature_importance.png", bbox_inches="tight")
plt.close()
print("Saved: shap_feature_importance.png ✔")


# ============================================================
# 8. Dependence Plots
# ============================================================

important = ["DTI", "LoanAmount", "ApplicantIncome", "Credit_History"]

for feat in important:
    if feat in feature_names:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X_transformed, feature_names=feature_names, show=False)
        plt.savefig(f"shap_dependence_{feat}.png", bbox_inches="tight")
        plt.close()
        print(f"Saved: shap_dependence_{feat}.png ✔")

print("\nAll SHAP plots completed successfully!")
