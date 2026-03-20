"""
HOGE Counterfactual Explanation Module

For each application, computes the minimal actionable feature changes
needed to flip the model's prediction (approved ↔ declined).

This addresses Miller's contrastive property: "Why X instead of Y?"
and enables what-if analysis for loan applicants and officers.
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import joblib
import shap
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = "loan_xgb_monotonic.joblib"
RAW_DATA_FILE = "df1_loan.csv"
SHAP_LONG_FILE = "shap_long.csv"

# Features that an applicant can realistically change
ACTIONABLE_FEATURES = [
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

# Features that cannot be changed (demographic / immutable)
IMMUTABLE_FEATURES = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "ApplicantIncome",
]


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

def load_resources():
    pipeline = joblib.load(MODEL_FILE)

    # Load raw data and prepare features (same as train_model.py)
    df = pd.read_csv(RAW_DATA_FILE)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Clean Total_Income
    df["Total_Income"] = (df["Total_Income"].astype(str)
                          .str.replace("$", "", regex=False)
                          .str.replace(",", "", regex=False))
    df["Total_Income"] = pd.to_numeric(df["Total_Income"], errors="coerce")

    # Convert Loan_Status
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    # Ensure numeric
    num_cols = ["LoanAmount", "ApplicantIncome", "CoapplicantIncome",
                "Loan_Amount_Term", "Credit_History", "Total_Income"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # DTI
    df["DTI"] = df["LoanAmount"] / (df["ApplicantIncome"] + df["CoapplicantIncome"] + 1)

    shap_long = pd.read_csv(SHAP_LONG_FILE)

    # Score only the applications we need (per-row to avoid NaN issues)
    # Predictions will be computed on demand in generate_counterfactual
    return pipeline, df, shap_long


# ============================================================
# COUNTERFACTUAL GENERATION
# ============================================================

def generate_counterfactual(pipeline, df, shap_long_df, application_id):
    """
    For a given application, find the minimal actionable changes
    that would flip the prediction.

    Strategy:
    1. Get current prediction and probability
    2. Get SHAP contributions for actionable features
    3. For each actionable feature, compute the effect of changing it
    4. Find the single-feature and minimal multi-feature flips
    """

    # Get the application row
    app_row = df[df["Loan_ID"] == application_id]
    if app_row.empty:
        raise ValueError(f"Application {application_id} not found")

    app_row = app_row.iloc[0]

    # Build feature vector and compute prediction
    feature_cols = [c for c in df.columns
                    if c not in ["Loan_ID", "Loan_Status"]]
    X_orig = pd.DataFrame([app_row[feature_cols].to_dict()])

    current_prob = float(pipeline.predict_proba(X_orig)[:, 1][0])
    current_pred = int(current_prob >= 0.5)
    target_pred = 1 - current_pred  # flip
    target_label = "Approved" if target_pred == 1 else "Declined"
    current_label = "Approved" if current_pred == 1 else "Declined"

    # Threshold: probability needs to cross 0.5
    prob_gap = abs(current_prob - 0.5)

    # Get SHAP values for this application
    app_shap = shap_long_df[shap_long_df["application_id"] == application_id].copy()

    # Build original feature dict
    original_features = app_row[feature_cols].to_dict()

    # Prepare the original dataframe for prediction (single row)
    X_original = X_orig.copy()

    # --- Single-feature counterfactuals ---
    counterfactuals = []

    for _, shap_row in app_shap.iterrows():
        feature = shap_row["feature"]
        shap_val = float(shap_row["shap_value"])
        direction = shap_row["direction"]

        # Only consider actionable features
        if feature not in ACTIONABLE_FEATURES:
            # Also check one-hot encoded versions
            base_feature = feature.split("_")[0] if "_" in feature else feature
            if base_feature not in ACTIONABLE_FEATURES and feature not in ACTIONABLE_FEATURES:
                continue

        current_value = original_features.get(feature)
        if current_value is None:
            continue

        # Determine counterfactual change based on feature type
        cf_changes = []

        if feature == "Credit_History":
            # Binary: flip it
            new_val = 1.0 if float(current_value) == 0 else 0.0
            cf_changes.append(("flip", new_val))

        elif feature == "LoanAmount":
            # Try reducing/increasing by 25%, 50%
            cv = float(current_value)
            if current_pred == 0:  # declined → try reducing loan
                cf_changes.append(("reduce by 25%", cv * 0.75))
                cf_changes.append(("reduce by 50%", cv * 0.50))
            else:  # approved → try increasing loan
                cf_changes.append(("increase by 50%", cv * 1.50))
                cf_changes.append(("increase by 100%", cv * 2.00))

        elif feature == "CoapplicantIncome":
            cv = float(current_value)
            if current_pred == 0:  # declined → try adding co-applicant income
                if cv == 0:
                    cf_changes.append(("add co-applicant income of 3000", 3000.0))
                    cf_changes.append(("add co-applicant income of 5000", 5000.0))
                else:
                    cf_changes.append(("increase by 50%", cv * 1.50))
                    cf_changes.append(("double", cv * 2.00))
            else:  # approved → try removing co-applicant
                cf_changes.append(("remove co-applicant", 0.0))

        elif feature == "Loan_Amount_Term":
            cv = float(current_value)
            if cv == 360:
                cf_changes.append(("reduce to 180 months", 180.0))
            elif cv < 360:
                cf_changes.append(("extend to 360 months", 360.0))

        elif feature.startswith("Property_Area"):
            # Categorical — skip individual one-hot, handle as group
            continue

        # Test each counterfactual change
        for change_desc, new_value in cf_changes:
            X_cf = X_original.copy()
            X_cf[feature] = new_value

            # Also update DTI if income or loan amount changed
            if feature in ["LoanAmount", "CoapplicantIncome", "ApplicantIncome"]:
                ai = float(X_cf["ApplicantIncome"].iloc[0])
                ci = float(X_cf["CoapplicantIncome"].iloc[0])
                la = float(X_cf["LoanAmount"].iloc[0])
                X_cf["DTI"] = la / (ai + ci + 1)
                if feature in ["CoapplicantIncome", "ApplicantIncome"]:
                    X_cf["Total_Income"] = ai + ci

            try:
                new_prob = float(pipeline.predict_proba(X_cf)[:, 1][0])
                new_pred = int(new_prob >= 0.5)
                flipped = (new_pred != current_pred)

                counterfactuals.append({
                    "feature": feature,
                    "current_value": current_value,
                    "changed_to": new_value,
                    "change_description": change_desc,
                    "original_probability": round(current_prob, 4),
                    "new_probability": round(new_prob, 4),
                    "probability_shift": round(new_prob - current_prob, 4),
                    "flipped_prediction": flipped,
                    "new_prediction": "Approved" if new_pred == 1 else "Declined",
                    "shap_importance": abs(shap_val),
                })
            except Exception:
                continue

    # Sort: flipped first, then by probability shift magnitude
    counterfactuals.sort(
        key=lambda x: (not x["flipped_prediction"], -abs(x["probability_shift"]))
    )

    # Find the minimal flip (fewest changes)
    flipping_changes = [c for c in counterfactuals if c["flipped_prediction"]]
    minimal_flip = flipping_changes[0] if flipping_changes else None

    result = {
        "application_id": application_id,
        "current_prediction": current_label,
        "current_probability": round(current_prob, 4),
        "target_prediction": target_label,
        "probability_gap_to_flip": round(prob_gap, 4),
        "counterfactual_scenarios": counterfactuals,
        "minimal_flip": minimal_flip,
        "actionable_features_tested": list(set(
            c["feature"] for c in counterfactuals
        )),
        "total_scenarios_tested": len(counterfactuals),
        "flipping_scenarios_found": len(flipping_changes),
    }

    return result


# ============================================================
# RUN FOR EVALUATION SAMPLE
# ============================================================

EVAL_SAMPLE_DEFAULT = [
    "LP002170", "LP001250", "LP002209", "LP001536", "LP001357",
    "LP002266", "LP002223", "LP001439", "LP001238", "LP002446",
]


def run_all(sample_ids=None):
    pipeline, df, shap_long = load_resources()
    all_results = []

    if sample_ids is None:
        sample_ids = EVAL_SAMPLE_DEFAULT

    for app_id in sample_ids:
        try:
            result = generate_counterfactual(pipeline, df, shap_long, app_id)
            all_results.append(result)

            flip = result["minimal_flip"]
            if flip:
                print(f"{app_id} ({result['current_prediction']}, "
                      f"p={result['current_probability']:.3f}): "
                      f"FLIP by {flip['change_description']} on {flip['feature']} "
                      f"→ {flip['new_prediction']} (p={flip['new_probability']:.3f})")
            else:
                print(f"{app_id} ({result['current_prediction']}, "
                      f"p={result['current_probability']:.3f}): "
                      f"No single-feature flip found "
                      f"({result['total_scenarios_tested']} scenarios tested)")
        except Exception as e:
            print(f"{app_id}: ERROR - {e}")

    # Save results
    with open("eval_counterfactual.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to eval_counterfactual.json ({len(all_results)} applications)")

    return all_results


# ============================================================
# SINGLE APPLICATION
# ============================================================

def run_single(application_id):
    pipeline, df, shap_long = load_resources()
    result = generate_counterfactual(pipeline, df, shap_long, application_id)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HOGE Counterfactual Explainer")
    parser.add_argument("app_id", nargs="?", default=None,
                        help="Single application ID to analyze")
    parser.add_argument("--all", action="store_true",
                        help="Run for default 10 evaluation samples")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of applications to sample (uses seed 42)")
    args = parser.parse_args()

    if args.n_samples is not None:
        # Dynamic sampling: same seed as evaluation scripts for consistency
        shap_long = pd.read_csv(SHAP_LONG_FILE)
        unique_ids = shap_long["application_id"].unique()
        rng = np.random.RandomState(42)
        sample_ids = list(rng.choice(unique_ids,
                                     size=min(args.n_samples, len(unique_ids)),
                                     replace=False))
        print(f"Sampling {len(sample_ids)} applications (seed=42)")
        run_all(sample_ids=sample_ids)
    elif args.app_id and args.app_id != "--all":
        run_single(args.app_id)
    else:
        run_all()
