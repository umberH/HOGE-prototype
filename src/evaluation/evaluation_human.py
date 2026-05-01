"""
HOGE Framework – Human-Centric Evaluation (Qualitative)
========================================================
Implements the qualitative evaluation for the HOGE paper:

B1. Sample check: predictions vs SHAP alignment
B2. Intermediate outcome Excel sheet
B3. LLM grounding verification (does LLM use all outcomes?)
B4. Print intermediate values and cross-check against LLM

Usage:
    python evaluation_human.py                          # run all, 10 samples
    python evaluation_human.py --n-samples 20           # 20 samples
    python evaluation_human.py --app-ids LP001002 LP001006  # specific apps

Outputs:
    - hoge_evaluation_workbook.xlsx  (multi-sheet Excel)
    - eval_human_results.json        (structured JSON)

Requires:
    - Neo4j running with KG loaded
    - OPENAI_API_KEY environment variable set
    - Trained model (loan_xgb_monotonic.joblib)
    - openpyxl: pip install openpyxl
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import joblib
import shap
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("WARNING: openpyxl not installed. Install with: pip install openpyxl")


# ============================================================
# CONFIG
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

MODEL_FILE = "models/loan_xgb_monotonic.joblib"
DATA_FILE = "data/raw/df1_loan.csv"
SHAP_LONG_FILE = "data/processed/shap_long.csv"
SHAP_WIDE_FILE = "data/processed/shap_wide.csv"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Excel styling
HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
PASS_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
FAIL_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
WARN_FILL = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
THIN_BORDER = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin")
)


# ============================================================
# SHARED HELPERS
# ============================================================

def load_model_and_data():
    pipeline = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE)

    if "Loan_ID" in df.columns:
        app_ids = df["Loan_ID"].copy()
    else:
        app_ids = pd.Series(range(len(df)), name="row_id")

    df_raw = df.copy()

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["Total_Income"] = (df["Total_Income"].astype(str)
                          .str.replace("$", "", regex=False)
                          .str.replace(",", "", regex=False))
    df["Total_Income"] = pd.to_numeric(df["Total_Income"], errors="coerce")
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    df = df.drop(columns=["Loan_ID"], errors="ignore")

    num_cols = ["LoanAmount", "ApplicantIncome", "CoapplicantIncome",
                "Loan_Amount_Term", "Credit_History", "Total_Income"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["DTI"] = df["LoanAmount"] / (df["ApplicantIncome"] + df["CoapplicantIncome"] + 1)

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]
    return pipeline, X, y, app_ids, df_raw


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


def call_openai(system_msg, user_msg, json_mode=True):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {"model": OPENAI_MODEL, "messages": [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ], "temperature": 0.1, "max_tokens": 2000}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def style_header_row(ws, num_cols):
    """Apply header styling to first row."""
    if not HAS_OPENPYXL:
        return
    for col_idx in range(1, num_cols + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = THIN_BORDER


def auto_width(ws, max_width=50):
    """Auto-adjust column widths."""
    if not HAS_OPENPYXL:
        return
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 2, max_width)


# ============================================================
# B1. SAMPLE CHECK: PREDICTIONS vs SHAP ALIGNMENT
# ============================================================

def check_prediction_shap_alignment(sample_ids, pipeline, X, y, app_ids, shap_long):
    """
    For each application:
    - Get model prediction and probability
    - Get SHAP values
    - Check if SHAP direction aligns with prediction
    - Check if top-K features make domain sense
    """
    print("\n" + "="*60)
    print("B1. PREDICTION vs SHAP ALIGNMENT CHECK")
    print("="*60)

    results = []
    for app_id in sample_ids:
        idx = app_ids[app_ids == app_id].index
        if len(idx) == 0:
            continue
        idx = idx[0]

        # Model prediction
        pred = pipeline.predict(X.iloc[[idx]])[0]
        prob = pipeline.predict_proba(X.iloc[[idx]])[:, 1][0]
        actual = y.iloc[idx]

        # SHAP data
        app_shap = shap_long[shap_long["application_id"] == app_id].sort_values(
            "abs_shap", ascending=False
        )

        total_shap = app_shap["shap_value"].sum()
        pos_shap = app_shap[app_shap["shap_value"] > 0]["shap_value"].sum()
        neg_shap = app_shap[app_shap["shap_value"] < 0]["shap_value"].sum()

        top3 = app_shap.head(3)
        top5 = app_shap.head(5)

        # Alignment checks
        # If approved (pred=1), net SHAP should be positive
        shap_direction = "positive" if total_shap > 0 else "negative"
        pred_direction = "positive" if pred == 1 else "negative"
        shap_pred_aligned = shap_direction == pred_direction

        # Check if prediction matches actual
        pred_correct = int(pred) == int(actual) if pd.notna(actual) else None

        result = {
            "application_id": app_id,
            "actual_label": int(actual) if pd.notna(actual) else None,
            "predicted_label": int(pred),
            "approval_probability": round(float(prob), 4),
            "prediction_correct": pred_correct,
            "total_shap_sum": round(float(total_shap), 4),
            "positive_shap_sum": round(float(pos_shap), 4),
            "negative_shap_sum": round(float(neg_shap), 4),
            "shap_pred_aligned": shap_pred_aligned,
            "top_1_feature": top3.iloc[0]["feature"] if len(top3) > 0 else None,
            "top_1_shap": round(float(top3.iloc[0]["shap_value"]), 4) if len(top3) > 0 else None,
            "top_1_direction": top3.iloc[0]["direction"] if len(top3) > 0 else None,
            "top_2_feature": top3.iloc[1]["feature"] if len(top3) > 1 else None,
            "top_2_shap": round(float(top3.iloc[1]["shap_value"]), 4) if len(top3) > 1 else None,
            "top_2_direction": top3.iloc[1]["direction"] if len(top3) > 1 else None,
            "top_3_feature": top3.iloc[2]["feature"] if len(top3) > 2 else None,
            "top_3_shap": round(float(top3.iloc[2]["shap_value"]), 4) if len(top3) > 2 else None,
            "top_3_direction": top3.iloc[2]["direction"] if len(top3) > 2 else None,
            "top5_features": ", ".join(top5["feature"].tolist()),
        }
        results.append(result)

        status = "ALIGNED" if shap_pred_aligned else "MISALIGNED"
        correct_str = "CORRECT" if pred_correct else "INCORRECT" if pred_correct is not None else "N/A"
        print(f"  {app_id}: pred={pred}, prob={prob:.4f}, actual={actual}, "
              f"prediction={correct_str}, shap_sum={total_shap:.4f} [{status}]")

    # Summary
    aligned_count = sum(1 for r in results if r["shap_pred_aligned"])
    correct_count = sum(1 for r in results if r["prediction_correct"])
    total = len(results)

    print(f"\n  Alignment rate: {aligned_count}/{total}")
    print(f"  Prediction accuracy: {correct_count}/{total}")

    return results


# ============================================================
# B2. INTERMEDIATE OUTCOME TRACKING
# ============================================================

def collect_intermediate_outcomes(sample_ids, pipeline, X, y, app_ids, df_raw, shap_long):
    """
    For each application, collect ALL intermediate outcomes:
    1. Raw input features
    2. Engineered features (DTI)
    3. Model prediction + probability
    4. SHAP values for every feature
    5. Policy rule evaluations
    6. KG-retrieved context
    7. LLM explanation (if API available)
    8. Cross-check: which intermediates appear in explanation
    """
    print("\n" + "="*60)
    print("B2. INTERMEDIATE OUTCOME COLLECTION")
    print("="*60)

    from llm_explainer import get_application_explanation_data, call_llm_for_explanation

    all_outcomes = []

    for app_id in sample_ids:
        print(f"\n--- Collecting outcomes for {app_id} ---")

        idx = app_ids[app_ids == app_id].index
        if len(idx) == 0:
            print(f"  Application {app_id} not found in dataset, skipping.")
            continue
        idx = idx[0]

        # ---- Stage 1: Raw Input Features ----
        raw_row = df_raw.iloc[idx]
        raw_features = {
            "ApplicantIncome": raw_row.get("ApplicantIncome"),
            "CoapplicantIncome": raw_row.get("CoapplicantIncome"),
            "LoanAmount": raw_row.get("LoanAmount"),
            "Loan_Amount_Term": raw_row.get("Loan_Amount_Term"),
            "Credit_History": raw_row.get("Credit_History"),
            "Total_Income": raw_row.get("Total_Income"),
            "Gender": raw_row.get("Gender"),
            "Married": raw_row.get("Married"),
            "Dependents": raw_row.get("Dependents"),
            "Education": raw_row.get("Education"),
            "Self_Employed": raw_row.get("Self_Employed"),
            "Property_Area": raw_row.get("Property_Area"),
        }

        # ---- Stage 2: Engineered Features ----
        dti = X.iloc[idx].get("DTI", None)
        engineered = {"DTI": round(float(dti), 6) if pd.notna(dti) else None}

        # ---- Stage 3: Model Prediction ----
        pred = int(pipeline.predict(X.iloc[[idx]])[0])
        prob = float(pipeline.predict_proba(X.iloc[[idx]])[:, 1][0])
        actual = int(y.iloc[idx]) if pd.notna(y.iloc[idx]) else None
        model_output = {
            "prediction": pred,
            "prediction_label": "Approved" if pred == 1 else "Declined",
            "approval_probability": round(prob, 6),
            "actual_label": actual,
        }

        # ---- Stage 4: SHAP Values ----
        app_shap = shap_long[shap_long["application_id"] == app_id].sort_values(
            "abs_shap", ascending=False
        )
        shap_features = {}
        for _, row in app_shap.iterrows():
            shap_features[row["feature"]] = {
                "shap_value": round(float(row["shap_value"]), 6),
                "abs_shap": round(float(row["abs_shap"]), 6),
                "direction": row["direction"],
                "rank": int(row["rank"]),
            }

        # ---- Stage 5: Policy Rule Evaluations ----
        policy_checks = {}
        credit_hist = raw_features.get("Credit_History")
        if credit_hist is not None:
            try:
                ch_val = float(credit_hist)
                policy_checks["RULE_CREDIT_001"] = {
                    "description": "Applicant must have clean credit history",
                    "feature_value": ch_val,
                    "threshold": 1,
                    "violated": ch_val != 1.0,
                }
            except (ValueError, TypeError):
                pass

        if dti is not None and pd.notna(dti):
            policy_checks["RULE_DTI_001"] = {
                "description": "Applications with DTI > 2.0 are declined",
                "feature_value": round(float(dti), 6),
                "threshold": 2.0,
                "violated": float(dti) > 2.0,
            }

        # ---- Stage 6: KG Context ----
        kg_context = None
        try:
            kg_context = get_application_explanation_data(app_id)
            kg_status = "retrieved"
        except Exception as e:
            kg_status = f"error: {e}"

        # ---- Stage 7: LLM Explanation ----
        llm_result = None
        if kg_context and OPENAI_API_KEY:
            try:
                llm_result = call_llm_for_explanation(kg_context)
                llm_status = "generated"
            except Exception as e:
                llm_status = f"error: {e}"
        else:
            llm_status = "skipped (no API key or no KG context)"

        # ---- Stage 8: Cross-Check LLM against intermediates ----
        crosscheck = {}
        if llm_result:
            narrative = llm_result.get("narrative", "")
            features_used = llm_result.get("features_used", [])

            # Check which raw features are mentioned
            for feat_name in raw_features:
                mentioned = feat_name.lower() in narrative.lower() or feat_name in features_used
                crosscheck[f"raw_{feat_name}_mentioned"] = mentioned

            # Check which SHAP top features are mentioned
            top5_shap = list(shap_features.keys())[:5]
            for feat_name in top5_shap:
                mentioned = feat_name.lower() in narrative.lower() or feat_name in features_used
                crosscheck[f"shap_top5_{feat_name}_mentioned"] = mentioned

            # Check policy violations mentioned
            for rule_id, rule_data in policy_checks.items():
                if rule_data["violated"]:
                    mentioned = rule_id.lower() in narrative.lower() or \
                                rule_data["description"].lower() in narrative.lower()
                    crosscheck[f"policy_{rule_id}_mentioned"] = mentioned

            # Calculate coverage scores
            shap_features_mentioned = sum(
                1 for f in top5_shap
                if f.lower() in narrative.lower() or f in features_used
            )
            crosscheck["shap_top5_coverage"] = shap_features_mentioned / max(len(top5_shap), 1)

            total_mentioned = sum(1 for v in crosscheck.values() if v is True)
            total_checks = sum(1 for v in crosscheck.values() if isinstance(v, bool))
            crosscheck["overall_coverage"] = total_mentioned / max(total_checks, 1)

        outcome = {
            "application_id": app_id,
            "stage_1_raw_features": raw_features,
            "stage_2_engineered_features": engineered,
            "stage_3_model_output": model_output,
            "stage_4_shap_values": shap_features,
            "stage_5_policy_checks": policy_checks,
            "stage_6_kg_status": kg_status,
            "stage_7_llm_status": llm_status,
            "stage_7_llm_narrative": llm_result.get("narrative", "") if llm_result else "",
            "stage_7_llm_features_used": llm_result.get("features_used", []) if llm_result else [],
            "stage_7_llm_positive_drivers": llm_result.get("positive_drivers", []) if llm_result else [],
            "stage_7_llm_negative_drivers": llm_result.get("negative_drivers", []) if llm_result else [],
            "stage_7_llm_policy_violations": llm_result.get("policy_violations", []) if llm_result else [],
            "stage_8_crosscheck": crosscheck,
        }
        all_outcomes.append(outcome)

        cov = crosscheck.get("overall_coverage", "N/A")
        shap_cov = crosscheck.get("shap_top5_coverage", "N/A")
        print(f"  Prediction: {model_output['prediction_label']} ({prob:.4f})")
        print(f"  LLM status: {llm_status}")
        print(f"  SHAP top-5 coverage in LLM: {shap_cov}")
        print(f"  Overall coverage: {cov}")

    return all_outcomes


# ============================================================
# B3. PRINT INTERMEDIATE VALUES & CROSS-CHECK
# ============================================================

def print_intermediate_crosscheck(outcomes):
    """
    Print all intermediate values for each application,
    then show which values the LLM used in its explanation.
    """
    print("\n" + "="*60)
    print("B3. INTERMEDIATE VALUES vs LLM CROSS-CHECK")
    print("="*60)

    for outcome in outcomes:
        app_id = outcome["application_id"]
        print(f"\n{'='*50}")
        print(f"APPLICATION: {app_id}")
        print(f"{'='*50}")

        # Stage 1: Raw Features
        print("\n[Stage 1] Raw Input Features:")
        for k, v in outcome["stage_1_raw_features"].items():
            print(f"  {k}: {v}")

        # Stage 2: Engineered
        print("\n[Stage 2] Engineered Features:")
        for k, v in outcome["stage_2_engineered_features"].items():
            print(f"  {k}: {v}")

        # Stage 3: Model Output
        print("\n[Stage 3] Model Output:")
        mo = outcome["stage_3_model_output"]
        print(f"  Prediction: {mo['prediction_label']} (label={mo['prediction']})")
        print(f"  Probability: {mo['approval_probability']}")
        print(f"  Actual: {mo['actual_label']}")

        # Stage 4: SHAP (top 5)
        print("\n[Stage 4] SHAP Values (top 5 by importance):")
        shap_items = list(outcome["stage_4_shap_values"].items())[:5]
        for feat, data in shap_items:
            print(f"  #{data['rank']} {feat}: shap={data['shap_value']:.4f} ({data['direction']})")

        # Stage 5: Policy
        print("\n[Stage 5] Policy Rule Checks:")
        for rule_id, rdata in outcome["stage_5_policy_checks"].items():
            status = "VIOLATED" if rdata["violated"] else "PASSED"
            print(f"  {rule_id}: {status} (value={rdata['feature_value']}, threshold={rdata['threshold']})")

        # Stage 6 & 7: KG + LLM
        print(f"\n[Stage 6] KG Status: {outcome['stage_6_kg_status']}")
        print(f"[Stage 7] LLM Status: {outcome['stage_7_llm_status']}")

        if outcome["stage_7_llm_narrative"]:
            print(f"\n[Stage 7] LLM Narrative (excerpt):")
            narrative = outcome["stage_7_llm_narrative"]
            print(f"  {narrative[:300]}{'...' if len(narrative) > 300 else ''}")

            print(f"\n[Stage 7] Features LLM claimed to use:")
            for f in outcome["stage_7_llm_features_used"]:
                print(f"  - {f}")

        # Stage 8: Cross-check
        print(f"\n[Stage 8] CROSS-CHECK (LLM vs Intermediates):")
        cc = outcome["stage_8_crosscheck"]
        for k, v in cc.items():
            if isinstance(v, bool):
                icon = "YES" if v else " NO"
                print(f"  [{icon}] {k}")
            elif isinstance(v, float):
                print(f"  [{v:.1%}] {k}")


# ============================================================
# EXCEL WORKBOOK GENERATOR
# ============================================================

def generate_excel_workbook(alignment_results, outcomes, output_file="data/evaluation/hoge_evaluation_workbook.xlsx"):
    """
    Create a multi-sheet Excel workbook with:
    - Sheet 1: Prediction vs SHAP Alignment
    - Sheet 2: Intermediate Outcomes (all stages)
    - Sheet 3: LLM Grounding Cross-Check
    - Sheet 4: Summary Dashboard
    """
    if not HAS_OPENPYXL:
        print("openpyxl not installed. Saving as CSV instead.")
        pd.DataFrame(alignment_results).to_csv("eval_alignment.csv", index=False)
        return

    wb = Workbook()

    # ---- Sheet 1: Prediction vs SHAP Alignment ----
    ws1 = wb.active
    ws1.title = "Prediction SHAP Alignment"

    headers1 = [
        "Application ID", "Actual Label", "Predicted Label", "Approval Prob",
        "Prediction Correct", "Total SHAP Sum", "Positive SHAP", "Negative SHAP",
        "SHAP-Pred Aligned", "Top 1 Feature", "Top 1 SHAP", "Top 1 Direction",
        "Top 2 Feature", "Top 2 SHAP", "Top 2 Direction",
        "Top 3 Feature", "Top 3 SHAP", "Top 3 Direction",
        "Top 5 Features"
    ]
    ws1.append(headers1)
    style_header_row(ws1, len(headers1))

    for r in alignment_results:
        row = [
            r["application_id"], r["actual_label"], r["predicted_label"],
            r["approval_probability"], r["prediction_correct"],
            r["total_shap_sum"], r["positive_shap_sum"], r["negative_shap_sum"],
            r["shap_pred_aligned"],
            r["top_1_feature"], r["top_1_shap"], r["top_1_direction"],
            r["top_2_feature"], r["top_2_shap"], r["top_2_direction"],
            r["top_3_feature"], r["top_3_shap"], r["top_3_direction"],
            r["top5_features"],
        ]
        ws1.append(row)

    # Color code alignment column
    for row_idx in range(2, len(alignment_results) + 2):
        cell = ws1.cell(row=row_idx, column=9)  # SHAP-Pred Aligned
        cell.fill = PASS_FILL if cell.value else FAIL_FILL
        cell_corr = ws1.cell(row=row_idx, column=5)  # Prediction Correct
        cell_corr.fill = PASS_FILL if cell_corr.value else FAIL_FILL

    auto_width(ws1)

    # ---- Sheet 2: Intermediate Outcomes ----
    ws2 = wb.create_sheet("Intermediate Outcomes")

    headers2 = [
        "Application ID",
        # Raw features
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Total_Income", "Gender", "Married", "Dependents",
        "Education", "Self_Employed", "Property_Area",
        # Engineered
        "DTI",
        # Model
        "Prediction", "Pred Label", "Probability", "Actual",
        # Top SHAP
        "SHAP Top1 Feature", "SHAP Top1 Value", "SHAP Top1 Dir",
        "SHAP Top2 Feature", "SHAP Top2 Value", "SHAP Top2 Dir",
        "SHAP Top3 Feature", "SHAP Top3 Value", "SHAP Top3 Dir",
        # Policy
        "DTI Rule Violated", "Credit Rule Violated",
        # KG & LLM
        "KG Status", "LLM Status",
        "LLM Narrative (first 200 chars)",
        "LLM Features Used",
    ]
    ws2.append(headers2)
    style_header_row(ws2, len(headers2))

    for o in outcomes:
        raw = o["stage_1_raw_features"]
        eng = o["stage_2_engineered_features"]
        mo = o["stage_3_model_output"]
        shap_vals = list(o["stage_4_shap_values"].items())
        policy = o["stage_5_policy_checks"]

        row = [
            o["application_id"],
            raw.get("ApplicantIncome"), raw.get("CoapplicantIncome"),
            raw.get("LoanAmount"), raw.get("Loan_Amount_Term"),
            raw.get("Credit_History"), raw.get("Total_Income"),
            raw.get("Gender"), raw.get("Married"), raw.get("Dependents"),
            raw.get("Education"), raw.get("Self_Employed"), raw.get("Property_Area"),
            eng.get("DTI"),
            mo["prediction"], mo["prediction_label"], mo["approval_probability"], mo["actual_label"],
        ]

        # Top 3 SHAP
        for i in range(3):
            if i < len(shap_vals):
                feat, data = shap_vals[i]
                row.extend([feat, data["shap_value"], data["direction"]])
            else:
                row.extend([None, None, None])

        # Policy
        dti_rule = policy.get("RULE_DTI_001", {})
        credit_rule = policy.get("RULE_CREDIT_001", {})
        row.append(dti_rule.get("violated", "N/A"))
        row.append(credit_rule.get("violated", "N/A"))

        # KG & LLM
        row.append(o["stage_6_kg_status"])
        row.append(o["stage_7_llm_status"])
        narrative = o.get("stage_7_llm_narrative", "")
        row.append(narrative[:200] if narrative else "")
        row.append(", ".join(o.get("stage_7_llm_features_used", [])))

        ws2.append(row)

    auto_width(ws2)

    # ---- Sheet 3: LLM Grounding Cross-Check ----
    ws3 = wb.create_sheet("LLM Grounding CrossCheck")

    headers3 = [
        "Application ID", "Check Type", "Item", "In LLM Narrative?", "Details"
    ]
    ws3.append(headers3)
    style_header_row(ws3, len(headers3))

    for o in outcomes:
        app_id = o["application_id"]
        cc = o["stage_8_crosscheck"]
        for key, val in cc.items():
            if isinstance(val, bool):
                parts = key.split("_", 1)
                check_type = parts[0] if len(parts) > 1 else "other"
                item = parts[1] if len(parts) > 1 else key
                row = [app_id, check_type, item, val, ""]
                row_idx = ws3.max_row + 1
                ws3.append(row)
                cell = ws3.cell(row=row_idx, column=4)
                cell.fill = PASS_FILL if val else FAIL_FILL

    auto_width(ws3)

    # ---- Sheet 4: Summary Dashboard ----
    ws4 = wb.create_sheet("Summary Dashboard")

    ws4.append(["HOGE Evaluation Summary"])
    ws4.cell(row=1, column=1).font = Font(bold=True, size=14)
    ws4.append(["Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    ws4.append([])

    # Alignment summary
    ws4.append(["Prediction vs SHAP Alignment"])
    ws4.cell(row=ws4.max_row, column=1).font = Font(bold=True, size=12)
    total = len(alignment_results)
    aligned = sum(1 for r in alignment_results if r["shap_pred_aligned"])
    correct = sum(1 for r in alignment_results if r["prediction_correct"])
    ws4.append(["Total Applications Sampled", total])
    ws4.append(["Predictions Correct", correct, f"{correct/max(total,1)*100:.1f}%"])
    ws4.append(["SHAP-Prediction Aligned", aligned, f"{aligned/max(total,1)*100:.1f}%"])
    ws4.append([])

    # LLM Coverage summary
    ws4.append(["LLM Grounding Coverage"])
    ws4.cell(row=ws4.max_row, column=1).font = Font(bold=True, size=12)
    coverages = [o["stage_8_crosscheck"].get("overall_coverage", 0) for o in outcomes if o["stage_8_crosscheck"]]
    shap_covs = [o["stage_8_crosscheck"].get("shap_top5_coverage", 0) for o in outcomes if o["stage_8_crosscheck"]]
    if coverages:
        ws4.append(["Avg Overall Coverage", f"{np.mean(coverages):.1%}"])
        ws4.append(["Avg SHAP Top-5 Coverage", f"{np.mean(shap_covs):.1%}"])

    auto_width(ws4)

    # Save
    wb.save(output_file)
    print(f"\nExcel workbook saved to: {output_file}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOGE Human-Centric Evaluation")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of applications to sample")
    parser.add_argument("--app-ids", nargs="+", default=None, help="Specific application IDs")
    parser.add_argument("--output", default="data/evaluation/hoge_evaluation_workbook.xlsx", help="Output Excel file")
    args = parser.parse_args()

    print("="*60)
    print("HOGE FRAMEWORK – HUMAN-CENTRIC EVALUATION")
    print("="*60)

    # Load data
    pipeline, X, y, app_ids, df_raw = load_model_and_data()
    shap_long = pd.read_csv(SHAP_LONG_FILE)

    # Select sample IDs
    if args.app_ids:
        sample_ids = args.app_ids
    else:
        unique_ids = shap_long["application_id"].unique()
        rng = np.random.RandomState(42)
        sample_ids = rng.choice(unique_ids, size=min(args.n_samples, len(unique_ids)), replace=False)

    print(f"\nSampled {len(sample_ids)} applications: {list(sample_ids)}")

    # B1: Prediction vs SHAP alignment
    alignment_results = check_prediction_shap_alignment(
        sample_ids, pipeline, X, y, app_ids, shap_long
    )

    # B2: Collect all intermediate outcomes
    outcomes = collect_intermediate_outcomes(
        sample_ids, pipeline, X, y, app_ids, df_raw, shap_long
    )

    # B3: Print cross-check
    print_intermediate_crosscheck(outcomes)

    # Generate Excel
    generate_excel_workbook(alignment_results, outcomes, args.output)

    # Save JSON
    json_output = {
        "evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "alignment_results": alignment_results,
        "intermediate_outcomes": outcomes,
    }
    with open("data/evaluation/eval_human_results.json", "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print("JSON results saved to: eval_human_results.json")

    print("\n" + "="*60)
    print("HUMAN-CENTRIC EVALUATION COMPLETE")
    print("="*60)
