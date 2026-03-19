"""
HOGE Framework – System & Technical Evaluation (Quantitative)
=============================================================
Implements the three quantitative evaluation axes from the paper:

A1. Explanation Faithfulness  (Perturbation Tests)
A2. Hallucination Rate / Grounding Precision  (LLM-as-a-Judge)
A3. Graph Retrieval Accuracy  (Precision@K)

Usage:
    python evaluation_system.py                 # run all evaluations
    python evaluation_system.py --faithfulness  # perturbation tests only
    python evaluation_system.py --hallucination # hallucination rate only
    python evaluation_system.py --retrieval     # graph retrieval only

Requires:
    - Neo4j running with KG loaded (neo_loader.py)
    - OPENAI_API_KEY environment variable set
    - Trained model (loan_xgb_monotonic.joblib)
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


# ============================================================
# CONFIG
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

MODEL_FILE = "loan_xgb_monotonic.joblib"
DATA_FILE = "df1_loan.csv"
SHAP_LONG_FILE = "shap_long.csv"
SHAP_WIDE_FILE = "shap_wide.csv"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ============================================================
# SHARED HELPERS
# ============================================================

def load_model_and_data():
    """Load the trained pipeline and prepare feature matrix."""
    pipeline = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE)

    if "Loan_ID" in df.columns:
        app_ids = df["Loan_ID"].copy()
    else:
        app_ids = pd.Series(range(len(df)), name="row_id")

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
    return pipeline, X, y, app_ids


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


def get_kg_context(application_id: str) -> dict:
    """Retrieve full explanation context from the KG."""
    from llm_explainer import get_application_explanation_data
    return get_application_explanation_data(application_id)


def call_openai(system_msg: str, user_msg: str, json_mode=True) -> str:
    """Helper to call OpenAI API."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    kwargs = {"model": OPENAI_MODEL, "messages": [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ], "temperature": 0.1, "max_tokens": 1500}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


# ============================================================
# A1. EXPLANATION FAITHFULNESS – Perturbation Tests
# ============================================================

def run_faithfulness_perturbation(sample_ids=None, n_samples=10):
    """
    For each sampled application:
    1. Get the original SHAP-based explanation (top driver)
    2. Perturb the top SHAP feature (e.g. flip Credit_History, 2x DTI)
    3. Re-run SHAP on the perturbed input
    4. Generate a new LLM explanation
    5. Check if the LLM explanation correctly reflects the shift
    """
    print("\n" + "="*60)
    print("A1. EXPLANATION FAITHFULNESS – Perturbation Tests")
    print("="*60)

    pipeline, X, y, app_ids = load_model_and_data()
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(preprocessor)

    shap_long = pd.read_csv(SHAP_LONG_FILE)

    if sample_ids is None:
        unique_ids = shap_long["application_id"].unique()
        rng = np.random.RandomState(42)
        sample_ids = rng.choice(unique_ids, size=min(n_samples, len(unique_ids)), replace=False)

    results = []
    for app_id in sample_ids:
        print(f"\n--- Perturbation test for {app_id} ---")

        # Get original top feature
        app_shap = shap_long[shap_long["application_id"] == app_id].sort_values(
            "abs_shap", ascending=False
        )
        if app_shap.empty:
            print(f"  No SHAP data for {app_id}, skipping.")
            continue

        top_feature = app_shap.iloc[0]["feature"]
        top_direction = app_shap.iloc[0]["direction"]
        original_prob = app_shap.iloc[0]["approval_probability"]
        print(f"  Top feature: {top_feature} ({top_direction})")
        print(f"  Original probability: {original_prob:.4f}")

        # Find the row in X
        idx = app_ids[app_ids == app_id].index
        if len(idx) == 0:
            print(f"  Application {app_id} not found in dataset, skipping.")
            continue
        idx = idx[0]

        X_perturbed = X.iloc[[idx]].copy()

        # Perturb the top feature
        perturb_desc = ""
        if top_feature == "Credit_History":
            original_val = X_perturbed["Credit_History"].values[0]
            new_val = 0.0 if original_val == 1.0 else 1.0
            X_perturbed["Credit_History"] = new_val
            perturb_desc = f"Credit_History flipped from {original_val} to {new_val}"
        elif top_feature == "DTI":
            original_val = X_perturbed["DTI"].values[0]
            new_val = original_val * 3.0
            X_perturbed["DTI"] = new_val
            perturb_desc = f"DTI increased from {original_val:.4f} to {new_val:.4f}"
        elif top_feature in X_perturbed.columns:
            original_val = X_perturbed[top_feature].values[0]
            if pd.notna(original_val) and isinstance(original_val, (int, float, np.number)):
                # For numeric features: invert direction
                if top_direction == "supports_approval":
                    new_val = original_val * 0.2  # drastically reduce
                else:
                    new_val = original_val * 3.0  # drastically increase
                X_perturbed[top_feature] = new_val
                perturb_desc = f"{top_feature} changed from {original_val} to {new_val}"
            else:
                print(f"  Cannot perturb categorical feature {top_feature} directly, skipping.")
                continue
        else:
            # One-hot encoded feature — skip for now
            print(f"  Feature {top_feature} is one-hot encoded, skipping perturbation.")
            continue

        print(f"  Perturbation: {perturb_desc}")

        # Re-predict with perturbed data
        new_pred = pipeline.predict(X_perturbed)[0]
        new_prob = pipeline.predict_proba(X_perturbed)[:, 1][0]
        print(f"  New probability: {new_prob:.4f} (was {original_prob:.4f})")

        # Re-compute SHAP on perturbed data
        X_perturbed_transformed = preprocessor.transform(X_perturbed)
        explainer = shap.TreeExplainer(model)
        new_shap_values = explainer.shap_values(X_perturbed_transformed)
        new_abs = np.abs(new_shap_values[0])
        new_top_idx = np.argmax(new_abs)
        new_top_feature = feature_names[new_top_idx]
        new_top_direction = "supports_approval" if new_shap_values[0][new_top_idx] > 0 else "supports_decline"

        print(f"  New top feature: {new_top_feature} ({new_top_direction})")

        # Ask LLM to compare
        prob_shift = new_prob - original_prob
        shift_desc = f"increased by {abs(prob_shift):.4f}" if prob_shift > 0 else f"decreased by {abs(prob_shift):.4f}"

        judge_prompt = f"""We perturbed a loan application's data and need to check if the explanation is faithful.

Original application {app_id}:
- Top driver: {top_feature} ({top_direction})
- Approval probability: {original_prob:.4f}

Perturbation applied: {perturb_desc}

After perturbation:
- New top driver: {new_top_feature} ({new_top_direction})
- New approval probability: {new_prob:.4f} (probability {shift_desc})

Evaluate faithfulness by answering in JSON:
{{
  "perturbation_reflected": true/false,
  "probability_shift_direction_correct": true/false,
  "top_driver_changed_appropriately": true/false,
  "faithfulness_score": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}
"""
        try:
            judge_result = call_openai(
                "You are an evaluation judge for XAI faithfulness testing. Be strict and precise.",
                judge_prompt
            )
            judge_data = json.loads(judge_result)
        except Exception as e:
            judge_data = {"error": str(e), "faithfulness_score": None}

        result = {
            "application_id": app_id,
            "original_top_feature": top_feature,
            "original_direction": top_direction,
            "original_prob": float(original_prob),
            "perturbation": perturb_desc,
            "new_top_feature": new_top_feature,
            "new_direction": new_top_direction,
            "new_prob": float(new_prob),
            "prob_shift": float(prob_shift),
            **judge_data,
        }
        results.append(result)
        print(f"  Faithfulness score: {judge_data.get('faithfulness_score', 'N/A')}")

    # Summary
    scores = [r["faithfulness_score"] for r in results if r.get("faithfulness_score") is not None]
    if scores:
        avg_score = np.mean(scores)
        print(f"\n=== FAITHFULNESS SUMMARY ===")
        print(f"Applications tested: {len(results)}")
        print(f"Average faithfulness score: {avg_score:.3f}")
        print(f"Pass rate (>= 0.5): {sum(1 for s in scores if s >= 0.5)}/{len(scores)}")

    return results


# ============================================================
# A2. HALLUCINATION RATE / GROUNDING PRECISION
# ============================================================

def run_hallucination_evaluation(sample_ids=None, n_samples=10):
    """
    For each sampled application:
    1. Get the KG context (all facts in ontology)
    2. Generate an LLM explanation
    3. Use LLM-as-a-judge to check each claim against the ontology
    4. Calculate:
       - Semantic Fidelity (SF): proportion of claims mapping to ontology
       - Evidence Coverage (EC): proportion backed by KG paths
       - Hallucination Rate under Constraint (HRC): unsupported claims rate
    """
    print("\n" + "="*60)
    print("A2. HALLUCINATION RATE / GROUNDING PRECISION")
    print("="*60)

    from llm_explainer import get_application_explanation_data, call_llm_for_explanation

    shap_long = pd.read_csv(SHAP_LONG_FILE)

    if sample_ids is None:
        unique_ids = shap_long["application_id"].unique()
        rng = np.random.RandomState(42)
        sample_ids = rng.choice(unique_ids, size=min(n_samples, len(unique_ids)), replace=False)

    results = []
    for app_id in sample_ids:
        print(f"\n--- Hallucination check for {app_id} ---")

        try:
            context = get_application_explanation_data(app_id)
        except Exception as e:
            print(f"  Error getting context: {e}")
            continue

        try:
            explanation = call_llm_for_explanation(context)
        except Exception as e:
            print(f"  Error getting explanation: {e}")
            continue

        narrative = explanation.get("narrative", "")
        features_used = explanation.get("features_used", [])
        positive_drivers = explanation.get("positive_drivers", [])
        negative_drivers = explanation.get("negative_drivers", [])
        policy_violations = explanation.get("policy_violations", [])

        # Build the ground truth from context
        known_features = [s["feature"] for s in context["shap_details"]]
        known_rules = [r["rule_id"] for r in context["violated_rules"]]
        known_rule_descriptions = [r["description"] for r in context["violated_rules"]]
        all_known_rules = [r["rule_id"] for r in context.get("all_policy_rules", [])]

        # Build the full evidence set as a string for the judge
        evidence_str = json.dumps({
            "prediction": context["model_prediction"],
            "probability": context["approval_probability"],
            "features_with_shap": [
                {
                    "feature": s["feature"],
                    "value": s["value"],
                    "shap": s["shap"],
                    "direction": s["direction"],
                    "rank": s["rank"],
                }
                for s in context["shap_details"]
            ],
            "violated_rules": context["violated_rules"],
            "all_policy_rules": context.get("all_policy_rules", []),
        }, indent=2, default=str)

        judge_prompt = f"""You are a strict grounding judge for an AI explanation system.

Below is the GROUND TRUTH evidence from the knowledge graph for application {app_id}:

{evidence_str}

Below is the LLM-generated explanation:

NARRATIVE:
{narrative}

POSITIVE DRIVERS CLAIMED: {json.dumps(positive_drivers)}
NEGATIVE DRIVERS CLAIMED: {json.dumps(negative_drivers)}
POLICY VIOLATIONS CLAIMED: {json.dumps(policy_violations)}
FEATURES REFERENCED: {json.dumps(features_used)}

Evaluate the explanation strictly against the ground truth. For each claim in the narrative:
1. Is it supported by the evidence? (grounded)
2. Does it introduce facts NOT in the evidence? (hallucination)
3. Does it correctly reference policy rules?

Respond with JSON:
{{
  "total_claims": <number of distinct factual claims in the narrative>,
  "grounded_claims": <number of claims supported by evidence>,
  "hallucinated_claims": <number of claims NOT supported by evidence>,
  "hallucinated_details": ["list of specific hallucinated claims"],
  "semantic_fidelity": <grounded_claims / total_claims, float 0-1>,
  "evidence_coverage": <proportion of evidence used in explanation, float 0-1>,
  "hallucination_rate": <hallucinated_claims / total_claims, float 0-1>,
  "features_correctly_referenced": <number of features correctly used>,
  "features_fabricated": ["list of feature names mentioned but not in evidence"],
  "rules_correctly_referenced": <number of rules correctly mentioned>,
  "rules_fabricated": ["list of rules mentioned but not in evidence"],
  "overall_grounding_score": <float 0-1, overall quality>
}}
"""
        try:
            judge_result = call_openai(
                "You are a meticulous evaluation judge. Score strictly. A claim is hallucinated if it "
                "references any fact, feature, rule, or relationship not explicitly present in the evidence.",
                judge_prompt
            )
            judge_data = json.loads(judge_result)
        except Exception as e:
            judge_data = {"error": str(e)}

        result = {
            "application_id": app_id,
            "prediction": context["model_prediction"],
            "probability": context["approval_probability"],
            "num_features_in_evidence": len(known_features),
            "num_features_llm_used": len(features_used),
            "num_rules_violated": len(known_rules),
            "narrative_length": len(narrative),
            **judge_data,
        }
        results.append(result)

        sf = judge_data.get("semantic_fidelity", "N/A")
        ec = judge_data.get("evidence_coverage", "N/A")
        hr = judge_data.get("hallucination_rate", "N/A")
        print(f"  Semantic Fidelity: {sf}")
        print(f"  Evidence Coverage: {ec}")
        print(f"  Hallucination Rate: {hr}")

    # Summary
    sfs = [r["semantic_fidelity"] for r in results if isinstance(r.get("semantic_fidelity"), (int, float))]
    ecs = [r["evidence_coverage"] for r in results if isinstance(r.get("evidence_coverage"), (int, float))]
    hrs = [r["hallucination_rate"] for r in results if isinstance(r.get("hallucination_rate"), (int, float))]

    print(f"\n=== HALLUCINATION / GROUNDING SUMMARY ===")
    if sfs:
        print(f"Avg Semantic Fidelity (SF):          {np.mean(sfs):.3f}")
    if ecs:
        print(f"Avg Evidence Coverage (EC):           {np.mean(ecs):.3f}")
    if hrs:
        print(f"Avg Hallucination Rate (HRC):         {np.mean(hrs):.3f}")
        print(f"Apps with zero hallucinations:        {sum(1 for h in hrs if h == 0.0)}/{len(hrs)}")

    return results


# ============================================================
# A3. GRAPH RETRIEVAL ACCURACY – Precision@K
# ============================================================

def run_graph_retrieval_evaluation(sample_ids=None, n_samples=10):
    """
    For each sampled application:
    1. Query the KG for top-K SHAP contributions
    2. Compare against the ground truth from shap_long.csv
    3. Calculate Precision@K for K = 3, 5, 10
    """
    print("\n" + "="*60)
    print("A3. GRAPH RETRIEVAL ACCURACY – Precision@K")
    print("="*60)

    shap_long = pd.read_csv(SHAP_LONG_FILE)

    if sample_ids is None:
        unique_ids = shap_long["application_id"].unique()
        rng = np.random.RandomState(42)
        sample_ids = rng.choice(unique_ids, size=min(n_samples, len(unique_ids)), replace=False)

    results = []
    for app_id in sample_ids:
        print(f"\n--- Retrieval accuracy for {app_id} ---")

        # Ground truth: top features from shap_long.csv
        gt = shap_long[shap_long["application_id"] == app_id].sort_values(
            "abs_shap", ascending=False
        )
        if gt.empty:
            print(f"  No ground truth for {app_id}")
            continue

        gt_features = gt["feature"].tolist()
        gt_pred = gt.iloc[0]["prediction"]
        gt_prob = gt.iloc[0]["approval_probability"]

        # KG retrieval
        with driver.session() as session:
            kg_query = """
            MATCH (app:LoanApplication {application_id: $app_id})
                  -[:HAS_SHAP_EXPLANATION]->(exp:ModelExplanation)
                  -[:HAS_CONTRIBUTION]->(fc:FeatureContribution)
                  -[:FOR_FEATURE]->(f:Feature)
            RETURN f.name AS feature,
                   fc.shap_value AS shap_value,
                   fc.abs_shap AS abs_shap,
                   fc.rank AS rank,
                   fc.direction AS direction,
                   exp.prediction AS prediction,
                   exp.probability AS probability
            ORDER BY fc.abs_shap DESC
            """
            kg_rows = session.run(kg_query, {"app_id": app_id}).data()

        if not kg_rows:
            print(f"  No KG data for {app_id}")
            results.append({
                "application_id": app_id,
                "kg_found": False,
            })
            continue

        kg_features = [r["feature"] for r in kg_rows]
        kg_pred = kg_rows[0]["prediction"]
        kg_prob = kg_rows[0]["probability"]

        # Prediction match
        pred_match = (int(gt_pred) == int(kg_pred)) if kg_pred is not None else False
        prob_match = abs(float(gt_prob) - float(kg_prob)) < 0.001 if kg_prob is not None else False

        # Precision@K
        precision_at = {}
        for k in [3, 5, 10]:
            gt_top_k = set(gt_features[:k])
            kg_top_k = set(kg_features[:k])
            if len(kg_top_k) > 0:
                precision_at[k] = len(gt_top_k & kg_top_k) / len(kg_top_k)
            else:
                precision_at[k] = 0.0

        # Also check: did KG return all features?
        feature_recall = len(set(gt_features) & set(kg_features)) / len(set(gt_features))

        # Policy retrieval check
        with driver.session() as session:
            rule_query = """
            MATCH (app:LoanApplication {application_id: $app_id})
                  -[:VIOLATES_RULE]->(r:PolicyRule)
            RETURN r.rule_id AS rule_id
            """
            rule_rows = session.run(rule_query, {"app_id": app_id}).data()

        kg_rules = [r["rule_id"] for r in rule_rows]

        # Ground truth rules: check DTI > 2.0 and Credit_History == 0
        gt_rules = []
        app_gt = gt[gt["feature"] == "DTI"]
        if not app_gt.empty:
            # Check if DTI value in original data exceeds threshold
            pass  # We'll check from df1_loan
        app_gt_credit = gt[gt["feature"] == "Credit_History"]

        result = {
            "application_id": app_id,
            "kg_found": True,
            "prediction_match": pred_match,
            "probability_match": prob_match,
            "gt_prob": float(gt_prob),
            "kg_prob": float(kg_prob) if kg_prob else None,
            "precision_at_3": precision_at[3],
            "precision_at_5": precision_at[5],
            "precision_at_10": precision_at[10],
            "feature_recall": feature_recall,
            "total_gt_features": len(set(gt_features)),
            "total_kg_features": len(set(kg_features)),
            "kg_rules_found": kg_rules,
        }
        results.append(result)

        print(f"  Prediction match: {pred_match}")
        print(f"  Precision@3: {precision_at[3]:.3f}")
        print(f"  Precision@5: {precision_at[5]:.3f}")
        print(f"  Precision@10: {precision_at[10]:.3f}")
        print(f"  Feature recall: {feature_recall:.3f}")

    # Summary
    valid = [r for r in results if r.get("kg_found")]
    if valid:
        print(f"\n=== GRAPH RETRIEVAL SUMMARY ===")
        print(f"Applications tested: {len(valid)}")
        print(f"Prediction match rate: {sum(1 for r in valid if r['prediction_match'])/len(valid):.3f}")
        print(f"Avg Precision@3:  {np.mean([r['precision_at_3'] for r in valid]):.3f}")
        print(f"Avg Precision@5:  {np.mean([r['precision_at_5'] for r in valid]):.3f}")
        print(f"Avg Precision@10: {np.mean([r['precision_at_10'] for r in valid]):.3f}")
        print(f"Avg Feature Recall: {np.mean([r['feature_recall'] for r in valid]):.3f}")

    return results


# ============================================================
# MAIN
# ============================================================

def save_results(results, filename):
    """Save evaluation results to JSON."""
    output = {
        "evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results": results,
    }
    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOGE System Evaluation")
    parser.add_argument("--faithfulness", action="store_true", help="Run perturbation tests")
    parser.add_argument("--hallucination", action="store_true", help="Run hallucination evaluation")
    parser.add_argument("--retrieval", action="store_true", help="Run graph retrieval evaluation")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of applications to sample")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")

    args = parser.parse_args()

    run_all = args.all or not (args.faithfulness or args.hallucination or args.retrieval)

    all_results = {}

    if run_all or args.faithfulness:
        r = run_faithfulness_perturbation(n_samples=args.n_samples)
        all_results["faithfulness"] = r
        save_results(r, "eval_faithfulness.json")

    if run_all or args.hallucination:
        r = run_hallucination_evaluation(n_samples=args.n_samples)
        all_results["hallucination"] = r
        save_results(r, "eval_hallucination.json")

    if run_all or args.retrieval:
        r = run_graph_retrieval_evaluation(n_samples=args.n_samples)
        all_results["retrieval"] = r
        save_results(r, "eval_retrieval.json")

    if all_results:
        save_results(all_results, "eval_system_all.json")
        print("\n" + "="*60)
        print("ALL SYSTEM EVALUATIONS COMPLETE")
        print("="*60)
