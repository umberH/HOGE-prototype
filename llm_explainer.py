import os
import sys
import json
from neo4j import GraphDatabase
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "test1234"  # <-- change if needed

OPENAI_API_KEY = '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'
OPENAI_MODEL = "gpt-4o"  # or "gpt-4o-mini", "gpt-4.1", etc.


# ============================================================
# NEO4J HELPER
# ============================================================

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_application_explanation_data(application_id: str):
    """
    Pulls:
      - prediction & probability
      - SHAP contributions (with feature values)
      - violated policy rules
    for a given application_id.
    Returns a dict ready to feed into the LLM.
    """

    with driver.session() as session:
        # ---------- 1) Prediction + SHAP rows ----------
        shap_query = """
        MATCH (app:LoanApplication {application_id: $application_id})
        OPTIONAL MATCH (app)-[:HAS_SHAP_EXPLANATION]->(exp:ModelExplanation)
        OPTIONAL MATCH (exp)-[:HAS_CONTRIBUTION]->(fc:FeatureContribution)-[:FOR_FEATURE]->(f:Feature)
        OPTIONAL MATCH (app)-[:HAS_FEATURE_VALUE]->(fv:FeatureValue)-[:OF_FEATURE]->(f)
        RETURN
            app.status AS status,
            exp.probability AS probability,
            f.name AS feature,
            fv.value AS input_value,
            fc.shap_value AS shap_value,
            fc.abs_shap AS importance,
            fc.rank AS rank,
            fc.direction AS direction
        ORDER BY importance DESC
        """

        shap_results = session.run(shap_query, {"application_id": application_id}).data()

        if not shap_results:
            raise ValueError(f"No SHAP/model data found for application_id={application_id}")

        # Extract overall status + probability from first row
        status = shap_results[0]["status"]
        probability = shap_results[0]["probability"]

        shap_details = []
        for row in shap_results:
            if row["feature"] is None:
                continue
            shap_details.append(
                {
                    "feature": row["feature"],
                    "value": row["input_value"],
                    "shap": float(row["shap_value"]) if row["shap_value"] is not None else None,
                    "importance": float(row["importance"]) if row["importance"] is not None else None,
                    "rank": int(row["rank"]) if row["rank"] is not None else None,
                    "direction": row["direction"],
                }
            )

        # ---------- 2) Violated Rules ----------
        rules_query = """
        MATCH (app:LoanApplication {application_id: $application_id})
        OPTIONAL MATCH (app)-[:VIOLATES_RULE]->(r:PolicyRule)
        RETURN r.rule_id AS rule_id, r.description AS description, r.severity AS severity
        """

        rule_rows = session.run(rules_query, {"application_id": application_id}).data()

        violated_rules = []
        for r in rule_rows:
            if r["rule_id"] is None:
                continue
            violated_rules.append(
                {
                    "rule_id": r["rule_id"],
                    "description": r["description"],
                    "severity": r["severity"],
                }
            )

    return {
        "application_id": application_id,
        "model_prediction": status,
        "approval_probability": float(probability) if probability is not None else None,
        "shap_details": shap_details,
        "violated_rules": violated_rules,
    }


# ============================================================
# LLM HELPER
# ============================================================

def build_llm_prompt(context: dict) -> str:
    """
    Builds a natural-language prompt for the LLM
    using the model output + SHAP + policy rules.
    """

    application_id = context["application_id"]
    prediction = context["model_prediction"]  # "Approved" / "Declined"
    prob = context["approval_probability"]
    shap_details = context["shap_details"]
    violated_rules = context["violated_rules"]

    # Sort by importance (defensive)
    shap_details_sorted = sorted(
        [s for s in shap_details if s.get("importance") is not None],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # Top 5 positive & top 5 negative contributors
    positives = [s for s in shap_details_sorted if s["shap"] is not None and s["shap"] > 0]
    negatives = [s for s in shap_details_sorted if s["shap"] is not None and s["shap"] < 0]

    top_pos = positives[:5]
    top_neg = negatives[:5]

    # Build a summary string for SHAP
    def fmt_feat(s):
        return f"{s['feature']} (value={s['value']}, shap={s['shap']:.3f}, direction={s['direction']})"

    pos_str = "\n".join([f"- {fmt_feat(s)}" for s in top_pos]) or "None"
    neg_str = "\n".join([f"- {fmt_feat(s)}" for s in top_neg]) or "None"

    # Policies
    if violated_rules:
        rules_str = "\n".join(
            [f"- {r['rule_id']} ({r['severity']}): {r['description']}" for r in violated_rules]
        )
    else:
        rules_str = "No explicit policy rules were violated for this application."

    prob_pct = f"{prob*100:.1f}%" if prob is not None else "N/A"

    prompt = f"""
You are an explainable-AI assistant helping a loan officer understand model decisions.

We have a loan application with ID: {application_id}.

Model output:
- Decision: {prediction}
- Approval probability: {prob_pct}

Top positive SHAP drivers (features that increased the approval score):
{pos_str}

Top negative SHAP drivers (features that decreased the approval score):
{neg_str}

Policy rules that were violated or relevant:
{rules_str}

Requirements:
1. Start with a one-sentence summary: e.g. "Your application was approved because..."
2. Then explain the MAIN positive drivers in plain language (Total_Income, CoapplicantIncome, Credit_History, etc.).
3. Then explain the MAIN negative drivers (e.g. high DTI, lower income, bad credit, risk-related features).
4. If any HARD rules were violated, explicitly mention that this overrides the score and leads to decline.
5. Avoid technical jargon like "SHAP value" or "log-odds". Talk like a human loan officer.
6. Keep it factual, and do NOT promise that the decision can be changed.
7. Finish with a short suggestion, e.g. "Improving X or Y could strengthen future applications".

Now, write the explanation for the applicant in 2–4 short paragraphs.
"""

    return prompt


def call_llm_for_explanation(context: dict) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = build_llm_prompt(context)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a responsible, cautious explainable-AI assistant in a bank. "
                    "You explain model decisions clearly and fairly, without making promises, "
                    "and you never reveal internal model parameters."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        application_id = sys.argv[1]
    else:
        application_id = input("Enter application ID (e.g. LP001006): ").strip()

    try:
        context = get_application_explanation_data(application_id)
    except Exception as e:
        print(f"Error fetching explanation data: {e}")
        sys.exit(1)

    print("\n=== RAW CONTEXT (for debugging) ===")
    print(json.dumps(context, indent=2, default=str))

    print("\n=== LLM EXPLANATION ===\n")
    try:
        explanation = call_llm_for_explanation(context)
        print(explanation)
    except Exception as e:
        print(f"Error calling LLM: {e}")
