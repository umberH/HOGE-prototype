import os
import sys
import json
import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

# ============================================================
# CONFIG  (credentials via environment variables)
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Provenance metadata (HOGE requirement: auditability)
PROVENANCE = {
    "model_name": "xgb_monotonic_v1",
    "xai_method": "TreeSHAP",
    "kg_loader_version": "neo_loader_v1",
    "hoge_version": "1.0.0",
}


# ============================================================
# NEO4J HELPER
# ============================================================

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_application_explanation_data(application_id: str):
    """
    HOGE Component B – Semantic Knowledge Retrieval.

    Pulls from the KG:
      - prediction & probability
      - SHAP contributions (with feature values)
      - violated policy rules
      - risk factors linked to features
      - evidence_bundle: list of KG paths backing each claim

    Returns a dict ready to feed into the LLM (Component C).
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
            exp.model_name AS model_name,
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

        status = shap_results[0]["status"]
        probability = shap_results[0]["probability"]
        model_name = shap_results[0].get("model_name", PROVENANCE["model_name"])

        shap_details = []
        evidence_bundle = []
        for row in shap_results:
            if row["feature"] is None:
                continue

            feat_entry = {
                "feature": row["feature"],
                "value": row["input_value"],
                "shap": float(row["shap_value"]) if row["shap_value"] is not None else None,
                "importance": float(row["importance"]) if row["importance"] is not None else None,
                "rank": int(row["rank"]) if row["rank"] is not None else None,
                "direction": row["direction"],
            }
            shap_details.append(feat_entry)

            # Evidence bundle: record KG path for each claim
            evidence_bundle.append({
                "claim_type": "feature_contribution",
                "feature": row["feature"],
                "kg_path": f"(LoanApplication:{application_id})-[:HAS_SHAP_EXPLANATION]->"
                           f"(ModelExplanation)-[:HAS_CONTRIBUTION]->(FeatureContribution)"
                           f"-[:FOR_FEATURE]->(Feature:{row['feature']})",
                "value": row["input_value"],
                "shap_value": float(row["shap_value"]) if row["shap_value"] is not None else None,
                "direction": row["direction"],
            })

        # ---------- 2) Violated Rules ----------
        rules_query = """
        MATCH (app:LoanApplication {application_id: $application_id})
        OPTIONAL MATCH (app)-[:VIOLATES_RULE]->(r:PolicyRule)
        OPTIONAL MATCH (r)-[:IMPACTS]->(rf:RiskFactor)
        RETURN r.rule_id AS rule_id,
               r.description AS description,
               r.severity AS severity,
               r.feature_name AS rule_feature,
               r.operator AS operator,
               r.threshold_value AS threshold,
               rf.name AS risk_factor_name,
               rf.risk_type AS risk_type
        """

        rule_rows = session.run(rules_query, {"application_id": application_id}).data()

        violated_rules = []
        for r in rule_rows:
            if r["rule_id"] is None:
                continue
            rule_entry = {
                "rule_id": r["rule_id"],
                "description": r["description"],
                "severity": r["severity"],
                "rule_feature": r["rule_feature"],
                "operator": r["operator"],
                "threshold": r["threshold"],
                "risk_factor": r.get("risk_factor_name"),
                "risk_type": r.get("risk_type"),
            }
            violated_rules.append(rule_entry)

            evidence_bundle.append({
                "claim_type": "policy_violation",
                "rule_id": r["rule_id"],
                "kg_path": f"(LoanApplication:{application_id})-[:VIOLATES_RULE]->"
                           f"(PolicyRule:{r['rule_id']})-[:IMPACTS]->(RiskFactor:{r.get('risk_factor_name')})",
                "description": r["description"],
                "severity": r["severity"],
            })

        # ---------- 3) All policy rules (for grounding context) ----------
        all_rules_query = """
        MATCH (r:PolicyRule)
        RETURN r.rule_id AS rule_id,
               r.description AS description,
               r.severity AS severity,
               r.feature_name AS feature_name,
               r.operator AS operator,
               r.threshold_value AS threshold
        """
        all_rules = session.run(all_rules_query).data()

        # ---------- 4) Counterfactual scenarios from KG ----------
        cf_query = """
        MATCH (app:LoanApplication {application_id: $application_id})
              -[:HAS_COUNTERFACTUAL]->(cs:CounterfactualScenario)
        RETURN cs.feature AS feature,
               cs.current_value AS current_value,
               cs.changed_to AS changed_to,
               cs.change_description AS change_description,
               cs.original_probability AS original_probability,
               cs.new_probability AS new_probability,
               cs.probability_shift AS probability_shift,
               cs.flipped_prediction AS flipped_prediction,
               cs.new_prediction AS new_prediction,
               cs.is_minimal_flip AS is_minimal_flip
        ORDER BY cs.is_minimal_flip DESC, abs(cs.probability_shift) DESC
        """
        cf_rows = session.run(cf_query, {"application_id": application_id}).data()

        counterfactual_scenarios = []
        for cf in cf_rows:
            counterfactual_scenarios.append({
                "feature": cf["feature"],
                "current_value": cf["current_value"],
                "changed_to": cf["changed_to"],
                "change_description": cf["change_description"],
                "original_probability": cf["original_probability"],
                "new_probability": cf["new_probability"],
                "probability_shift": cf["probability_shift"],
                "flipped_prediction": cf["flipped_prediction"],
                "new_prediction": cf["new_prediction"],
                "is_minimal_flip": cf["is_minimal_flip"],
            })
            evidence_bundle.append({
                "claim_type": "counterfactual",
                "feature": cf["feature"],
                "kg_path": f"(LoanApplication:{application_id})-[:HAS_COUNTERFACTUAL]->"
                           f"(CounterfactualScenario)-[:PERTURBS_FEATURE]->(Feature:{cf['feature']})",
                "change_description": cf["change_description"],
                "probability_shift": cf["probability_shift"],
                "flipped": cf["flipped_prediction"],
            })

    return {
        "application_id": application_id,
        "model_prediction": status,
        "approval_probability": float(probability) if probability is not None else None,
        "shap_details": shap_details,
        "violated_rules": violated_rules,
        "all_policy_rules": all_rules,
        "counterfactual_scenarios": counterfactual_scenarios,
        "evidence_bundle": evidence_bundle,
        "provenance": {
            **PROVENANCE,
            "model_name": model_name,
            "retrieval_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    }


# ============================================================
# LLM HELPER
# ============================================================

def build_llm_prompt(context: dict) -> str:
    """
    HOGE Component C – Narrative Explainer.

    Builds a constraint-aware prompt that:
    - Provides ONLY ontology-grounded facts (semantic contract)
    - Requires the LLM to cite evidence for each claim
    - Asks for structured JSON + plain-text explanation
    - Prevents hallucination by listing ALL available policy rules
    """

    application_id = context["application_id"]
    prediction = context["model_prediction"]
    prob = context["approval_probability"]
    shap_details = context["shap_details"]
    violated_rules = context["violated_rules"]
    all_policy_rules = context.get("all_policy_rules", [])

    # Sort by importance
    shap_details_sorted = sorted(
        [s for s in shap_details if s.get("importance") is not None],
        key=lambda x: x["importance"],
        reverse=True,
    )

    positives = [s for s in shap_details_sorted if s["shap"] is not None and s["shap"] > 0]
    negatives = [s for s in shap_details_sorted if s["shap"] is not None and s["shap"] < 0]

    top_pos = positives[:5]
    top_neg = negatives[:5]

    def fmt_feat(s):
        return f"{s['feature']} (value={s['value']}, shap={s['shap']:.3f}, direction={s['direction']})"

    pos_str = "\n".join([f"- {fmt_feat(s)}" for s in top_pos]) or "None"
    neg_str = "\n".join([f"- {fmt_feat(s)}" for s in top_neg]) or "None"

    # All SHAP features for completeness check
    all_features_str = "\n".join(
        [f"- {fmt_feat(s)}" for s in shap_details_sorted]
    )

    # Policy rules (violated)
    if violated_rules:
        rules_str = "\n".join(
            [f"- {r['rule_id']} ({r['severity']}): {r['description']} "
             f"[feature: {r.get('rule_feature', 'N/A')}, "
             f"operator: {r.get('operator', 'N/A')}, "
             f"threshold: {r.get('threshold', 'N/A')}]"
             for r in violated_rules]
        )
    else:
        rules_str = "No policy rules were violated for this application."

    # All known policy rules (ontology grounding constraint)
    if all_policy_rules:
        all_rules_str = "\n".join(
            [f"- {r['rule_id']}: {r['description']} "
             f"(feature={r.get('feature_name','N/A')}, "
             f"op={r.get('operator','N/A')}, "
             f"threshold={r.get('threshold','N/A')}, "
             f"severity={r.get('severity','N/A')})"
             for r in all_policy_rules]
        )
    else:
        all_rules_str = "No policy rules are defined in the ontology."

    prob_pct = f"{prob*100:.1f}%" if prob is not None else "N/A"

    # Counterfactual what-if scenarios (if provided)
    counterfactuals = context.get("counterfactual_scenarios", [])
    if counterfactuals:
        cf_lines = []
        for cf in counterfactuals:
            direction = "increased" if cf["probability_shift"] > 0 else "decreased"
            cf_lines.append(
                f"- If {cf['feature']} changed from {cf['current_value']} to "
                f"{cf['changed_to']} ({cf['change_description']}): "
                f"approval probability would {direction} from "
                f"{cf['original_probability']*100:.1f}% to {cf['new_probability']*100:.1f}%"
                f"{' (FLIPS decision)' if cf['flipped_prediction'] else ''}"
            )
        cf_str = "\n".join(cf_lines)
    else:
        cf_str = "No counterfactual scenarios available."

    prompt = f"""You are an explainable-AI assistant operating under the HOGE framework.
You explain loan decisions using ONLY the evidence provided below.

=== APPLICATION ID: {application_id} ===

Model output:
- Decision: {prediction}
- Approval probability: {prob_pct}

ALL feature contributions from the model (ranked by importance):
{all_features_str}

Top positive SHAP drivers (increased approval score):
{pos_str}

Top negative SHAP drivers (decreased approval score):
{neg_str}

Policy rules violated for this application:
{rules_str}

ALL known policy rules in the domain ontology:
{all_rules_str}

Counterfactual what-if scenarios (what could change the decision):
{cf_str}

=== STRICT GROUNDING RULES ===
1. You must ONLY reference features and rules listed above. Do NOT invent or assume any fact not provided.
2. Every claim must be traceable to the evidence above.
3. Do NOT mention SHAP values, log-odds, or any technical jargon.
4. If a HARD policy rule was violated, state it explicitly as the primary reason for decline.
5. Do NOT promise that the decision can be changed.

=== OUTPUT FORMAT ===
You must respond with a valid JSON object with EXACTLY these keys:

{{
  "summary": "One-sentence summary of the decision",
  "positive_drivers": ["List of plain-language sentences about features supporting approval"],
  "negative_drivers": ["List of plain-language sentences about features supporting decline"],
  "policy_violations": ["List of plain-language sentences about violated rules, or empty list"],
  "recommendation": "One actionable suggestion for future applications",
  "counterfactual_what_if": "One sentence describing the most impactful change that could flip the decision, based ONLY on the counterfactual scenarios above. If none available, say so.",
  "narrative": "Full 2-4 paragraph plain-language explanation combining all of the above, including a what-if paragraph",
  "features_used": ["List of feature names referenced in the narrative"]
}}

Now generate the explanation.
"""
    return prompt


def call_llm_for_explanation(context: dict) -> dict:
    """
    Calls the LLM with ontology-grounded context and returns
    both structured JSON and the raw narrative.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it with: export OPENAI_API_KEY=sk-..."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = build_llm_prompt(context)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a responsible, cautious explainable-AI assistant in a bank "
                    "operating under the HOGE (Human-Centric Ontology-Grounded Explanation) "
                    "framework. You explain model decisions clearly and fairly using ONLY "
                    "the evidence provided. You never reveal internal model parameters. "
                    "You always respond with valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content.strip()

    try:
        structured = json.loads(raw_text)
    except json.JSONDecodeError:
        structured = {
            "summary": raw_text,
            "narrative": raw_text,
            "positive_drivers": [],
            "negative_drivers": [],
            "policy_violations": [],
            "recommendation": "",
            "features_used": [],
            "parse_error": True,
        }

    # Attach provenance and evidence bundle to the output
    structured["provenance"] = context.get("provenance", {})
    structured["evidence_bundle"] = context.get("evidence_bundle", [])

    return structured


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

    # Counterfactual scenarios are now retrieved from the KG
    # (loaded by neo_loader.py from eval_counterfactual.json)
    cf_count = len(context.get("counterfactual_scenarios", []))
    if cf_count > 0:
        flips = sum(1 for c in context["counterfactual_scenarios"] if c["flipped_prediction"])
        print(f"Counterfactual scenarios from KG: {cf_count} scenarios, {flips} flip(s)")
    else:
        print("No counterfactual scenarios found in KG for this application")

    print("\n=== RAW CONTEXT (for debugging) ===")
    print(json.dumps(context, indent=2, default=str))

    print("\n=== LLM EXPLANATION ===\n")
    try:
        result = call_llm_for_explanation(context)

        # Print structured output
        print("--- Summary ---")
        print(result.get("summary", ""))

        print("\n--- Positive Drivers ---")
        for d in result.get("positive_drivers", []):
            print(f"  + {d}")

        print("\n--- Negative Drivers ---")
        for d in result.get("negative_drivers", []):
            print(f"  - {d}")

        print("\n--- Policy Violations ---")
        for v in result.get("policy_violations", []):
            print(f"  ! {v}")

        print("\n--- Recommendation ---")
        print(result.get("recommendation", ""))

        print("\n--- Counterfactual What-If ---")
        print(result.get("counterfactual_what_if", "N/A"))

        print("\n--- Full Narrative ---")
        print(result.get("narrative", ""))

        print("\n--- Features Used by LLM ---")
        print(result.get("features_used", []))

        print("\n--- Provenance ---")
        print(json.dumps(result.get("provenance", {}), indent=2, default=str))

        # Save full result for evaluation
        output_file = f"data/evaluation/explanation_{application_id}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull explanation saved to: {output_file}")

    except Exception as e:
        print(f"Error calling LLM: {e}")
