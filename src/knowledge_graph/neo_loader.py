from neo4j import GraphDatabase
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD must be set in .env file")

APPLICATION_DATA_FILE = "data/raw/df1_loan.csv"
SHAP_LONG_FILE = "data/processed/shap_long.csv"


# ============================================================
# NEO4J DRIVER
# ============================================================

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ============================================================
# HELPER FUNCTION
# ============================================================

def run_query(q, params=None):
    with driver.session() as session:
        session.run(q, params or {})


# ============================================================
# 1. CREATE SCHEMA CONSTRAINTS
# ============================================================

print("Creating constraints...")

constraints = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Applicant) REQUIRE a.applicant_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (l:LoanApplication) REQUIRE l.application_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Feature) REQUIRE f.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:ModelExplanation) REQUIRE e.explanation_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (fc:FeatureContribution) REQUIRE fc.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (fv:FeatureValue) REQUIRE fv.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (cs:CounterfactualScenario) REQUIRE cs.id IS UNIQUE"
]

for c in constraints:
    run_query(c)

print("✔ Constraints created")


# ============================================================
# 2. LOAD APPLICATION DATA
# ============================================================

print("Loading application data CSV...")
apps = pd.read_csv(APPLICATION_DATA_FILE)
print(f"Loaded {len(apps)} application records")

application_query = """
MERGE (ap:Applicant {applicant_id: $Loan_ID})
SET ap.gender = $Gender,
    ap.married = $Married,
    ap.education = $Education,
    ap.self_employed = $Self_Employed,
    ap.dependents = $Dependents

MERGE (l:LoanApplication {application_id: $Loan_ID})
SET l.requested_amount = $LoanAmount,
    l.tenor_months = $Loan_Amount_Term,
    l.credit_history_flag = $Credit_History,
    l.property_area = $Property_Area,
    l.status = CASE $Loan_Status WHEN 'Y' THEN 'Approved' ELSE 'Declined' END

MERGE (ap)-[:HAS_APPLICATION]->(l)

WITH l

// RAW FEATURES
UNWIND [
    ['ApplicantIncome', $ApplicantIncome],
    ['CoapplicantIncome', $CoapplicantIncome],
    ['LoanAmount', $LoanAmount],
    ['Loan_Amount_Term', $Loan_Amount_Term],
    ['Credit_History', $Credit_History],
    ['Total_Income', $Total_Income],
    ['DTI', $DTI]
] AS fv

MERGE (f:Feature {name: fv[0]})
MERGE (val:FeatureValue {id: $Loan_ID + '_' + fv[0]})
SET val.value = fv[1],
    val.feature_name = fv[0]
MERGE (val)-[:OF_FEATURE]->(f)
MERGE (l)-[:HAS_FEATURE_VALUE]->(val)

WITH l

// ONE HOT FEATURES
UNWIND [
    ['Gender_Female', CASE WHEN $Gender = 'Female' THEN 1 ELSE 0 END],
    ['Gender_Male', CASE WHEN $Gender = 'Male' THEN 1 ELSE 0 END],

    ['Married_Yes', CASE WHEN $Married = 'Yes' THEN 1 ELSE 0 END],
    ['Married_No', CASE WHEN $Married = 'No' THEN 1 ELSE 0 END],

    ['Education_Graduate', CASE WHEN $Education = 'Graduate' THEN 1 ELSE 0 END],
    ['Education_Not Graduate', CASE WHEN $Education = 'Not Graduate' THEN 1 ELSE 0 END],

    ['Self_Employed_Yes', CASE WHEN $Self_Employed = 'Yes' THEN 1 ELSE 0 END],
    ['Self_Employed_No', CASE WHEN $Self_Employed = 'No' THEN 1 ELSE 0 END],

    ['Property_Area_Urban', CASE WHEN $Property_Area = 'Urban' THEN 1 ELSE 0 END],
    ['Property_Area_Rural', CASE WHEN $Property_Area = 'Rural' THEN 1 ELSE 0 END],
    ['Property_Area_Semiurban', CASE WHEN $Property_Area = 'Semiurban' THEN 1 ELSE 0 END],

    ['Dependents_0', CASE WHEN $Dependents = '0' THEN 1 ELSE 0 END],
    ['Dependents_1', CASE WHEN $Dependents = '1' THEN 1 ELSE 0 END],
    ['Dependents_2', CASE WHEN $Dependents = '2' THEN 1 ELSE 0 END],
    ['Dependents_3+', CASE WHEN $Dependents = '3+' THEN 1 ELSE 0 END]
] AS oh

MERGE (f2:Feature {name: oh[0]})
MERGE (val2:FeatureValue {id: $Loan_ID + '_' + oh[0]})
SET val2.value = oh[1],
    val2.feature_name = oh[0]
MERGE (val2)-[:OF_FEATURE]->(f2)
MERGE (l)-[:HAS_FEATURE_VALUE]->(val2)
"""

for _, row in apps.iterrows():
    run_query(application_query, row.to_dict())

print("✔ Application + Applicant + FeatureValues loaded")


# ============================================================
# 3. LOAD SHAP DATA
# ============================================================

print("Loading SHAP long CSV...")
shap_long = pd.read_csv(SHAP_LONG_FILE)
print(f"Loaded {len(shap_long)} SHAP rows")

shap_query = """
MERGE (l:LoanApplication {application_id: $application_id})

MERGE (exp:ModelExplanation {explanation_id: 'EXP_' + $application_id})
SET exp.model_name = 'xgb_monotonic_v1',
    exp.prediction = $prediction,
    exp.probability = $approval_probability,
    exp.timestamp = datetime()

MERGE (l)-[:HAS_SHAP_EXPLANATION]->(exp)

MERGE (f:Feature {name: $feature})

MERGE (fc:FeatureContribution {id: 'EXP_' + $application_id + '_' + $feature})
SET fc.shap_value = $shap_value,
    fc.abs_shap = $abs_shap,
    fc.rank = $rank,
    fc.direction = $direction

MERGE (exp)-[:HAS_CONTRIBUTION]->(fc)
MERGE (fc)-[:FOR_FEATURE]->(f)
"""

for _, row in shap_long.iterrows():
    run_query(shap_query, row.to_dict())

print("✔ SHAP explanations loaded")


# ============================================================
# 4. STATIC NODES: POLICY RULES & RISK FACTORS
# ============================================================

print("Creating Policy Rules + Risk Factors...")

static_statements = [
"""
MERGE (rule:PolicyRule {
    rule_id:'RULE_DTI_001',
    name:'DTI Hard Cutoff',
    feature_name:'DTI',
    operator:'>',
    threshold_value:2.0,
    severity:'HARD',
    description:'Applications with DTI > 2.0 are declined.'
})
""",
"""
MERGE (rule:PolicyRule {
    rule_id:'RULE_CREDIT_001',
    name:'Credit History Required',
    feature_name:'Credit_History',
    operator:'=',
    threshold_value:1,
    severity:'HARD',
    description:'Applicant must have clean credit history.'
})
""",
"""
MERGE (rf:RiskFactor {
    risk_id:'RF_HIGH_DTI',
    name:'High DTI',
    risk_type:'income'
})
""",
"""
MERGE (rf:RiskFactor {
    risk_id:'RF_BAD_CREDIT',
    name:'Bad Credit History',
    risk_type:'credit'
})
""",
"""
MATCH (f:Feature {name:'DTI'}),
      (r:PolicyRule {rule_id:'RULE_DTI_001'}),
      (rf:RiskFactor {risk_id:'RF_HIGH_DTI'})
MERGE (r)-[:IMPACTS]->(rf)
MERGE (rf)-[:DERIVED_FROM]->(f)
""",
"""
MATCH (f:Feature {name:'Credit_History'}),
      (r:PolicyRule {rule_id:'RULE_CREDIT_001'}),
      (rf:RiskFactor {risk_id:'RF_BAD_CREDIT'})
MERGE (r)-[:IMPACTS]->(rf)
MERGE (rf)-[:DERIVED_FROM]->(f)
"""
]

for stmt in static_statements:
    run_query(stmt)

print("✔ Policies + Risk Factors loaded")


# ============================================================
# 5. POLICY VIOLATIONS VIA FEATURES
# ============================================================

print("Linking SHAP contributions to policy violations...")

# DTI
run_query("""
MATCH (l:LoanApplication)-[:HAS_FEATURE_VALUE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature {name:'DTI'})
WHERE toFloat(fv.value) > 2.0
MATCH (exp:ModelExplanation {explanation_id: 'EXP_' + l.application_id})
MATCH (exp)-[:HAS_CONTRIBUTION]->(fc)-[:FOR_FEATURE]->(f)
MATCH (rule:PolicyRule {rule_id:'RULE_DTI_001'})

MERGE (fc)-[:LINKED_TO_RULE]->(rule)
MERGE (l)-[:VIOLATES_RULE]->(rule);

""")

# Credit history
run_query("""
MATCH (l:LoanApplication)-[:HAS_FEATURE_VALUE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature {name:'Credit_History'})
WHERE toInteger(fv.value) = 0
MATCH (exp:ModelExplanation {explanation_id: 'EXP_' + l.application_id})
MATCH (exp)-[:HAS_CONTRIBUTION]->(fc)-[:FOR_FEATURE]->(f)
MATCH (rule:PolicyRule {rule_id:'RULE_CREDIT_001'})

MERGE (fc)-[:LINKED_TO_RULE]->(rule)
MERGE (l)-[:VIOLATES_RULE]->(rule);

""")

print("✔ Policy violations linked")

# ============================================================
# DONE
# ============================================================

# ============================================================
# 6. LOAD COUNTERFACTUAL SCENARIOS (if available)
# ============================================================

import json, os

COUNTERFACTUAL_FILE = "data/evaluation/eval_counterfactual.json"

if os.path.exists(COUNTERFACTUAL_FILE):
    print("Loading counterfactual scenarios...")
    with open(COUNTERFACTUAL_FILE) as f:
        cf_results = json.load(f)

    cf_query = """
    MATCH (l:LoanApplication {application_id: $application_id})
    MERGE (cs:CounterfactualScenario {id: $cf_id})
    SET cs.feature = $feature,
        cs.current_value = $current_value,
        cs.changed_to = $changed_to,
        cs.change_description = $change_description,
        cs.original_probability = $original_probability,
        cs.new_probability = $new_probability,
        cs.probability_shift = $probability_shift,
        cs.flipped_prediction = $flipped_prediction,
        cs.new_prediction = $new_prediction,
        cs.is_minimal_flip = $is_minimal_flip
    MERGE (l)-[:HAS_COUNTERFACTUAL]->(cs)
    MERGE (f:Feature {name: $feature})
    MERGE (cs)-[:PERTURBS_FEATURE]->(f)
    """

    count = 0
    for app_result in cf_results:
        app_id = app_result["application_id"]
        minimal_flip = app_result.get("minimal_flip")
        minimal_flip_feature = minimal_flip["feature"] if minimal_flip else None
        minimal_flip_change = minimal_flip["change_description"] if minimal_flip else None

        for i, scenario in enumerate(app_result.get("counterfactual_scenarios", [])):
            is_minimal = (
                scenario["feature"] == minimal_flip_feature
                and scenario["change_description"] == minimal_flip_change
            ) if minimal_flip else False

            params = {
                "application_id": app_id,
                "cf_id": f"CF_{app_id}_{i}",
                "feature": scenario["feature"],
                "current_value": float(scenario["current_value"]) if scenario["current_value"] is not None else 0.0,
                "changed_to": float(scenario["changed_to"]),
                "change_description": scenario["change_description"],
                "original_probability": scenario["original_probability"],
                "new_probability": scenario["new_probability"],
                "probability_shift": scenario["probability_shift"],
                "flipped_prediction": scenario["flipped_prediction"],
                "new_prediction": scenario["new_prediction"],
                "is_minimal_flip": is_minimal,
            }
            run_query(cf_query, params)
            count += 1

    print(f"✔ {count} counterfactual scenarios loaded for {len(cf_results)} applications")
else:
    print(f"⚠ {COUNTERFACTUAL_FILE} not found — skipping counterfactual ingestion")
    print("  Run: python counterfactual_explainer.py --all  to generate it first")


print("\n🚀 Neo4j KG successfully built!")
driver.close()
