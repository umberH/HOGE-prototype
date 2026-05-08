from neo4j import GraphDatabase
import pandas as pd
import os
import json
import datetime
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from backend.src.provenance.enhanced_provenance import ProvenanceTracker
    PROVENANCE_AVAILABLE = True
except ImportError:
    PROVENANCE_AVAILABLE = False
    print("Warning: Provenance tracking not available")

# Import config adapter for local/remote Neo4j switching
from backend.src.api.adapters.config import get_neo4j_config

# ============================================================
# CONFIG
# ============================================================

# Get Neo4j config (supports local/remote switching via USE_REMOTE_NEO4J flag)
neo4j_config = get_neo4j_config()
NEO4J_URI = neo4j_config["uri"]
NEO4J_USER = neo4j_config["user"]
NEO4J_PASSWORD = neo4j_config["password"]
NEO4J_DATABASE = neo4j_config["database"]

print(f"🔧 Using {neo4j_config['environment']} Neo4j: {NEO4J_URI}")

# Updated paths for new project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
APPLICATION_DATA_FILE = str(PROJECT_ROOT / ".resources" / "data" / "raw" / "df1_loan.csv")
SHAP_LONG_FILE = str(PROJECT_ROOT / ".resources" / "data" / "processed" / "shap_long.csv")


# ============================================================
# NEO4J DRIVER
# ============================================================

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ============================================================
# HELPER FUNCTION
# ============================================================

def run_query(q, params=None):
    with driver.session(database=NEO4J_DATABASE) as session:
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

COUNTERFACTUAL_FILE = str(PROJECT_ROOT / ".resources" / "data" / "evaluation" / "eval_counterfactual.json")

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

# ============================================================
# SAVE KG PROVENANCE
# ============================================================

if PROVENANCE_AVAILABLE:
    print("\nCapturing KG provenance...")

    # Query Neo4j for node and relationship counts
    def get_counts(query):
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query)
            return result.single()[0]

    # Get node counts by label
    node_count_query = """
    MATCH (n)
    RETURN labels(n)[0] as label, count(n) as count
    """

    # Get relationship counts by type
    rel_count_query = """
    MATCH ()-[r]->()
    RETURN type(r) as type, count(r) as count
    """

    # Get total counts
    total_nodes = get_counts("MATCH (n) RETURN count(n)")
    total_rels = get_counts("MATCH ()-[r]->() RETURN count(r)")

    # Get Neo4j version
    version_query = "CALL dbms.components() YIELD name, versions RETURN versions[0]"
    try:
        neo4j_version = get_counts(version_query)
    except:
        neo4j_version = "unknown"

    # Initialize tracker
    tracker = ProvenanceTracker()

    # Add KG metadata
    kg_metadata = {
        "kg_loader_version": "1.0.0",
        "neo4j_uri": NEO4J_URI,
        "neo4j_version": neo4j_version,
        "database_name": NEO4J_DATABASE,
        "total_nodes": int(total_nodes),
        "total_relationships": int(total_rels),
        "node_counts": {
            "LoanApplication": len(apps),
            "Applicant": len(apps),
            "Feature": "multiple",
            "FeatureValue": "multiple",
            "FeatureContribution": len(shap_long) if os.path.exists(SHAP_LONG_FILE) else 0,
            "PolicyRule": 4,  # Hardcoded rules count
        },
        "relationship_counts": {
            "HAS_APPLICATION": len(apps),
            "HAS_FEATURE_VALUE": "multiple",
            "HAS_SHAP_CONTRIBUTION": len(shap_long) if os.path.exists(SHAP_LONG_FILE) else 0,
            "VIOLATES_RULE": "conditional",
            "HAS_COUNTERFACTUAL": count if os.path.exists(COUNTERFACTUAL_FILE) else 0,
        },
        "data_sources": {
            "applications": APPLICATION_DATA_FILE,
            "shap_values": SHAP_LONG_FILE,
            "counterfactuals": COUNTERFACTUAL_FILE if os.path.exists("data/processed/counterfactuals.csv") else None,
        },
        "loading_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    tracker.add_kg_provenance(kg_metadata)

    # Save provenance to JSON
    provenance_output = tracker.export_provenance()
    provenance_dir = PROJECT_ROOT / ".resources" / "data" / "provenance"
    os.makedirs(provenance_dir, exist_ok=True)
    provenance_file = provenance_dir / "kg_provenance.json"
    with open(provenance_file, "w") as f:
        json.dump(provenance_output, f, indent=2, default=str)

    print(f"KG provenance saved to {provenance_file} ✔")
else:
    print("\nSkipping KG provenance capture (module not available)")

driver.close()
