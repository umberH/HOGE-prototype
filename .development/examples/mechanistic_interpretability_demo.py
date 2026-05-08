"""
Mechanistic Interpretability Demo for HOGE Framework

This script demonstrates how to use mechanistic interpretability
to understand the internal workings of the XGBoost model beyond
standard SHAP explanations.

Mechanistic interpretability reveals:
1. Decision paths through individual trees
2. Feature interactions and dependencies
3. Tree-level specialization patterns
4. Internal activation patterns

Run this after training the model and loading SHAP data into Neo4j.

Usage:
    python mechanistic_interpretability_demo.py LP001006
    python mechanistic_interpretability_demo.py --all  # Analyze multiple applications
"""

import os
import sys
import pandas as pd
import json
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from explainability.mechanistic_interpreter import MechanisticInterpreter

# Optional Neo4j integration
try:
    from knowledge_graph.neo_mechanistic_loader import MechanisticKGLoader
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Note: Neo4j module not available. Install with: pip install neo4j")


def analyze_single_application(application_id: str,
                               load_to_neo4j: bool = True) -> dict:
    """
    Perform complete mechanistic analysis for one application.

    Args:
        application_id: Application to analyze
        load_to_neo4j: Whether to load results into Neo4j

    Returns:
        Dictionary with all mechanistic insights
    """
    print(f"\n{'='*70}")
    print(f"MECHANISTIC INTERPRETABILITY ANALYSIS: {application_id}")
    print(f"{'='*70}\n")

    # Configuration
    MODEL_PATH = "models/loan_xgb_monotonic.joblib"
    RAW_DATA_PATH = "data/raw/df1_loan.csv"

    # Load raw data with Loan_ID
    df_raw = pd.read_csv(RAW_DATA_PATH)

    # Clean data (same as train_model.py)
    df_raw = df_raw.drop(columns=["Unnamed: 0"], errors="ignore")
    df_raw["Total_Income"] = (df_raw["Total_Income"].astype(str)
                              .str.replace("$", "", regex=False)
                              .str.replace(",", "", regex=False))
    df_raw["Total_Income"] = pd.to_numeric(df_raw["Total_Income"], errors="coerce")

    # Create DTI
    df_raw["DTI"] = df_raw["LoanAmount"] / (df_raw["ApplicantIncome"] + df_raw["CoapplicantIncome"] + 1)
    df_raw["DTI"] = pd.to_numeric(df_raw["DTI"], errors="coerce")

    # Filter to application
    app_data = df_raw[df_raw['Loan_ID'] == application_id]

    if app_data.empty:
        print(f"[ERROR] Application {application_id} not found")
        return {}

    # Prepare features
    feature_cols = [col for col in df_raw.columns
                   if col not in ['Loan_ID', 'Loan_Status', 'Unnamed: 0']]
    X = app_data[feature_cols]

    # Load model and predict to get probability
    pipeline = joblib.load(MODEL_PATH)
    prediction = pipeline.predict(X)[0]
    probability = pipeline.predict_proba(X)[0, 1]

    print(f"[DATA] Application Details:")
    print(f"   Decision: {'APPROVED' if prediction == 1 else 'DECLINED'}")
    print(f"   Approval Probability: {probability*100:.1f}%")
    print()

    # Initialize interpreter
    print("[INIT] Initializing Mechanistic Interpreter...")
    interpreter = MechanisticInterpreter(MODEL_PATH, feature_cols)
    print(f"   Model loaded: {len(interpreter.trees_df['Tree'].unique())} trees\n")

    # ========================================
    # 1. DECISION PATH ANALYSIS
    # ========================================
    print("1.  DECISION PATH ANALYSIS")
    print("   " + "-"*60)

    paths = interpreter.extract_decision_path(application_id, X, tree_limit=None)

    print(f"   Analyzed: {len(paths)} decision paths (one per tree)")
    print(f"   Average path length: {sum(p.path_length for p in paths)/len(paths):.2f} splits")
    print(f"   Average leaf value: {sum(p.leaf_value for p in paths)/len(paths):.4f}")

    # Find most influential trees
    paths_sorted = sorted(paths, key=lambda p: abs(p.leaf_value), reverse=True)
    print(f"\n   *TREE* Top 5 Most Influential Trees:")
    for i, path in enumerate(paths_sorted[:5], 1):
        direction = "-> APPROVAL" if path.leaf_value > 0 else "-> DECLINE"
        print(f"      {i}. Tree #{path.tree_id}: "
              f"contribution={path.leaf_value:+.4f} {direction}")
        print(f"         Path length: {path.path_length} splits")
        print(f"         Key features: {', '.join(path.split_features[:3])}")

    # ========================================
    # 2. FEATURE INTERACTION DETECTION
    # ========================================
    print(f"\n2.  FEATURE INTERACTION DETECTION")
    print("   " + "-"*60)

    interactions = interpreter.detect_feature_interactions(min_co_occurrence=5)

    print(f"   Detected: {len(interactions)} significant feature interactions")

    # Analyze interaction types
    from collections import Counter
    type_counts = Counter([i.interaction_type for i in interactions])

    print(f"\n   Interaction Types:")
    for itype, count in type_counts.items():
        print(f"      - {itype.capitalize()}: {count} interactions")

    print(f"\n   *LINK* Top 10 Feature Interactions:")
    for i, interaction in enumerate(interactions[:10], 1):
        print(f"      {i}. {interaction.feature_a} × {interaction.feature_b}")
        print(f"         Strength: {interaction.interaction_strength:.3f} | "
              f"Type: {interaction.interaction_type} | "
              f"Co-occurrence: {interaction.co_occurrence_count} trees")

    # ========================================
    # 3. TREE SPECIALIZATION PATTERNS
    # ========================================
    print(f"\n3.  TREE SPECIALIZATION PATTERNS")
    print("   " + "-"*60)

    tree_insights = interpreter.analyze_tree_level_patterns()

    # Find which features trees specialize in
    specialization_counts = Counter([t.dominant_feature for t in tree_insights])

    print(f"   Analyzed: {len(tree_insights)} trees")
    print(f"\n   *TARGET* Feature Specialization Distribution:")
    for feature, count in specialization_counts.most_common(10):
        pct = count / len(tree_insights) * 100
        print(f"      {feature}: {count} trees ({pct:.1f}%)")

    # ========================================
    # 4. ACTIVATION PATTERN ANALYSIS
    # ========================================
    print(f"\n4.  ACTIVATION PATTERN ANALYSIS")
    print("   " + "-"*60)

    # Analyze top features from interactions
    top_features = [interactions[0].feature_a, interactions[0].feature_b]

    for feature in top_features[:3]:
        activation = interpreter.compute_activation_patterns(X, feature)
        print(f"\n   *SPARK* {feature}:")
        print(f"      Used in: {activation['total_trees_using']}/{len(paths)} trees "
              f"({activation['tree_usage_rate']*100:.1f}%)")
        print(f"      Unique thresholds: {activation['unique_thresholds']}")

    # ========================================
    # 5. VISUALIZATIONS
    # ========================================
    print(f"\n5.  GENERATING VISUALIZATIONS")
    print("   " + "-"*60)

    os.makedirs("figures/mechanistic", exist_ok=True)

    # Decision path visualization
    path_viz = f"figures/mechanistic/decision_paths_{application_id}.png"
    interpreter.visualize_decision_path(paths, output_path=path_viz, top_k=30)
    print(f"   OK Decision paths: {path_viz}")

    # Feature interaction visualization
    interaction_viz = "figures/mechanistic/feature_interactions.png"
    interpreter.visualize_feature_interactions(interactions, output_path=interaction_viz, top_k=20)
    print(f"   OK Feature interactions: {interaction_viz}")

    # ========================================
    # 6. EXPORT TO JSON
    # ========================================
    print(f"\n6.  EXPORTING MECHANISTIC INSIGHTS")
    print("   " + "-"*60)

    os.makedirs("data/evaluation", exist_ok=True)
    json_path = f"data/evaluation/mechanistic_{application_id}.json"

    interpreter.export_to_json(
        application_id,
        paths,
        interactions,
        tree_insights,
        output_path=json_path
    )
    print(f"   OK JSON export: {json_path}")

    # ========================================
    # 7. LOAD INTO NEO4J (OPTIONAL)
    # ========================================
    if load_to_neo4j:
        if not NEO4J_AVAILABLE:
            print(f"\n7.  NEO4J LOADING SKIPPED")
            print("   " + "-"*60)
            print(f"   *WARNING* Neo4j module not installed")
            print(f"      Install with: pip install neo4j")
        else:
            print(f"\n7.  LOADING INTO NEO4J KNOWLEDGE GRAPH")
            print("   " + "-"*60)

            try:
                loader = MechanisticKGLoader()
                loader.create_constraints()

                stats = loader.load_from_json(json_path)

                print(f"   OK Decision paths loaded: {stats['decision_paths']}")
                print(f"   OK Feature interactions loaded: {stats['feature_interactions']}")
                print(f"   OK Tree specializations loaded: {stats['tree_specializations']}")

                # Query back enriched explanation
                print(f"\n   [DATA] Mechanistic Explanation Query:")
                explanation = loader.query_mechanistic_explanation(application_id)

                print(f"      Top contributing trees: {len(explanation['top_decision_paths'])}")
                print(f"      Active interactions: {len(explanation['active_feature_interactions'])}")

                loader.close()

            except Exception as e:
                print(f"   *WARNING* Neo4j loading failed: {e}")
                print(f"      (Make sure Neo4j is running)")

    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*70}")
    print(f"[OK] MECHANISTIC ANALYSIS COMPLETE")
    print(f"{'='*70}\n")

    summary = {
        "application_id": application_id,
        "prediction": "APPROVED" if prediction == 1 else "DECLINED",
        "probability": float(probability),
        "mechanistic_insights": {
            "total_trees_analyzed": len(paths),
            "avg_path_length": sum(p.path_length for p in paths)/len(paths),
            "most_influential_tree": paths_sorted[0].tree_id,
            "max_tree_contribution": paths_sorted[0].leaf_value,
            "total_interactions_detected": len(interactions),
            "top_interaction": f"{interactions[0].feature_a} × {interactions[0].feature_b}",
            "interaction_strength": interactions[0].interaction_strength,
            "dominant_specialization": specialization_counts.most_common(1)[0][0]
        },
        "outputs": {
            "json": json_path,
            "decision_paths_viz": path_viz,
            "interactions_viz": interaction_viz
        }
    }

    return summary


def analyze_multiple_applications(n_samples: int = 10):
    """Analyze multiple applications and compare mechanistic patterns"""
    print(f"\n{'='*70}")
    print(f"MECHANISTIC INTERPRETABILITY: COMPARATIVE ANALYSIS")
    print(f"{'='*70}\n")

    # Load data
    RAW_DATA_PATH = "data/raw/df1_loan.csv"
    df = pd.read_csv(RAW_DATA_PATH)

    # Map Loan_Status Y/N to 1/0
    df['Loan_Status'] = df['Loan_Status'].map({"Y": 1, "N": 0})

    # Sample applications
    approved = df[df['Loan_Status'] == 1].sample(n=min(n_samples//2, len(df[df['Loan_Status'] == 1])))
    declined = df[df['Loan_Status'] == 0].sample(n=min(n_samples//2, len(df[df['Loan_Status'] == 0])))

    applications = pd.concat([approved, declined])

    print(f"Analyzing {len(applications)} applications:")
    print(f"  - Approved: {len(approved)}")
    print(f"  - Declined: {len(declined)}\n")

    results = []

    for idx, row in applications.iterrows():
        app_id = row['Loan_ID']
        result = analyze_single_application(app_id, load_to_neo4j=False)
        results.append(result)

    # Comparative analysis
    print(f"\n{'='*70}")
    print(f"COMPARATIVE MECHANISTIC PATTERNS")
    print(f"{'='*70}\n")

    # Compare approved vs declined
    approved_results = [r for r in results if r.get('prediction') == 'APPROVED']
    declined_results = [r for r in results if r.get('prediction') == 'DECLINED']

    if approved_results and declined_results:
        avg_path_approved = sum(r['mechanistic_insights']['avg_path_length']
                               for r in approved_results) / len(approved_results)
        avg_path_declined = sum(r['mechanistic_insights']['avg_path_length']
                               for r in declined_results) / len(declined_results)

        print(f"Average path length:")
        print(f"  Approved: {avg_path_approved:.2f} splits")
        print(f"  Declined: {avg_path_declined:.2f} splits")

        avg_interactions_approved = sum(r['mechanistic_insights']['total_interactions_detected']
                                       for r in approved_results) / len(approved_results)
        avg_interactions_declined = sum(r['mechanistic_insights']['total_interactions_detected']
                                       for r in declined_results) / len(declined_results)

        print(f"\nAverage feature interactions detected:")
        print(f"  Approved: {avg_interactions_approved:.1f}")
        print(f"  Declined: {avg_interactions_declined:.1f}")

    # Save comparative results
    comparative_path = "data/evaluation/mechanistic_comparative.json"
    with open(comparative_path, 'w') as f:
        json.dump({
            "summary": {
                "total_analyzed": len(results),
                "approved": len(approved_results),
                "declined": len(declined_results)
            },
            "results": results
        }, f, indent=2)

    print(f"\nOK Comparative results saved to: {comparative_path}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single application: python mechanistic_interpretability_demo.py LP001006")
        print("  Multiple apps:      python mechanistic_interpretability_demo.py --all [n_samples]")
        sys.exit(1)

    if sys.argv[1] == '--all':
        n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        analyze_multiple_applications(n_samples)
    else:
        application_id = sys.argv[1]
        analyze_single_application(application_id, load_to_neo4j=True)


if __name__ == "__main__":
    main()
