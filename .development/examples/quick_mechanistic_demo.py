"""
Quick Mechanistic Interpretability Demo

A simplified script to demonstrate mechanistic interpretability
without requiring full Neo4j setup.

Usage:
    python examples/quick_mechanistic_demo.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from explainability.mechanistic_interpreter import MechanisticInterpreter


def main():
    print("\n" + "="*70)
    print("MECHANISTIC INTERPRETABILITY - QUICK DEMO")
    print("="*70 + "\n")

    # Configuration
    MODEL_PATH = "models/loan_xgb_monotonic.joblib"
    RAW_DATA_PATH = "data/raw/df1_loan.csv"

    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("   Please run: python src/models/train_model.py")
        return

    if not os.path.exists(RAW_DATA_PATH):
        print(f"[ERROR] Data not found: {RAW_DATA_PATH}")
        print("   Please ensure raw data exists")
        return

    # Load data
    print("[LOAD] Loading data...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Clean data (same as train_model.py)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["Total_Income"] = (df["Total_Income"].astype(str)
                          .str.replace("$", "", regex=False)
                          .str.replace(",", "", regex=False))
    df["Total_Income"] = pd.to_numeric(df["Total_Income"], errors="coerce")

    # Create DTI
    df["DTI"] = df["LoanAmount"] / (df["ApplicantIncome"] + df["CoapplicantIncome"] + 1)
    df["DTI"] = pd.to_numeric(df["DTI"], errors="coerce")

    # Map Loan_Status
    df['Loan_Status'] = df['Loan_Status'].map({"Y": 1, "N": 0})

    print(f"   Loaded {len(df)} applications\n")

    # Load model
    pipeline = joblib.load(MODEL_PATH)

    # Select an interesting application (one that was declined)
    declined = df[df['Loan_Status'] == 0].head(1)
    if declined.empty:
        print("   No declined applications found, using first application")
        declined = df.head(1)

    application_id = declined.iloc[0]['Loan_ID']

    # Prepare features
    feature_cols = [col for col in df.columns
                   if col not in ['Loan_ID', 'Loan_Status', 'Unnamed: 0']]
    X = declined[feature_cols]

    # Get prediction
    prediction = pipeline.predict(X)[0]
    probability = pipeline.predict_proba(X)[0, 1]

    print(f"[SEARCH] Analyzing: {application_id}")
    print(f"   Decision: {'APPROVED' if prediction == 1 else 'DECLINED'}")
    print(f"   Approval Probability: {probability*100:.1f}%\n")

    # Initialize interpreter
    print("[INIT] Initializing Mechanistic Interpreter...")
    interpreter = MechanisticInterpreter(MODEL_PATH, feature_cols)
    n_trees = len(interpreter.trees_df['Tree'].unique())
    print(f"   OK Model loaded: {n_trees} trees in ensemble\n")

    # ========================================
    # 1. DECISION PATHS
    # ========================================
    print("="*70)
    print("1. DECISION PATH ANALYSIS")
    print("="*70 + "\n")

    paths = interpreter.extract_decision_path(application_id, X)

    avg_length = np.mean([p.path_length for p in paths])
    avg_leaf = np.mean([p.leaf_value for p in paths])

    print(f"OK Analyzed {len(paths)} decision paths")
    print(f"  Average path length: {avg_length:.2f} splits")
    print(f"  Average leaf contribution: {avg_leaf:+.4f}\n")

    # Find most influential trees
    paths_sorted = sorted(paths, key=lambda p: abs(p.leaf_value), reverse=True)

    print("*TREE* Top 10 Most Influential Trees:\n")
    print(f"{'Tree':<8} {'Contribution':<15} {'Direction':<12} {'Path Length':<12} {'Key Features'}")
    print("-"*70)

    for i, path in enumerate(paths_sorted[:10], 1):
        direction = "-> APPROVAL" if path.leaf_value > 0 else "-> DECLINE"
        key_features = ', '.join(path.split_features[:3]) if path.split_features else 'None'
        print(f"#{path.tree_id:<7} {path.leaf_value:+.4f}          {direction:<12} "
              f"{path.path_length:<12} {key_features}")

    # ========================================
    # 2. FEATURE INTERACTIONS
    # ========================================
    print("\n" + "="*70)
    print("2. FEATURE INTERACTION DETECTION")
    print("="*70 + "\n")

    print("[SEARCH] Detecting feature interactions...")
    interactions = interpreter.detect_feature_interactions(min_co_occurrence=5)
    print(f"OK Found {len(interactions)} significant interactions\n")

    # Count interaction types
    from collections import Counter
    type_counts = Counter([i.interaction_type for i in interactions])

    print("[DATA] Interaction Type Distribution:")
    for itype, count in type_counts.items():
        pct = count / len(interactions) * 100
        print(f"   {itype.capitalize():<12} {count:>3} ({pct:>5.1f}%)")

    print("\n*LINK* Top 15 Feature Interactions:\n")
    print(f"{'Rank':<6} {'Feature A':<20} {'×':<3} {'Feature B':<20} {'Strength':<10} {'Type':<12} {'Trees'}")
    print("-"*90)

    for i, interaction in enumerate(interactions[:15], 1):
        print(f"{i:<6} {interaction.feature_a:<20} {'×':<3} {interaction.feature_b:<20} "
              f"{interaction.interaction_strength:.3f}      {interaction.interaction_type:<12} "
              f"{interaction.co_occurrence_count}")

    # ========================================
    # 3. TREE SPECIALIZATION
    # ========================================
    print("\n" + "="*70)
    print("3. TREE SPECIALIZATION PATTERNS")
    print("="*70 + "\n")

    print("[SEARCH] Analyzing tree specializations...")
    tree_insights = interpreter.analyze_tree_level_patterns()
    print(f"OK Analyzed {len(tree_insights)} trees\n")

    # Feature specialization distribution
    specialization_counts = Counter([t.dominant_feature for t in tree_insights])

    print("*TARGET* Feature Specialization Distribution:\n")
    print(f"{'Feature':<25} {'# Trees':<10} {'Percentage'}")
    print("-"*50)

    for feature, count in specialization_counts.most_common(15):
        pct = count / len(tree_insights) * 100
        bar = '#' * int(pct / 2)
        print(f"{feature:<25} {count:<10} {pct:>5.1f}% {bar}")

    # ========================================
    # 4. ACTIVATION PATTERNS
    # ========================================
    print("\n" + "="*70)
    print("4. ACTIVATION PATTERN ANALYSIS")
    print("="*70 + "\n")

    # Analyze top 3 features from interactions
    top_features = list(set([
        interactions[0].feature_a, interactions[0].feature_b,
        interactions[1].feature_a if len(interactions) > 1 else None,
    ]))
    top_features = [f for f in top_features if f is not None][:3]

    print("*SPARK* Feature Activation Patterns:\n")

    for feature in top_features:
        activation = interpreter.compute_activation_patterns(X, feature)
        usage_pct = activation['tree_usage_rate'] * 100

        print(f"{feature}:")
        print(f"  Used in: {activation['total_trees_using']}/{n_trees} trees ({usage_pct:.1f}%)")
        print(f"  Unique split thresholds: {activation['unique_thresholds']}")

        # Show threshold distribution
        if activation['activations']:
            all_thresholds = [t for act in activation['activations']
                            for t in act['thresholds']]
            if all_thresholds:
                print(f"  Threshold range: [{min(all_thresholds):.3f}, {max(all_thresholds):.3f}]")
        print()

    # ========================================
    # SUMMARY
    # ========================================
    print("="*70)
    print("MECHANISTIC INSIGHTS SUMMARY")
    print("="*70 + "\n")

    print(f"Application: {application_id}")
    print(f"Decision: {'APPROVED' if prediction == 1 else 'DECLINED'} ({probability*100:.1f}% approval probability)\n")

    print("Key Findings:")
    print(f"  • Most influential tree: #{paths_sorted[0].tree_id} "
          f"(contribution: {paths_sorted[0].leaf_value:+.4f})")
    print(f"  • Strongest interaction: {interactions[0].feature_a} × {interactions[0].feature_b} "
          f"(strength: {interactions[0].interaction_strength:.3f})")
    print(f"  • Most specialized feature: {specialization_counts.most_common(1)[0][0]} "
          f"(used in {specialization_counts.most_common(1)[0][1]} trees)")
    print(f"  • Average decision path length: {avg_length:.2f} splits")

    # ========================================
    # VISUALIZATION
    # ========================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")

    os.makedirs("figures/mechanistic", exist_ok=True)

    print("[DATA] Creating visualizations...")

    # Decision paths
    path_viz = f"figures/mechanistic/decision_paths_{application_id}.png"
    interpreter.visualize_decision_path(paths, output_path=path_viz, top_k=30)
    print(f"  OK Decision paths: {path_viz}")

    # Feature interactions
    interaction_viz = "figures/mechanistic/feature_interactions.png"
    interpreter.visualize_feature_interactions(interactions, output_path=interaction_viz, top_k=20)
    print(f"  OK Feature interactions: {interaction_viz}")

    # ========================================
    # EXPORT
    # ========================================
    print("\n" + "="*70)
    print("EXPORTING RESULTS")
    print("="*70 + "\n")

    os.makedirs("data/evaluation", exist_ok=True)
    json_path = f"data/evaluation/mechanistic_{application_id}.json"

    interpreter.export_to_json(
        application_id, paths, interactions, tree_insights,
        output_path=json_path
    )
    print(f"OK JSON export: {json_path}")

    print("\n" + "="*70)
    print("[OK] MECHANISTIC ANALYSIS COMPLETE")
    print("="*70 + "\n")

    print("Next steps:")
    print("  1. View visualizations in figures/mechanistic/")
    print("  2. Review JSON export in data/evaluation/")
    print("  3. Load into Neo4j:")
    print(f"     python src/knowledge_graph/neo_mechanistic_loader.py {json_path}")
    print()


if __name__ == "__main__":
    main()
