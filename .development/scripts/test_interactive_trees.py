"""
Test Script for Interactive Tree Visualizations

Demonstrates the new Plotly-based interactive visualizations for:
1. Decision paths across trees
2. Feature interactions
3. Tree ensemble analysis
"""

import sys
import os
from src.explainability.mechanistic_interpreter import MechanisticInterpreter
import pandas as pd
import numpy as np

def test_interactive_visualizations(application_id: str = "LP001006"):
    """
    Generate interactive tree visualizations for a given application

    Args:
        application_id: Loan application ID to analyze
    """
    print(f"\n{'='*80}")
    print(f"Testing Interactive Tree Visualizations for {application_id}")
    print(f"{'='*80}\n")

    # Configuration
    MODEL_PATH = "models/loan_xgb_monotonic.joblib"
    DATA_PATH = "data/processed/scored_applications_xgb.csv"

    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Filter to application
    app_data = df[df['Loan_ID'] == application_id]
    if app_data.empty:
        print(f"❌ Application {application_id} not found")
        print("\nAvailable applications:")
        print(df['Loan_ID'].head(10).tolist())
        return

    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['Loan_ID', 'Loan_Status', 'Probability']]
    X = app_data[feature_cols]

    print(f"✓ Found application: {application_id}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Prediction: {app_data['Loan_Status'].values[0]}")
    print(f"  Probability: {app_data['Probability'].values[0]:.4f}")

    # Initialize interpreter
    print("\nInitializing mechanistic interpreter...")
    interpreter = MechanisticInterpreter(MODEL_PATH, feature_cols)
    print("✓ Model loaded")

    # 1. Extract decision paths
    print("\n1️⃣  Extracting decision paths...")
    paths = interpreter.extract_decision_path(application_id, X, tree_limit=50)
    print(f"   ✓ Analyzed {len(paths)} trees")
    print(f"   ✓ Average path length: {np.mean([p.path_length for p in paths]):.2f} nodes")
    print(f"   ✓ Average leaf value: {np.mean([p.leaf_value for p in paths]):.4f}")

    # 2. Detect feature interactions
    print("\n2️⃣  Detecting feature interactions...")
    interactions = interpreter.detect_feature_interactions(min_co_occurrence=5)
    print(f"   ✓ Found {len(interactions)} significant interactions")

    if interactions:
        print(f"\n   Top 3 interactions:")
        for i, interaction in enumerate(interactions[:3], 1):
            print(f"   {i}. {interaction.feature_a} × {interaction.feature_b}")
            print(f"      • Strength: {interaction.interaction_strength:.3f}")
            print(f"      • Type: {interaction.interaction_type}")
            print(f"      • Co-occurrence: {interaction.co_occurrence_count} times")

    # 3. Analyze tree-level patterns
    print("\n3️⃣  Analyzing tree-level patterns...")
    tree_insights = interpreter.analyze_tree_level_patterns()

    from collections import Counter
    dominant_features = Counter([t.dominant_feature for t in tree_insights])
    print(f"   ✓ Analyzed {len(tree_insights)} trees")
    print(f"\n   Most specialized features across trees:")
    for feat, count in dominant_features.most_common(5):
        print(f"   • {feat}: dominates {count} trees ({count/len(tree_insights)*100:.1f}%)")

    # 4. Generate Interactive Visualizations
    print("\n4️⃣  Generating interactive Plotly visualizations...")

    output_files = interpreter.create_interactive_visualizations(
        application_id,
        paths,
        interactions,
        tree_insights,
        output_dir="figures/mechanistic_interactive"
    )

    if output_files:
        print(f"   ✓ Created {len(output_files)} interactive visualizations:\n")
        for viz_type, filepath in output_files.items():
            filesize = os.path.getsize(filepath) / 1024  # KB
            print(f"   📊 {viz_type}:")
            print(f"      {filepath}")
            print(f"      Size: {filesize:.1f} KB")
            print()

        print("\n   🌐 Open these HTML files in your browser for interactive exploration!")
        print("\n   Features:")
        print("   • Hover for detailed information")
        print("   • Zoom and pan")
        print("   • Click legend items to toggle visibility")
        print("   • Export as static images")
    else:
        print("   ⚠️  No visualizations created (check if Plotly is installed)")

    # 5. Export JSON data
    print("\n5️⃣  Exporting mechanistic insights to JSON...")
    os.makedirs("data/evaluation", exist_ok=True)
    json_path = f"data/evaluation/mechanistic_{application_id}.json"

    interpreter.export_to_json(
        application_id,
        paths,
        interactions,
        tree_insights,
        output_path=json_path
    )

    filesize = os.path.getsize(json_path) / 1024
    print(f"   ✓ Saved: {json_path} ({filesize:.1f} KB)")

    # 6. Generate narrative explanation
    print("\n6️⃣  Generating narrative explanation...")
    narrative = interpreter.generate_narrative_explanation(
        application_id,
        paths,
        interactions,
        tree_insights,
        audience="technical"
    )

    print("\n" + "="*80)
    print(narrative)
    print("="*80)

    # Summary
    print(f"\n✅ COMPLETE! Interactive visualizations ready for {application_id}")
    print(f"\nNext steps:")
    print(f"1. Open figures/mechanistic_interactive/ in your browser")
    print(f"2. Explore the interactive charts")
    print(f"3. Review mechanistic insights in {json_path}")


def compare_multiple_applications(app_ids: list):
    """
    Compare tree visualizations for multiple applications

    Args:
        app_ids: List of application IDs to compare
    """
    print(f"\n{'='*80}")
    print(f"Comparing {len(app_ids)} Applications")
    print(f"{'='*80}\n")

    for app_id in app_ids:
        test_interactive_visualizations(app_id)
        print("\n" + "-"*80 + "\n")

    print("\n✅ All applications processed!")
    print(f"\nCompare visualizations in figures/mechanistic_interactive/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Multiple applications specified
        app_ids = sys.argv[1:]
        if len(app_ids) > 1:
            compare_multiple_applications(app_ids)
        else:
            test_interactive_visualizations(app_ids[0])
    else:
        # Default test applications
        print("\nNo application ID specified. Testing with default applications...")
        print("Usage: python test_interactive_trees.py LP001006 [LP001003 ...]")
        print("\nUsing default: LP001006")
        test_interactive_visualizations("LP001006")
