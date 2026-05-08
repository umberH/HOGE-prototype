"""
Test script for audience-based explanations

This script demonstrates how to generate explanations for different audiences.
"""

import sys
from src.explainability.llm_explainer import get_application_explanation_data, call_llm_for_explanation

def test_audience_explanations(application_id: str = "LP001006"):
    """
    Test explanation generation for all audience types

    Args:
        application_id: Application ID to analyze
    """
    print(f"\n{'='*80}")
    print(f"Testing Audience-Based Explanations for Application: {application_id}")
    print(f"{'='*80}\n")

    # Get context data once (same for all audiences)
    print("📊 Fetching application data from Knowledge Graph...")
    try:
        context = get_application_explanation_data(application_id)
        print(f"✅ Data retrieved: {context['model_prediction']} with {context['approval_probability']*100:.1f}% probability\n")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return

    # Test all audience types
    audiences = ["technical", "non_technical", "executive", "business"]

    for audience in audiences:
        print(f"\n{'-'*80}")
        print(f"🎯 Generating explanation for: {audience.upper().replace('_', ' ')}")
        print(f"{'-'*80}\n")

        try:
            explanation = call_llm_for_explanation(context, audience=audience)

            # Display summary
            print(f"📋 Summary:\n{explanation.get('summary', 'N/A')}\n")

            # Display narrative (truncated)
            narrative = explanation.get('narrative', 'N/A')
            if len(narrative) > 500:
                print(f"📖 Narrative (first 500 chars):\n{narrative[:500]}...\n")
            else:
                print(f"📖 Narrative:\n{narrative}\n")

            # Display metadata
            print(f"ℹ️ Metadata:")
            print(f"  - Audience: {explanation.get('audience')}")
            print(f"  - Positive Drivers: {len(explanation.get('positive_drivers', []))}")
            print(f"  - Negative Drivers: {len(explanation.get('negative_drivers', []))}")
            print(f"  - Policy Violations: {len(explanation.get('policy_violations', []))}")
            print(f"  - Features Used: {len(explanation.get('features_used', []))}")

        except Exception as e:
            print(f"❌ Error generating explanation: {e}")

    print(f"\n{'='*80}")
    print("✅ Test Complete!")
    print(f"{'='*80}\n")


def compare_audiences(application_id: str = "LP001006"):
    """
    Generate side-by-side comparison of narratives for all audiences

    Args:
        application_id: Application ID to analyze
    """
    print(f"\n{'='*80}")
    print(f"Audience Comparison for Application: {application_id}")
    print(f"{'='*80}\n")

    try:
        context = get_application_explanation_data(application_id)
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return

    audiences = ["technical", "non_technical", "executive", "business"]
    explanations = {}

    for audience in audiences:
        try:
            explanations[audience] = call_llm_for_explanation(context, audience=audience)
        except Exception as e:
            print(f"❌ Error for {audience}: {e}")

    # Compare summaries
    print("\n📋 SUMMARIES COMPARISON:\n")
    for audience, expl in explanations.items():
        print(f"[{audience.upper().replace('_', ' ')}]")
        print(f"{expl.get('summary', 'N/A')}\n")

    # Compare narrative lengths
    print("\n📊 NARRATIVE STATISTICS:\n")
    for audience, expl in explanations.items():
        narrative = expl.get('narrative', '')
        word_count = len(narrative.split())
        print(f"{audience.upper().replace('_', ' '):20s} - {word_count:4d} words")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Get application ID from command line or use default
    app_id = sys.argv[1] if len(sys.argv) > 1 else "LP001006"

    # Choose test mode
    print("\nSelect test mode:")
    print("1. Full test (generate all audiences)")
    print("2. Quick comparison (summaries only)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "2":
        compare_audiences(app_id)
    else:
        test_audience_explanations(app_id)
