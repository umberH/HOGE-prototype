"""
Test script for Concept Mapping functionality

Demonstrates how business concepts abstract technical features for executive/business audiences.
"""

import sys
import json
from src.explainability.llm_explainer import get_application_explanation_data, call_llm_for_explanation
from src.explainability.concept_mapper import ConceptMapper


def test_concept_mapper_standalone():
    """Test the concept mapper in standalone mode"""
    print("\n" + "="*80)
    print("TEST 1: Standalone Concept Mapper")
    print("="*80 + "\n")

    # Sample SHAP details (simulated loan application)
    sample_shap = [
        {"feature": "ApplicantIncome", "value": 5000, "shap": 0.15, "importance": 0.25, "direction": "positive"},
        {"feature": "CoapplicantIncome", "value": 2000, "shap": 0.08, "importance": 0.12, "direction": "positive"},
        {"feature": "LoanAmount", "value": 150, "shap": -0.05, "importance": 0.18, "direction": "negative"},
        {"feature": "Credit_History", "value": 1.0, "shap": 0.22, "importance": 0.35, "direction": "positive"},
        {"feature": "Property_Area", "value": "Urban", "shap": 0.03, "importance": 0.08, "direction": "positive"},
        {"feature": "Self_Employed", "value": 0, "shap": 0.02, "importance": 0.05, "direction": "positive"},
        {"feature": "Married", "value": 1, "shap": 0.01, "importance": 0.03, "direction": "positive"},
    ]

    # Create mapper
    mapper = ConceptMapper()

    # Map to concepts
    concepts = mapper.map_features_to_concepts(sample_shap)

    print("📊 Feature-to-Concept Mapping:\n")
    print(f"Total Features: {len(sample_shap)} → Total Concepts: {len(concepts)}\n")

    for concept in concepts:
        print(f"\n🧩 {concept.concept_name}")
        print(f"   Category: {concept.concept_category}")
        print(f"   Features: {', '.join(concept.related_features)}")
        print(f"   Aggregated Impact: {concept.aggregated_shap_value:+.3f}")
        print(f"   Direction: {concept.overall_direction}")
        print(f"   Interpretation: {concept.business_interpretation}")

    # Generate narratives
    print("\n" + "-"*80)
    print("Executive Narrative:")
    print("-"*80 + "\n")

    narrative = mapper.generate_concept_narrative(
        concepts,
        audience="executive",
        decision="Approved",
        probability=0.78
    )
    print(narrative)

    print("\n" + "-"*80)
    print("Business Narrative:")
    print("-"*80 + "\n")

    narrative = mapper.generate_concept_narrative(
        concepts,
        audience="business",
        decision="Approved",
        probability=0.78
    )
    print(narrative)


def test_concept_with_real_application(application_id: str = "LP001006"):
    """Test concept mapping with real application from Neo4j"""
    print("\n" + "="*80)
    print(f"TEST 2: Concept Mapping with Real Application ({application_id})")
    print("="*80 + "\n")

    try:
        # Get data from KG
        print("📊 Fetching data from Knowledge Graph...")
        context = get_application_explanation_data(application_id)
        print(f"✅ Retrieved: {context['model_prediction']} ({context['approval_probability']*100:.1f}%)\n")

        # Create concept mapper
        mapper = ConceptMapper()

        # Map features to concepts
        concepts = mapper.map_features_to_concepts(
            context['shap_details'],
            context
        )

        print(f"📈 Mapped {len(context['shap_details'])} features → {len(concepts)} concepts\n")

        # Show concept breakdown
        print("Concept Breakdown:")
        print("-"*80)

        for i, concept in enumerate(concepts, 1):
            print(f"\n{i}. {concept.concept_name} ({concept.concept_category})")
            print(f"   Impact: {concept.aggregated_shap_value:+.3f} | Direction: {concept.overall_direction}")
            print(f"   Confidence: {concept.confidence_level}")
            print(f"   Features ({len(concept.related_features)}): {', '.join(concept.related_features[:5])}")
            print(f"   Interpretation: {concept.business_interpretation}")

        # Export concept map
        print("\n" + "-"*80)
        print("Exporting concept map to JSON...")

        output_file = f"data/evaluation/concept_map_{application_id}.json"
        mapper.export_concept_map(concepts, output_file)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_concept_in_llm_explanation(application_id: str = "LP001006"):
    """Test how concepts are integrated into LLM explanations"""
    print("\n" + "="*80)
    print(f"TEST 3: Concept-Enhanced LLM Explanations ({application_id})")
    print("="*80 + "\n")

    try:
        # Get data
        print("📊 Fetching data...")
        context = get_application_explanation_data(application_id)

        # Test with different audiences
        audiences = [
            ("technical", False),      # Technical without concepts
            ("executive", True),       # Executive with concepts (auto-enabled)
            ("business", True),        # Business with concepts (auto-enabled)
            ("non_technical", False),  # Non-technical without concepts
        ]

        for audience, will_use_concepts in audiences:
            print(f"\n{'-'*80}")
            print(f"🎯 Audience: {audience.upper()}")
            print(f"   Expected to use concepts: {'Yes' if will_use_concepts else 'No'}")
            print(f"{'-'*80}\n")

            # Generate explanation
            explanation = call_llm_for_explanation(context, audience=audience)

            # Check if concepts were used
            used_concepts = explanation.get("used_concepts", False)
            print(f"✓ Actually used concepts: {'Yes' if used_concepts else 'No'}")

            if used_concepts:
                num_concepts = len(explanation.get("business_concepts", []))
                print(f"✓ Number of concepts: {num_concepts}")

                # Show first concept as example
                if num_concepts > 0:
                    first_concept = explanation["business_concepts"][0]
                    print(f"\n   Example concept:")
                    print(f"   - Name: {first_concept['concept_name']}")
                    print(f"   - Impact: {first_concept['aggregated_shap_value']:+.3f}")
                    print(f"   - Interpretation: {first_concept['business_interpretation']}")

            # Show summary
            print(f"\n📋 Summary: {explanation['summary'][:150]}...")

            # Show first paragraph of narrative
            narrative = explanation.get('narrative', '')
            first_para = narrative.split('\n\n')[0] if narrative else ''
            print(f"\n📖 Narrative (first paragraph):\n{first_para[:200]}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def compare_feature_vs_concept_explanations(application_id: str = "LP001006"):
    """Compare explanations with and without concept mapping"""
    print("\n" + "="*80)
    print(f"TEST 4: Feature vs. Concept Explanation Comparison ({application_id})")
    print("="*80 + "\n")

    try:
        context = get_application_explanation_data(application_id)

        # Executive explanation WITHOUT concepts (force disable)
        print("1️⃣ EXECUTIVE - Feature-Level Explanation:")
        print("-"*80)

        # Manually disable concepts by removing from context
        context_no_concepts = context.copy()
        if "business_concepts" in context_no_concepts:
            del context_no_concepts["business_concepts"]

        exp_features = call_llm_for_explanation(context_no_concepts, audience="executive", use_concepts=False)
        print(f"Summary: {exp_features['summary']}\n")
        print(f"Narrative:\n{exp_features['narrative'][:400]}...\n")

        # Executive explanation WITH concepts (auto-enable)
        print("\n2️⃣ EXECUTIVE - Concept-Level Explanation:")
        print("-"*80)

        exp_concepts = call_llm_for_explanation(context, audience="executive", use_concepts=True)
        print(f"Summary: {exp_concepts['summary']}\n")
        print(f"Narrative:\n{exp_concepts['narrative'][:400]}...\n")

        # Comparison stats
        print("\n" + "="*80)
        print("COMPARISON:")
        print("="*80)
        print(f"Feature-level narrative length: {len(exp_features['narrative'].split())} words")
        print(f"Concept-level narrative length: {len(exp_concepts['narrative'].split())} words")
        print(f"\nConcept mode enabled: {exp_concepts.get('used_concepts', False)}")
        print(f"Number of concepts: {len(exp_concepts.get('business_concepts', []))}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test runner"""
    print("\n" + "="*80)
    print("CONCEPT MAPPING TEST SUITE")
    print("="*80)

    # Get application ID
    app_id = sys.argv[1] if len(sys.argv) > 1 else "LP001006"

    print("\nAvailable Tests:")
    print("1. Standalone concept mapper (simulated data)")
    print("2. Concept mapping with real application")
    print("3. Concept-enhanced LLM explanations")
    print("4. Feature vs. Concept comparison")
    print("5. Run all tests")

    choice = input("\nSelect test (1-5): ").strip()

    if choice == "1":
        test_concept_mapper_standalone()
    elif choice == "2":
        test_concept_with_real_application(app_id)
    elif choice == "3":
        test_concept_in_llm_explanation(app_id)
    elif choice == "4":
        compare_feature_vs_concept_explanations(app_id)
    elif choice == "5":
        test_concept_mapper_standalone()
        test_concept_with_real_application(app_id)
        test_concept_in_llm_explanation(app_id)
        compare_feature_vs_concept_explanations(app_id)
    else:
        print("Invalid choice. Running all tests...")
        test_concept_mapper_standalone()
        test_concept_with_real_application(app_id)
        test_concept_in_llm_explanation(app_id)
        compare_feature_vs_concept_explanations(app_id)

    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
