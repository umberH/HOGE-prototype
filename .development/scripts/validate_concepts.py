"""
Concept Validation Script
==========================

Validates concept mappings and detects potential issues:
- Unmapped features (features not assigned to any concept)
- Low coverage concepts (concepts with very few features)
- Concept quality metrics
- Feature distribution across concepts

This script should be run:
- After modifying concept definitions
- When adding new features to the model
- As part of model validation pipeline

Usage:
    python scripts/validate_concepts.py
    python scripts/validate_concepts.py --detailed
    python scripts/validate_concepts.py --export-report
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.explainability.concept_mapper import ConceptMapper


def validate_concept_mappings(detailed=False):
    """
    Validate feature-to-concept mappings

    Args:
        detailed: Show detailed breakdown per concept

    Returns:
        Validation report dictionary
    """
    mapper = ConceptMapper()

    # Get all features from model
    try:
        df = pd.read_csv("data/processed/scored_applications_xgb.csv")
        all_features = [col for col in df.columns if col not in ['Loan_ID', 'Loan_Status', 'Probability']]
    except FileNotFoundError:
        print("❌ Error: Could not load processed data to get feature list")
        return None

    # Get mapped features
    mapped_features = set(mapper.feature_to_concept_map.keys())
    all_features_set = set(all_features)

    # Find unmapped features
    unmapped_features = all_features_set - mapped_features

    # Analyze concept distribution
    concept_distribution = Counter(mapper.feature_to_concept_map.values())

    # Calculate statistics
    total_features = len(all_features)
    total_mapped = len(mapped_features)
    coverage_pct = (total_mapped / total_features * 100) if total_features > 0 else 0

    # Build report
    report = {
        "summary": {
            "total_features": total_features,
            "mapped_features": total_mapped,
            "unmapped_features": len(unmapped_features),
            "coverage_percentage": coverage_pct,
            "total_concepts": len(concept_distribution),
            "avg_features_per_concept": total_mapped / len(concept_distribution) if concept_distribution else 0
        },
        "unmapped_features": list(unmapped_features),
        "concept_distribution": dict(concept_distribution),
        "concept_details": {},
        "warnings": [],
        "recommendations": []
    }

    # Detailed concept analysis
    if detailed:
        concept_features = defaultdict(list)
        for feature, concept in mapper.feature_to_concept_map.items():
            concept_features[concept].append(feature)

        for concept, features in concept_features.items():
            concept_def = mapper.get_concept_definition(concept)
            report["concept_details"][concept] = {
                "feature_count": len(features),
                "features": features,
                "category": concept_def.get("category", "unknown"),
                "description": concept_def.get("description", "N/A")
            }

    # Generate warnings
    if coverage_pct < 100:
        report["warnings"].append(
            f"⚠️  Only {coverage_pct:.1f}% of features are mapped to concepts. "
            f"{len(unmapped_features)} features are unmapped."
        )

    # Check for low-coverage concepts
    for concept, count in concept_distribution.items():
        if count < 2:
            report["warnings"].append(
                f"⚠️  Concept '{concept}' has only {count} feature(s). "
                "Consider combining with related concepts."
            )

    # Check for undefined concepts
    all_concept_defs = set(mapper.concept_definitions.keys())
    used_concepts = set(concept_distribution.keys())
    undefined_concepts = used_concepts - all_concept_defs

    if undefined_concepts:
        report["warnings"].append(
            f"⚠️  The following concepts are used but not defined: {undefined_concepts}"
        )

    # Generate recommendations
    if unmapped_features:
        report["recommendations"].append(
            f"➕ Add {len(unmapped_features)} unmapped features to concept_mapper.py:\n" +
            "\n".join([f"    '{f}': '[ConceptName]'," for f in list(unmapped_features)[:5]]) +
            ("\n    ..." if len(unmapped_features) > 5 else "")
        )

    if coverage_pct == 100:
        report["recommendations"].append("✅ All features are mapped! No action needed.")

    # Quality score
    quality_score = coverage_pct

    if report["warnings"]:
        quality_score -= len(report["warnings"]) * 5

    report["quality_score"] = max(0, min(100, quality_score))

    return report


def print_validation_report(report):
    """
    Print validation report in human-readable format

    Args:
        report: Validation report dictionary
    """
    print(f"\n{'='*80}")
    print(f"CONCEPT MAPPING VALIDATION REPORT")
    print(f"{'='*80}\n")

    # Summary
    summary = report["summary"]
    print(f"📊 Summary:")
    print(f"  Total Features: {summary['total_features']}")
    print(f"  Mapped Features: {summary['mapped_features']}")
    print(f"  Unmapped Features: {summary['unmapped_features']}")
    print(f"  Coverage: {summary['coverage_percentage']:.1f}%")
    print(f"  Total Concepts: {summary['total_concepts']}")
    print(f"  Avg Features per Concept: {summary['avg_features_per_concept']:.1f}")

    # Quality Score
    print(f"\n🎯 Quality Score: {report['quality_score']:.0f}/100")

    if report['quality_score'] >= 90:
        print("  ✅ Excellent concept coverage!")
    elif report['quality_score'] >= 70:
        print("  🟡 Good coverage, minor improvements needed")
    else:
        print("  🔴 Poor coverage, action required")

    # Warnings
    if report["warnings"]:
        print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"  {warning}")

    # Recommendations
    if report["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  {rec}")

    # Concept Distribution
    print(f"\n📈 Concept Distribution:")
    for concept, count in sorted(report["concept_distribution"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * min(count, 50)
        print(f"  {concept:25s} [{count:2d}] {bar}")

    # Unmapped Features
    if report["unmapped_features"]:
        print(f"\n🔍 Unmapped Features ({len(report['unmapped_features'])}):")
        for feature in sorted(report["unmapped_features"])[:20]:
            print(f"  - {feature}")
        if len(report["unmapped_features"]) > 20:
            print(f"  ... and {len(report['unmapped_features']) - 20} more")

    # Concept Details (if available)
    if report["concept_details"]:
        print(f"\n📚 Concept Details:")
        for concept, details in sorted(report["concept_details"].items()):
            print(f"\n  {concept} ({details['category']}):")
            print(f"    Description: {details['description']}")
            print(f"    Features ({details['feature_count']}): {', '.join(details['features'])}")

    print(f"\n{'='*80}\n")


def export_validation_report(report, output_file="data/evaluation/concept_validation_report.json"):
    """
    Export validation report to JSON file

    Args:
        report: Validation report dictionary
        output_file: Path to output JSON file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 Validation report exported to: {output_file}")


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Validate business concept mappings"
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help="Show detailed breakdown per concept"
    )

    parser.add_argument(
        '--export-report',
        type=str,
        default=None,
        help="Export report to JSON file (default: data/evaluation/concept_validation_report.json)"
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Only show warnings and errors"
    )

    args = parser.parse_args()

    # Run validation
    report = validate_concept_mappings(detailed=args.detailed)

    if report is None:
        sys.exit(1)

    # Print report (unless quiet mode with no warnings)
    if not (args.quiet and not report["warnings"]):
        print_validation_report(report)

    # Export if requested
    if args.export_report is not None or args.export_report == "":
        output_file = args.export_report if args.export_report else "data/evaluation/concept_validation_report.json"
        export_validation_report(report, output_file)

    # Exit with appropriate code
    if report["quality_score"] >= 70:
        sys.exit(0)
    else:
        print("\n❌ Validation failed: Quality score below 70%")
        sys.exit(1)


if __name__ == "__main__":
    main()
