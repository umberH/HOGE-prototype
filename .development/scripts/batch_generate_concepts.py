"""
Batch Concept Generation Script
================================

Generates and stores business concept mappings for all loan applications.

This script should be run:
- After model retraining
- Nightly as a batch job
- When adding new applications to the system

Output:
- JSON files in data/evaluation/concepts/
- (Future) Concept nodes in Neo4j knowledge graph

Usage:
    python scripts/batch_generate_concepts.py
    python scripts/batch_generate_concepts.py --limit 100
    python scripts/batch_generate_concepts.py --output-dir custom/path
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.explainability.llm_explainer import get_application_explanation_data
from src.explainability.concept_mapper import ConceptMapper


def batch_generate_concepts(
    application_ids=None,
    output_dir="data/evaluation/concepts",
    limit=None,
    skip_existing=True
):
    """
    Generate concept mappings for multiple applications

    Args:
        application_ids: List of application IDs to process (None = all from CSV)
        output_dir: Directory to save concept JSON files
        limit: Maximum number of applications to process
        skip_existing: Skip applications that already have concept files

    Returns:
        Dictionary with generation statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize mapper
    mapper = ConceptMapper()

    # Get application IDs
    if application_ids is None:
        # Load from processed data
        try:
            df = pd.read_csv("data/processed/scored_applications_xgb.csv")
            application_ids = df['Loan_ID'].tolist()
        except FileNotFoundError:
            print("❌ Error: data/processed/scored_applications_xgb.csv not found")
            return None

    # Apply limit if specified
    if limit:
        application_ids = application_ids[:limit]

    # Statistics
    stats = {
        "total": len(application_ids),
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "errors": []
    }

    print(f"\n{'='*80}")
    print(f"BATCH CONCEPT GENERATION")
    print(f"{'='*80}\n")
    print(f"Processing {stats['total']} applications...")
    print(f"Output directory: {output_dir}\n")

    # Process each application
    for app_id in tqdm(application_ids, desc="Generating concepts"):
        output_file = os.path.join(output_dir, f"concept_map_{app_id}.json")

        # Skip if already exists
        if skip_existing and os.path.exists(output_file):
            stats['skipped'] += 1
            continue

        try:
            # Get explanation context
            context = get_application_explanation_data(app_id)

            # Map features to concepts
            concepts = mapper.map_features_to_concepts(
                context.get('shap_details', []),
                context
            )

            # Export concept map
            export_data = mapper.export_concept_map(concepts, output_path=None)

            # Add application metadata
            export_data['application_id'] = app_id
            export_data['model_prediction'] = context.get('model_prediction')
            export_data['approval_probability'] = context.get('approval_probability')

            # Save to file
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            stats['success'] += 1

        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append({
                'application_id': app_id,
                'error': str(e)
            })

    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH GENERATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"✅ Successfully processed: {stats['success']}/{stats['total']}")
    print(f"⏭️  Skipped (already exist): {stats['skipped']}")
    print(f"❌ Failed: {stats['failed']}")

    if stats['failed'] > 0:
        print(f"\n⚠️  Errors encountered:")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error['application_id']}: {error['error']}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")

    # Save summary report
    summary_file = os.path.join(output_dir, "_batch_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n📊 Summary report saved to: {summary_file}\n")

    return stats


def load_concept_map(application_id, concepts_dir="data/evaluation/concepts"):
    """
    Load a previously generated concept map

    Args:
        application_id: Application ID
        concepts_dir: Directory containing concept files

    Returns:
        Concept map dictionary or None if not found
    """
    concept_file = os.path.join(concepts_dir, f"concept_map_{application_id}.json")

    if not os.path.exists(concept_file):
        return None

    with open(concept_file, 'r') as f:
        return json.load(f)


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Generate business concept mappings for loan applications"
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help="Maximum number of applications to process"
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default="data/evaluation/concepts",
        help="Output directory for concept JSON files"
    )

    parser.add_argument(
        '--app-ids',
        type=str,
        nargs='+',
        default=None,
        help="Specific application IDs to process (space-separated)"
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help="Regenerate even if concept files already exist"
    )

    args = parser.parse_args()

    # Run batch generation
    stats = batch_generate_concepts(
        application_ids=args.app_ids,
        output_dir=args.output_dir,
        limit=args.limit,
        skip_existing=not args.force
    )

    # Exit with appropriate code
    if stats and stats['failed'] == 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
