"""
Generate Cached Explanations
=============================
Pre-generate LLM explanations for all applications and cache them locally.
This avoids runtime LLM API calls in Streamlit Cloud deployment.

Usage:
    python .development/scripts/generate_cached_explanations.py [options]

Options:
    --app-ids LP001003,LP001006  Generate for specific IDs (comma-separated)
    --audiences technical,business  Generate for specific audiences (default: all)
    --concepts  Only generate with concepts enabled
    --no-concepts  Only generate without concepts
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.src.explainability.llm_explainer import call_llm_for_explanation
from backend.src.explainability.explanation_cache import ExplanationCache
from backend.src.knowledge_graph.neo_loader import get_application_ids_from_neo4j
import os
from dotenv import load_dotenv

load_dotenv()


def get_application_explanation_data(application_id: str) -> dict:
    """Import the actual function from your codebase"""
    # This is a placeholder - you'll need to import your actual function
    # For now, returning a minimal context
    from neo4j import GraphDatabase

    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run("""
            MATCH (a:Application {application_id: $app_id})
            OPTIONAL MATCH (a)-[:HAS_FEATURE]->(f:Feature)
            RETURN a, collect(f) as features
        """, app_id=application_id)

        record = result.single()
        if not record:
            return None

        app = record['a']
        features = record['features']

        return {
            'application_id': application_id,
            'prediction': app.get('prediction', 'Unknown'),
            'features': {f['name']: f['value'] for f in features if f},
            'shap_values': {},  # Add if available
        }

    driver.close()


def generate_explanations(
    application_ids=None,
    audiences=None,
    use_concepts_options=None
):
    """
    Generate and cache explanations.

    Args:
        application_ids: List of app IDs or None for all
        audiences: List of audiences or None for all
        use_concepts_options: List of True/False or None for both
    """
    cache = ExplanationCache()

    # Get application IDs
    if application_ids is None:
        print("[GENERATE] Fetching application IDs from Neo4j...")
        try:
            application_ids = get_application_ids_from_neo4j()
        except Exception as e:
            print(f"[ERROR] Could not fetch from Neo4j: {e}")
            print("[GENERATE] Using sample IDs: LP001003, LP001006")
            application_ids = ["LP001003", "LP001006"]

    # Default audiences
    if audiences is None:
        audiences = ["technical", "business", "customer"]

    # Default concept options
    if use_concepts_options is None:
        use_concepts_options = [True, False]

    print(f"\n[GENERATE] Configuration:")
    print(f"  Application IDs: {len(application_ids)}")
    print(f"  Audiences: {audiences}")
    print(f"  Concept options: {use_concepts_options}")
    print(f"  Total to generate: {len(application_ids) * len(audiences) * len(use_concepts_options)}")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY not found in environment!")
        print("Set your API key in .env file:")
        print("  OPENAI_API_KEY=sk-...")
        return

    generated = 0
    skipped = 0
    errors = 0

    for app_id in application_ids:
        print(f"\n[GENERATE] Processing {app_id}...")

        # Get application context
        try:
            context = get_application_explanation_data(app_id)
            if not context:
                print(f"  [SKIP] No data found for {app_id}")
                continue
        except Exception as e:
            print(f"  [ERROR] Failed to get data: {e}")
            errors += 1
            continue

        for audience in audiences:
            for use_concepts in use_concepts_options:
                # Check if already cached
                if cache.has_explanation(app_id, audience, use_concepts):
                    print(f"  [SKIP] {audience}, concepts={use_concepts} (already cached)")
                    skipped += 1
                    continue

                print(f"  [GENERATE] {audience}, concepts={use_concepts}...", end=" ")

                try:
                    # Generate explanation
                    explanation = call_llm_for_explanation(
                        context,
                        audience=audience,
                        use_concepts=use_concepts,
                        enable_logging=False  # Don't log when caching
                    )

                    # Cache it
                    cache.set_explanation(app_id, audience, use_concepts, explanation)
                    generated += 1
                    print("✓")

                except Exception as e:
                    print(f"✗ ({e})")
                    errors += 1

    print(f"\n[SUMMARY]")
    print(f"  Generated: {generated}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Errors: {errors}")
    print(f"\n[CACHE] {cache.cache_file}")

    stats = cache.stats()
    print(f"  Total cached: {stats['total_cached']}")
    print(f"  Cache size: {stats['cache_size_bytes']:,} bytes")


def get_application_ids_from_neo4j():
    """Fetch all application IDs from Neo4j"""
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run("MATCH (a:Application) RETURN a.application_id as id ORDER BY id")
        ids = [record['id'] for record in result]

    driver.close()
    return ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate cached explanations")
    parser.add_argument("--app-ids", help="Comma-separated application IDs")
    parser.add_argument("--audiences", help="Comma-separated audiences (technical,business,customer)")
    parser.add_argument("--concepts", action="store_true", help="Only generate with concepts=True")
    parser.add_argument("--no-concepts", action="store_true", help="Only generate with concepts=False")

    args = parser.parse_args()

    # Parse arguments
    app_ids = args.app_ids.split(',') if args.app_ids else None
    audiences = args.audiences.split(',') if args.audiences else None

    # Determine concept options
    if args.concepts:
        use_concepts_options = [True]
    elif args.no_concepts:
        use_concepts_options = [False]
    else:
        use_concepts_options = None

    # Generate
    generate_explanations(
        application_ids=app_ids,
        audiences=audiences,
        use_concepts_options=use_concepts_options
    )
