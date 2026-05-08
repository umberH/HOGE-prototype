"""
Neo4j Mechanistic Insights Loader

Extends HOGE knowledge graph with mechanistic interpretability insights:
- DecisionPath nodes (tree-level decision traces)
- FeatureInteraction nodes (detected feature dependencies)
- TreeSpecialization nodes (what each tree "specializes" in)
- ActivationPattern relationships

Integrates with mechanistic_interpreter.py output.
"""

import os
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class MechanisticKGLoader:
    """
    Loads mechanistic interpretability insights into Neo4j knowledge graph.

    Creates new node types:
    - DecisionPath: Individual tree's decision trace for an application
    - FeatureInteraction: Detected interactions between features
    - TreeSpecialization: What each tree in the ensemble specializes in
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Create uniqueness constraints for mechanistic nodes"""
        constraints = [
            "CREATE CONSTRAINT decision_path_id IF NOT EXISTS FOR (dp:DecisionPath) REQUIRE dp.path_id IS UNIQUE",
            "CREATE CONSTRAINT feature_interaction_id IF NOT EXISTS FOR (fi:FeatureInteraction) REQUIRE fi.interaction_id IS UNIQUE",
            "CREATE CONSTRAINT tree_spec_id IF NOT EXISTS FOR (ts:TreeSpecialization) REQUIRE ts.tree_id IS UNIQUE",
        ]

        with self.driver.session(database=NEO4J_DATABASE) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"✓ Created constraint: {constraint.split('(')[1].split(')')[0]}")
                except Exception as e:
                    if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                        print(f"  Constraint already exists (skipping)")
                    else:
                        print(f"  Warning: {e}")

    def load_decision_paths(self,
                           application_id: str,
                           paths: List[Dict[str, Any]]) -> int:
        """
        Load decision paths for an application into the knowledge graph.

        Creates:
        - DecisionPath nodes (one per tree)
        - HAS_DECISION_PATH relationships from LoanApplication
        - USES_FEATURE relationships from DecisionPath to Feature
        - ACTIVATES relationships showing threshold activations

        Args:
            application_id: Application identifier
            paths: List of decision path dictionaries

        Returns:
            Number of paths loaded
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            for path in paths:
                path_id = f"{application_id}_tree_{path['tree_id']}"

                # Create DecisionPath node
                session.run("""
                    MERGE (dp:DecisionPath {path_id: $path_id})
                    SET dp.application_id = $application_id,
                        dp.tree_id = $tree_id,
                        dp.path_length = $path_length,
                        dp.leaf_value = $leaf_value,
                        dp.split_features = $split_features,
                        dp.split_thresholds = $split_thresholds
                """, {
                    "path_id": path_id,
                    "application_id": application_id,
                    "tree_id": path['tree_id'],
                    "path_length": path['path_length'],
                    "leaf_value": path['leaf_value'],
                    "split_features": path['split_features'],
                    "split_thresholds": path['split_thresholds']
                })

                # Link to LoanApplication
                session.run("""
                    MATCH (app:LoanApplication {application_id: $application_id})
                    MATCH (dp:DecisionPath {path_id: $path_id})
                    MERGE (app)-[:HAS_DECISION_PATH]->(dp)
                """, {
                    "application_id": application_id,
                    "path_id": path_id
                })

                # Link to features used in this path
                for i, feature_name in enumerate(path['split_features']):
                    threshold = path['split_thresholds'][i] if i < len(path['split_thresholds']) else None

                    session.run("""
                        MATCH (dp:DecisionPath {path_id: $path_id})
                        MATCH (f:Feature {name: $feature_name})
                        MERGE (dp)-[r:USES_FEATURE]->(f)
                        SET r.split_order = $split_order,
                            r.threshold = $threshold
                    """, {
                        "path_id": path_id,
                        "feature_name": feature_name,
                        "split_order": i,
                        "threshold": threshold
                    })

                # Create activation patterns
                for feature_name, thresholds in path.get('feature_activation_pattern', {}).items():
                    session.run("""
                        MATCH (dp:DecisionPath {path_id: $path_id})
                        MATCH (f:Feature {name: $feature_name})
                        MERGE (dp)-[r:ACTIVATES]->(f)
                        SET r.thresholds = $thresholds,
                            r.num_activations = $num_activations
                    """, {
                        "path_id": path_id,
                        "feature_name": feature_name,
                        "thresholds": thresholds,
                        "num_activations": len(thresholds)
                    })

        print(f"✓ Loaded {len(paths)} decision paths for {application_id}")
        return len(paths)

    def load_feature_interactions(self, interactions: List[Dict[str, Any]]) -> int:
        """
        Load detected feature interactions into the knowledge graph.

        Creates:
        - FeatureInteraction nodes
        - INTERACTS_WITH relationships between features
        - Metadata: interaction type, strength, co-occurrence

        Args:
            interactions: List of feature interaction dictionaries

        Returns:
            Number of interactions loaded
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            for interaction in interactions:
                interaction_id = f"{interaction['feature_a']}_×_{interaction['feature_b']}"

                # Create FeatureInteraction node
                session.run("""
                    MERGE (fi:FeatureInteraction {interaction_id: $interaction_id})
                    SET fi.feature_a = $feature_a,
                        fi.feature_b = $feature_b,
                        fi.interaction_strength = $interaction_strength,
                        fi.co_occurrence_count = $co_occurrence_count,
                        fi.avg_depth_difference = $avg_depth_difference,
                        fi.interaction_type = $interaction_type,
                        fi.trees_with_interaction = $trees_with_interaction
                """, {
                    "interaction_id": interaction_id,
                    "feature_a": interaction['feature_a'],
                    "feature_b": interaction['feature_b'],
                    "interaction_strength": interaction['interaction_strength'],
                    "co_occurrence_count": interaction['co_occurrence_count'],
                    "avg_depth_difference": interaction['avg_depth_difference'],
                    "interaction_type": interaction['interaction_type'],
                    "trees_with_interaction": interaction['trees_with_interaction']
                })

                # Link to both features
                session.run("""
                    MATCH (fi:FeatureInteraction {interaction_id: $interaction_id})
                    MATCH (fa:Feature {name: $feature_a})
                    MATCH (fb:Feature {name: $feature_b})
                    MERGE (fa)-[r1:INTERACTS_WITH]->(fi)
                    MERGE (fb)-[r2:INTERACTS_WITH]->(fi)
                    SET r1.strength = $interaction_strength,
                        r1.type = $interaction_type,
                        r2.strength = $interaction_strength,
                        r2.type = $interaction_type
                """, {
                    "interaction_id": interaction_id,
                    "feature_a": interaction['feature_a'],
                    "feature_b": interaction['feature_b'],
                    "interaction_strength": interaction['interaction_strength'],
                    "interaction_type": interaction['interaction_type']
                })

        print(f"✓ Loaded {len(interactions)} feature interactions")
        return len(interactions)

    def load_tree_specializations(self, tree_insights: List[Dict[str, Any]]) -> int:
        """
        Load tree specialization insights into the knowledge graph.

        Creates:
        - TreeSpecialization nodes (what each tree specializes in)
        - SPECIALIZES_IN relationships to dominant features

        Args:
            tree_insights: List of tree insight dictionaries

        Returns:
            Number of tree specializations loaded
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            for insight in tree_insights:
                tree_id = insight['tree_id']

                # Create TreeSpecialization node
                session.run("""
                    MERGE (ts:TreeSpecialization {tree_id: $tree_id})
                    SET ts.max_depth = $max_depth,
                        ts.num_splits = $num_splits,
                        ts.dominant_feature = $dominant_feature,
                        ts.leaf_value_mean = $leaf_value_mean,
                        ts.leaf_value_std = $leaf_value_std
                """, {
                    "tree_id": tree_id,
                    "max_depth": insight['max_depth'],
                    "num_splits": insight['num_splits'],
                    "dominant_feature": insight['dominant_feature'],
                    "leaf_value_mean": insight['leaf_value_mean'],
                    "leaf_value_std": insight['leaf_value_std']
                })

                # Link to dominant feature
                if insight['dominant_feature'] != 'none':
                    session.run("""
                        MATCH (ts:TreeSpecialization {tree_id: $tree_id})
                        MATCH (f:Feature {name: $dominant_feature})
                        MERGE (ts)-[r:SPECIALIZES_IN]->(f)
                        SET r.importance = $importance
                    """, {
                        "tree_id": tree_id,
                        "dominant_feature": insight['dominant_feature'],
                        "importance": max(insight.get('feature_importance_local', {}).values()) if insight.get('feature_importance_local') else 0.0
                    })

                # Store feature importance distribution
                for feature, importance in insight.get('feature_importance_local', {}).items():
                    session.run("""
                        MATCH (ts:TreeSpecialization {tree_id: $tree_id})
                        MATCH (f:Feature {name: $feature})
                        MERGE (ts)-[r:USES_FEATURE]->(f)
                        SET r.local_importance = $importance
                    """, {
                        "tree_id": tree_id,
                        "feature": feature,
                        "importance": importance
                    })

        print(f"✓ Loaded {len(tree_insights)} tree specializations")
        return len(tree_insights)

    def load_from_json(self, json_path: str) -> Dict[str, int]:
        """
        Load all mechanistic insights from JSON file.

        Args:
            json_path: Path to mechanistic analysis JSON

        Returns:
            Dictionary with counts of loaded entities
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        application_id = data['application_id']
        mechanistic = data['mechanistic_analysis']

        stats = {
            'decision_paths': self.load_decision_paths(
                application_id,
                mechanistic['decision_paths']
            ),
            'feature_interactions': self.load_feature_interactions(
                mechanistic['feature_interactions']
            ),
            'tree_specializations': self.load_tree_specializations(
                mechanistic['tree_insights']
            )
        }

        return stats

    def query_mechanistic_explanation(self, application_id: str) -> Dict[str, Any]:
        """
        Query mechanistic insights for an application.

        Returns enriched explanation combining:
        - Which trees contributed most
        - Which feature interactions were active
        - Detailed decision paths

        Args:
            application_id: Application identifier

        Returns:
            Dictionary with mechanistic explanation data
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            # Get decision paths
            paths_result = session.run("""
                MATCH (app:LoanApplication {application_id: $application_id})
                      -[:HAS_DECISION_PATH]->(dp:DecisionPath)
                OPTIONAL MATCH (dp)-[u:USES_FEATURE]->(f:Feature)
                RETURN dp.tree_id AS tree_id,
                       dp.path_length AS path_length,
                       dp.leaf_value AS leaf_value,
                       collect(DISTINCT {
                           feature: f.name,
                           split_order: u.split_order,
                           threshold: u.threshold
                       }) AS features_used
                ORDER BY abs(dp.leaf_value) DESC
                LIMIT 10
            """, {"application_id": application_id}).data()

            # Get active feature interactions
            interactions_result = session.run("""
                MATCH (app:LoanApplication {application_id: $application_id})
                      -[:HAS_DECISION_PATH]->(dp:DecisionPath)
                      -[:USES_FEATURE]->(f1:Feature)
                MATCH (f1)-[:INTERACTS_WITH]->(fi:FeatureInteraction)
                MATCH (f2:Feature)-[:INTERACTS_WITH]->(fi)
                WHERE f1 <> f2
                RETURN DISTINCT fi.interaction_id AS interaction_id,
                       fi.feature_a AS feature_a,
                       fi.feature_b AS feature_b,
                       fi.interaction_strength AS strength,
                       fi.interaction_type AS type
                ORDER BY strength DESC
                LIMIT 10
            """, {"application_id": application_id}).data()

            # Get tree specializations that were used
            trees_result = session.run("""
                MATCH (app:LoanApplication {application_id: $application_id})
                      -[:HAS_DECISION_PATH]->(dp:DecisionPath)
                MATCH (ts:TreeSpecialization {tree_id: dp.tree_id})
                OPTIONAL MATCH (ts)-[:SPECIALIZES_IN]->(f:Feature)
                RETURN ts.tree_id AS tree_id,
                       ts.dominant_feature AS dominant_feature,
                       ts.max_depth AS max_depth,
                       dp.leaf_value AS contribution
                ORDER BY abs(dp.leaf_value) DESC
                LIMIT 10
            """, {"application_id": application_id}).data()

        return {
            "application_id": application_id,
            "top_decision_paths": paths_result,
            "active_feature_interactions": interactions_result,
            "contributing_tree_specializations": trees_result,
            "summary": {
                "total_paths": len(paths_result),
                "total_active_interactions": len(interactions_result),
                "total_specialized_trees": len(trees_result)
            }
        }


def main():
    """Example usage: Load mechanistic insights into Neo4j"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python neo_mechanistic_loader.py <mechanistic_json_path>")
        print("Example: python neo_mechanistic_loader.py data/evaluation/mechanistic_LP001006.json")
        sys.exit(1)

    json_path = sys.argv[1]

    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    print(f"\n=== Loading Mechanistic Insights into Neo4j ===\n")
    print(f"Source: {json_path}\n")

    loader = MechanisticKGLoader()

    try:
        # Create constraints
        print("Creating constraints...")
        loader.create_constraints()

        # Load data
        print("\nLoading mechanistic data...")
        stats = loader.load_from_json(json_path)

        print("\n=== Load Summary ===")
        print(f"Decision paths: {stats['decision_paths']}")
        print(f"Feature interactions: {stats['feature_interactions']}")
        print(f"Tree specializations: {stats['tree_specializations']}")

        # Query back to verify
        with open(json_path, 'r') as f:
            data = json.load(f)
            application_id = data['application_id']

        print(f"\n=== Querying Mechanistic Explanation for {application_id} ===\n")
        explanation = loader.query_mechanistic_explanation(application_id)

        print(f"Top contributing trees:")
        for tree in explanation['contributing_tree_specializations'][:5]:
            print(f"  Tree {tree['tree_id']}: "
                  f"specializes in {tree['dominant_feature']}, "
                  f"contribution = {tree['contribution']:.4f}")

        print(f"\nActive feature interactions:")
        for interaction in explanation['active_feature_interactions'][:5]:
            print(f"  {interaction['feature_a']} × {interaction['feature_b']}: "
                  f"strength = {interaction['strength']:.3f}, "
                  f"type = {interaction['type']}")

        print("\n=== Complete ===")

    finally:
        loader.close()


if __name__ == "__main__":
    main()
