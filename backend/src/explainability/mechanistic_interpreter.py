"""
Mechanistic Interpretability Module for HOGE Framework

This module provides deep mechanistic analysis of XGBoost models by:
1. Analyzing individual tree decision paths
2. Detecting feature interactions and dependencies
3. Mapping decision boundaries and split patterns
4. Identifying internal representations and activations
5. Computing tree-level contribution patterns

Integrates with HOGE's Neo4j knowledge graph to persist mechanistic insights.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Import Plotly visualizer for interactive charts
try:
    from src.explainability.tree_visualizer import TreeVisualizer, create_all_interactive_visualizations
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly visualizer not available. Install plotly for interactive visualizations.")


@dataclass
class DecisionPath:
    """Represents a single decision path through one tree"""
    tree_id: int
    path_nodes: List[int]
    split_features: List[str]
    split_thresholds: List[float]
    split_directions: List[str]  # "left" or "right"
    leaf_value: float
    path_length: int
    feature_activation_pattern: Dict[str, List[float]]


@dataclass
class FeatureInteraction:
    """Represents an interaction between two features"""
    feature_a: str
    feature_b: str
    interaction_strength: float
    co_occurrence_count: int
    avg_depth_difference: float
    trees_with_interaction: List[int]
    interaction_type: str  # "sequential", "parallel", "nested"


@dataclass
class TreeLevelInsight:
    """Mechanistic insights at the individual tree level"""
    tree_id: int
    max_depth: int
    num_splits: int
    feature_importance_local: Dict[str, float]
    dominant_feature: str
    split_distribution: Dict[str, int]
    leaf_value_mean: float
    leaf_value_std: float


class MechanisticInterpreter:
    """
    Mechanistic interpretability analyzer for XGBoost models.

    Provides deep introspection into model internals:
    - Decision path analysis
    - Feature interaction detection
    - Tree-level pattern recognition
    - Internal activation mapping
    """

    def __init__(self, model_path: str, feature_names: List[str]):
        """
        Initialize mechanistic interpreter.

        Args:
            model_path: Path to trained XGBoost model (.joblib)
            feature_names: List of feature names in training order
        """
        self.pipeline = joblib.load(model_path)
        self.feature_names = feature_names

        # Extract the actual XGBoost model from pipeline
        if hasattr(self.pipeline, 'named_steps'):
            self.model = self.pipeline.named_steps['model']
        else:
            self.model = self.pipeline

        self.booster = self.model.get_booster()
        self.trees_df = self.booster.trees_to_dataframe()

        # Create feature name mapping from f0, f1, f2... to actual names
        # After preprocessing, features are named f0, f1, f2, etc.
        # We need to map these back to original feature names
        self.feature_name_map = self._create_feature_name_map()

        # Cache for computed insights
        self.decision_paths_cache = {}
        self.feature_interactions_cache = None
        self.tree_insights_cache = None

    def _create_feature_name_map(self) -> Dict[str, str]:
        """
        Create mapping from XGBoost feature indices (f0, f1...) to actual feature names.

        After ColumnTransformer preprocessing, XGBoost sees features as f0, f1, f2...
        We need to map these back to human-readable names.
        """
        feature_map = {}

        try:
            # Get feature names after preprocessing
            if hasattr(self.pipeline, 'named_steps') and 'preprocess' in self.pipeline.named_steps:
                preprocessor = self.pipeline.named_steps['preprocess']

                # Try to get feature names from the preprocessor
                if hasattr(preprocessor, 'get_feature_names_out'):
                    try:
                        transformed_names = preprocessor.get_feature_names_out()

                        # Map f0, f1, f2... to actual names
                        for i, name in enumerate(transformed_names):
                            # Clean up name (remove prefixes like 'num__', 'cat__')
                            clean_name = name.split('__')[-1] if '__' in name else name
                            feature_map[f'f{i}'] = clean_name

                    except Exception as e:
                        print(f"Could not get feature names from preprocessor: {e}")

            # Fallback: if mapping failed, at least map to input feature names
            if not feature_map:
                for i, name in enumerate(self.feature_names):
                    feature_map[f'f{i}'] = name

        except Exception as e:
            print(f"Error creating feature map: {e}")
            # Ultimate fallback: keep f0, f1, f2...
            pass

        return feature_map if feature_map else {f'f{i}': f'f{i}' for i in range(100)}

    def _map_feature_name(self, xgb_feature: str) -> str:
        """Convert XGBoost feature name (f0, f1...) to human-readable name"""
        return self.feature_name_map.get(xgb_feature, xgb_feature)

    def extract_decision_path(self,
                             application_id: str,
                             X: pd.DataFrame,
                             tree_limit: int = None) -> List[DecisionPath]:
        """
        Extract complete decision paths for a single application across all trees.

        This reveals the mechanistic flow of information through the ensemble.

        Args:
            application_id: Unique application identifier
            X: Preprocessed feature matrix (single row or full dataset)
            tree_limit: Optional limit on number of trees to analyze

        Returns:
            List of DecisionPath objects, one per tree
        """
        if application_id in self.decision_paths_cache:
            return self.decision_paths_cache[application_id]

        # Get prediction leaf indices for this sample
        X_transformed = self.pipeline.named_steps['preprocess'].transform(X)
        leaf_indices = self.model.apply(X_transformed)[0]  # Shape: (n_trees,)

        num_trees = len(leaf_indices) if tree_limit is None else min(tree_limit, len(leaf_indices))
        decision_paths = []

        for tree_id in range(num_trees):
            leaf_id = leaf_indices[tree_id]
            path = self._trace_path_to_leaf(tree_id, leaf_id)
            decision_paths.append(path)

        self.decision_paths_cache[application_id] = decision_paths
        return decision_paths

    def _trace_path_to_leaf(self, tree_id: int, leaf_id: int) -> DecisionPath:
        """
        Trace the path from root to a specific leaf in a tree.

        This is the core mechanistic analysis: understanding which features
        were activated and in what order.
        """
        tree_df = self.trees_df[self.trees_df['Tree'] == tree_id].copy()

        # Find the leaf node by matching Node number (not ID)
        leaf_row = tree_df[tree_df['Node'] == leaf_id]
        if leaf_row.empty:
            # Fallback: return empty path
            return DecisionPath(
                tree_id=tree_id,
                path_nodes=[],
                split_features=[],
                split_thresholds=[],
                split_directions=[],
                leaf_value=0.0,
                path_length=0,
                feature_activation_pattern={}
            )

        leaf_value = leaf_row.iloc[0]['Gain'] if 'Gain' in leaf_row else 0.0

        # Build parent-child map using 'Yes' and 'No' columns
        # In XGBoost trees_to_dataframe:
        # - 'Yes' column = left child ID (where condition is True)
        # - 'No' column = right child ID (where condition is False)

        # Create reverse mapping: child_id -> (parent_id, direction)
        child_to_parent = {}
        for idx, row in tree_df.iterrows():
            if row['Feature'] != 'Leaf':  # Not a leaf, has children
                yes_child = row.get('Yes')  # Left child
                no_child = row.get('No')    # Right child
                parent_id = row['ID']

                if pd.notna(yes_child):
                    child_to_parent[yes_child] = (parent_id, 'left')
                if pd.notna(no_child):
                    child_to_parent[no_child] = (parent_id, 'right')

        # Backtrack from leaf to root
        path_nodes = []
        split_features = []
        split_thresholds = []
        split_directions = []
        feature_activations = defaultdict(list)

        current_id = leaf_row.iloc[0]['ID']
        visited = set()
        max_iterations = 100  # Safety limit

        # Trace path backwards from leaf to root
        while current_id not in visited and len(visited) < max_iterations:
            visited.add(current_id)
            node_row = tree_df[tree_df['ID'] == current_id]

            if node_row.empty:
                break

            node_row = node_row.iloc[0]
            node_num = node_row['Node']
            path_nodes.insert(0, node_num)

            # Check if this is a split node (not a leaf)
            feature = node_row.get('Feature')
            if pd.notna(feature) and feature != 'Leaf':
                # Map XGBoost feature name (f0, f1...) to human-readable name
                feature_name = self._map_feature_name(feature)
                split_features.insert(0, feature_name)
                threshold = node_row.get('Split', 0.0)
                split_thresholds.insert(0, threshold)
                feature_activations[feature_name].append(threshold)

                # Get direction from child_to_parent mapping
                if current_id in child_to_parent:
                    _, direction = child_to_parent[current_id]
                    split_directions.insert(0, direction)

            # Move to parent
            if current_id in child_to_parent:
                parent_id, _ = child_to_parent[current_id]
                current_id = parent_id
            else:
                # Reached root (no parent)
                break

        return DecisionPath(
            tree_id=tree_id,
            path_nodes=path_nodes,
            split_features=split_features,
            split_thresholds=split_thresholds,
            split_directions=split_directions,
            leaf_value=float(leaf_value),
            path_length=len(path_nodes),
            feature_activation_pattern=dict(feature_activations)
        )

    def detect_feature_interactions(self,
                                    min_co_occurrence: int = 5) -> List[FeatureInteraction]:
        """
        Detect feature interactions by analyzing co-occurrence patterns in tree splits.

        This reveals mechanistic dependencies between features - which features
        "work together" in the model's decision process.

        Args:
            min_co_occurrence: Minimum number of trees where features must co-occur

        Returns:
            List of FeatureInteraction objects
        """
        if self.feature_interactions_cache is not None:
            return self.feature_interactions_cache

        interactions = []
        feature_pairs = defaultdict(lambda: {
            'count': 0,
            'trees': [],
            'depth_diffs': []
        })

        # Analyze each tree
        for tree_id in self.trees_df['Tree'].unique():
            tree_df = self.trees_df[self.trees_df['Tree'] == tree_id]

            # Get all split features in this tree and map to human-readable names
            split_features_raw = tree_df[tree_df['Feature'] != 'Leaf']['Feature'].tolist()
            split_features = [self._map_feature_name(f) for f in split_features_raw]
            feature_depths = {}

            for idx, row in tree_df.iterrows():
                if row['Feature'] != 'Leaf':
                    mapped_name = self._map_feature_name(row['Feature'])
                    feature_depths[mapped_name] = row.get('Depth', 0)

            # Find all pairs
            unique_features = list(set(split_features))
            for i, feat_a in enumerate(unique_features):
                for feat_b in unique_features[i+1:]:
                    pair_key = tuple(sorted([feat_a, feat_b]))
                    feature_pairs[pair_key]['count'] += 1
                    feature_pairs[pair_key]['trees'].append(tree_id)

                    depth_a = feature_depths.get(feat_a, 0)
                    depth_b = feature_depths.get(feat_b, 0)
                    feature_pairs[pair_key]['depth_diffs'].append(abs(depth_a - depth_b))

        # Convert to FeatureInteraction objects
        for (feat_a, feat_b), stats in feature_pairs.items():
            if stats['count'] >= min_co_occurrence:
                avg_depth_diff = np.mean(stats['depth_diffs']) if stats['depth_diffs'] else 0

                # Classify interaction type
                if avg_depth_diff < 0.5:
                    interaction_type = "parallel"  # Same depth - parallel processing
                elif avg_depth_diff < 2:
                    interaction_type = "sequential"  # Close depth - sequential
                else:
                    interaction_type = "nested"  # Large depth difference

                interaction = FeatureInteraction(
                    feature_a=feat_a,
                    feature_b=feat_b,
                    interaction_strength=stats['count'] / len(self.trees_df['Tree'].unique()),
                    co_occurrence_count=stats['count'],
                    avg_depth_difference=avg_depth_diff,
                    trees_with_interaction=stats['trees'],
                    interaction_type=interaction_type
                )
                interactions.append(interaction)

        # Sort by interaction strength
        interactions.sort(key=lambda x: x.interaction_strength, reverse=True)
        self.feature_interactions_cache = interactions
        return interactions

    def analyze_tree_level_patterns(self) -> List[TreeLevelInsight]:
        """
        Analyze mechanistic patterns at individual tree level.

        Each tree in the ensemble acts like a "neuron" - this function
        analyzes what each tree specializes in.

        Returns:
            List of TreeLevelInsight objects
        """
        if self.tree_insights_cache is not None:
            return self.tree_insights_cache

        insights = []

        for tree_id in self.trees_df['Tree'].unique():
            tree_df = self.trees_df[self.trees_df['Tree'] == tree_id]

            # Split nodes only
            split_df = tree_df[tree_df['Feature'] != 'Leaf']
            leaf_df = tree_df[tree_df['Feature'] == 'Leaf']

            # Feature importance within this tree - map to human-readable names
            feature_counts_raw = Counter(split_df['Feature'].tolist())
            feature_counts = Counter({
                self._map_feature_name(feat): count
                for feat, count in feature_counts_raw.items()
            })
            total_splits = len(split_df)

            feature_importance = {
                feat: count / total_splits
                for feat, count in feature_counts.items()
            } if total_splits > 0 else {}

            dominant_feature = max(feature_importance.items(),
                                  key=lambda x: x[1])[0] if feature_importance else "none"

            # Leaf statistics
            leaf_values = leaf_df['Gain'].values if 'Gain' in leaf_df else []
            leaf_mean = float(np.mean(leaf_values)) if len(leaf_values) > 0 else 0.0
            leaf_std = float(np.std(leaf_values)) if len(leaf_values) > 0 else 0.0

            insight = TreeLevelInsight(
                tree_id=int(tree_id),
                max_depth=int(tree_df['Depth'].max()) if 'Depth' in tree_df else 0,
                num_splits=len(split_df),
                feature_importance_local=feature_importance,
                dominant_feature=dominant_feature,
                split_distribution=dict(feature_counts),
                leaf_value_mean=leaf_mean,
                leaf_value_std=leaf_std
            )
            insights.append(insight)

        self.tree_insights_cache = insights
        return insights

    def compute_activation_patterns(self,
                                    X: pd.DataFrame,
                                    feature_name: str) -> Dict[str, Any]:
        """
        Compute activation patterns for a specific feature across all trees.

        This shows how different input values activate different parts of the model.

        Args:
            X: Input data
            feature_name: Feature to analyze

        Returns:
            Dictionary with activation statistics
        """
        X_transformed = self.pipeline.named_steps['preprocess'].transform(X)

        # Get feature index (after preprocessing)
        # This is simplified - real implementation needs proper feature mapping
        feature_activations = []

        for tree_id in self.trees_df['Tree'].unique():
            tree_df = self.trees_df[self.trees_df['Tree'] == tree_id]
            feature_splits = tree_df[tree_df['Feature'] == feature_name]

            if not feature_splits.empty:
                if 'Split' in feature_splits.columns:
                    thresholds = feature_splits['Split'].values.tolist()
                else:
                    thresholds = []

                if 'Depth' in feature_splits.columns:
                    depths = feature_splits['Depth'].values.tolist()
                else:
                    depths = [0] * len(feature_splits)

                feature_activations.append({
                    'tree_id': tree_id,
                    'thresholds': thresholds,
                    'depths': depths,
                    'num_splits': len(feature_splits)
                })

        return {
            'feature': feature_name,
            'total_trees_using': len(feature_activations),
            'tree_usage_rate': len(feature_activations) / len(self.trees_df['Tree'].unique()),
            'activations': feature_activations,
            'unique_thresholds': len(set(
                t for act in feature_activations for t in act['thresholds']
            )) if feature_activations else 0
        }

    def visualize_decision_path(self,
                               paths: List[DecisionPath],
                               output_path: str = None,
                               top_k: int = 10) -> None:
        """
        Visualize decision paths across trees.

        Args:
            paths: List of DecisionPath objects
            output_path: Where to save the visualization
            top_k: Number of trees to visualize
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Path lengths across trees
        tree_ids = [p.tree_id for p in paths[:top_k]]
        path_lengths = [p.path_length for p in paths[:top_k]]
        leaf_values = [p.leaf_value for p in paths[:top_k]]

        ax1 = axes[0]
        ax1.bar(tree_ids, path_lengths, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Tree ID')
        ax1.set_ylabel('Path Length (Depth)')
        ax1.set_title('Decision Path Depths Across Trees')
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Feature activation heatmap
        ax2 = axes[1]
        feature_usage = defaultdict(list)

        for path in paths[:top_k]:
            for feat in path.split_features:
                feature_usage[feat].append(path.tree_id)

        if feature_usage:
            features = list(feature_usage.keys())[:15]  # Top 15 features
            usage_matrix = np.zeros((len(features), top_k))

            for i, feat in enumerate(features):
                for tree_id in feature_usage[feat]:
                    if tree_id < top_k:
                        usage_matrix[i, tree_id] = 1

            sns.heatmap(usage_matrix,
                       yticklabels=features,
                       xticklabels=tree_ids,
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Feature Used'},
                       ax=ax2)
            ax2.set_xlabel('Tree ID')
            ax2.set_ylabel('Feature')
            ax2.set_title('Feature Activation Pattern Across Trees')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Decision path visualization saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_feature_interactions(self,
                                       interactions: List[FeatureInteraction],
                                       output_path: str = None,
                                       top_k: int = 20) -> None:
        """
        Visualize feature interaction network.

        Args:
            interactions: List of FeatureInteraction objects
            output_path: Where to save the visualization
            top_k: Number of top interactions to show
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Interaction strength bar chart
        ax1 = axes[0]
        top_interactions = interactions[:top_k]
        labels = [f"{i.feature_a}\n×\n{i.feature_b}" for i in top_interactions]
        strengths = [i.interaction_strength for i in top_interactions]
        colors = ['#1f77b4' if i.interaction_type == 'sequential' else
                 '#ff7f0e' if i.interaction_type == 'parallel' else '#2ca02c'
                 for i in top_interactions]

        ax1.barh(range(len(labels)), strengths, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel('Interaction Strength')
        ax1.set_title('Top Feature Interactions by Strength')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        # Add legend for interaction types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Sequential'),
            Patch(facecolor='#ff7f0e', label='Parallel'),
            Patch(facecolor='#2ca02c', label='Nested')
        ]
        ax1.legend(handles=legend_elements, loc='lower right')

        # Plot 2: Interaction type distribution
        ax2 = axes[1]
        type_counts = Counter([i.interaction_type for i in interactions])
        ax2.pie(type_counts.values(),
               labels=type_counts.keys(),
               autopct='%1.1f%%',
               colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('Distribution of Interaction Types')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Feature interaction visualization saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def generate_narrative_explanation(self,
                                       application_id: str,
                                       paths: List[DecisionPath],
                                       interactions: List[FeatureInteraction],
                                       tree_insights: List[TreeLevelInsight],
                                       audience: str = "technical") -> str:
        """
        Generate human-readable narrative explanation of mechanistic insights.

        Args:
            application_id: Application identifier
            paths: Decision paths
            interactions: Feature interactions
            tree_insights: Tree-level insights
            audience: Target audience type - "technical", "non_technical", "executive", or "business"

        Returns:
            Narrative explanation string
        """
        # Compute summary statistics
        avg_path_length = float(np.mean([p.path_length for p in paths]))
        most_common_features = self._get_most_common_features(paths, top_k=5)
        top_interactions = interactions[:3]

        # Audience-specific narratives
        if audience == "technical":
            narrative = f"""
**Mechanistic Interpretability Analysis for Application {application_id}**

**Decision Path Analysis:**
- Analyzed {len(paths)} decision trees in the ensemble
- Average path length: {avg_path_length:.2f} nodes (indicates model complexity)
- Total feature interactions detected: {len(interactions)}

**Feature Activation Patterns:**
The most frequently activated features across the ensemble are:
{chr(10).join([f"- {feat}: used in {count} trees ({count/len(paths)*100:.1f}%)" for feat, count in most_common_features.items()])}

**Feature Interactions (Mechanistic Dependencies):**
{chr(10).join([f"- {i.feature_a} × {i.feature_b}: strength={i.interaction_strength:.3f}, type={i.interaction_type}" for i in top_interactions]) if top_interactions else "No significant interactions detected"}

**Tree Specialization:**
Individual trees in the ensemble show specialization, with each tree focusing on different feature combinations.
This ensemble diversity contributes to robust prediction performance.
"""
        elif audience == "non_technical":
            narrative = f"""
**How the System Made This Decision (Simple Explanation)**

The loan decision system is like a panel of {len(paths)} expert judges, where each judge looks at different aspects of your application.

**Main Factors Considered:**
The system paid most attention to these factors:
{chr(10).join([f"- {feat.replace('_', ' ').title()}" for feat in list(most_common_features.keys())[:3]])}

**How Factors Work Together:**
{f"The system noticed that {top_interactions[0].feature_a.replace('_', ' ')} and {top_interactions[0].feature_b.replace('_', ' ')} work together when making decisions." if top_interactions else "Each factor was considered independently."}

**Overall Process:**
On average, each expert judge considered about {int(avg_path_length)} different factors before making their assessment.
"""
        elif audience == "executive":
            narrative = f"""
**Executive Summary: Model Decision Analysis for {application_id}**

**Model Architecture:**
- Ensemble model with {len(paths)} decision components
- Average decision complexity: {avg_path_length:.1f} evaluation steps

**Key Decision Drivers:**
Top 3 factors influencing this decision:
{chr(10).join([f"{i+1}. {feat.replace('_', ' ').title()} (used in {count/len(paths)*100:.0f}% of model components)" for i, (feat, count) in enumerate(list(most_common_features.items())[:3])])}

**Risk Assessment:**
- Feature interaction complexity: {len(interactions)} interactions detected
- Decision stability: {"High" if len(top_interactions) < 10 else "Moderate"} (based on interaction patterns)

**Business Implications:**
The model demonstrates {avg_path_length/10*100:.0f}% decision complexity, indicating {"detailed" if avg_path_length > 5 else "streamlined"} evaluation process.
"""
        else:  # business
            narrative = f"""
**Business Analysis: Decision Breakdown for {application_id}**

**Decision Process Overview:**
The model evaluated this application using {len(paths)} parallel decision pathways, averaging {avg_path_length:.1f} evaluation steps per pathway.

**Primary Evaluation Factors:**
{chr(10).join([f"• {feat.replace('_', ' ').title()}: appeared in {count} decision paths ({count/len(paths)*100:.1f}%)" for feat, count in list(most_common_features.items())[:5]])}

**Feature Relationships:**
{f"Notable interaction patterns detected:" + chr(10) + chr(10).join([f"• {i.feature_a.replace('_', ' ').title()} and {i.feature_b.replace('_', ' ').title()} show {i.interaction_type} relationship (strength: {i.interaction_strength:.2f})" for i in top_interactions[:3]]) if top_interactions else "Features were evaluated independently without significant cross-dependencies."}

**Operational Insight:**
This analysis reveals which application characteristics drive model decisions, useful for understanding approval patterns and applicant guidance.
"""

        return narrative.strip()

    def export_to_json(self,
                      application_id: str,
                      paths: List[DecisionPath],
                      interactions: List[FeatureInteraction],
                      tree_insights: List[TreeLevelInsight],
                      output_path: str,
                      audience: str = "technical") -> None:
        """
        Export all mechanistic insights to JSON for KG integration.

        Args:
            application_id: Application identifier
            paths: Decision paths
            interactions: Feature interactions
            tree_insights: Tree-level insights
            output_path: Where to save JSON
            audience: Target audience for narrative generation
        """
        export_data = {
            "application_id": application_id,
            "mechanistic_analysis": {
                "decision_paths": [
                    {
                        "tree_id": p.tree_id,
                        "path_length": p.path_length,
                        "split_features": p.split_features,
                        "split_thresholds": [float(t) for t in p.split_thresholds],
                        "leaf_value": float(p.leaf_value),
                        "feature_activation_pattern": {
                            k: [float(v) for v in vals]
                            for k, vals in p.feature_activation_pattern.items()
                        }
                    }
                    for p in paths
                ],
                "feature_interactions": [
                    asdict(i) for i in interactions
                ],
                "tree_insights": [
                    asdict(t) for t in tree_insights
                ],
                "summary": {
                    "total_trees_analyzed": len(paths),
                    "total_interactions_detected": len(interactions),
                    "avg_path_length": float(np.mean([p.path_length for p in paths])),
                    "most_common_features_in_paths": self._get_most_common_features(paths, top_k=10)
                },
                "narrative_explanation": self.generate_narrative_explanation(
                    application_id, paths, interactions, tree_insights, audience
                )
            },
            "provenance": {
                "analyzer": "MechanisticInterpreter",
                "version": "1.0.0",
                "method": "tree_path_analysis",
                "audience": audience
            }
        }

        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, cls=NumpyEncoder)

        print(f"Mechanistic analysis exported to {output_path}")

    def create_interactive_visualizations(self,
                                         application_id: str,
                                         paths: List[DecisionPath],
                                         interactions: List[FeatureInteraction],
                                         tree_insights: List[TreeLevelInsight],
                                         output_dir: str = "figures/mechanistic_interactive") -> Dict[str, str]:
        """
        Create interactive Plotly visualizations instead of static matplotlib charts

        Args:
            application_id: Application identifier
            paths: Decision paths
            interactions: Feature interactions
            tree_insights: Tree-level insights
            output_dir: Directory to save HTML visualizations

        Returns:
            Dictionary mapping visualization type to file path
        """
        if not PLOTLY_AVAILABLE:
            print("Error: Plotly not available. Install with: pip install plotly")
            print("Falling back to matplotlib visualizations...")
            return {}

        os.makedirs(output_dir, exist_ok=True)

        # Create results dictionary for visualizer
        mechanistic_results = {
            'application_id': application_id,
            'decision_paths': paths,
            'feature_interactions': interactions,
            'tree_insights': tree_insights
        }

        # Generate all interactive visualizations
        output_files = create_all_interactive_visualizations(
            mechanistic_results,
            output_dir=output_dir
        )

        return output_files

    def _get_most_common_features(self, paths: List[DecisionPath], top_k: int = 10) -> Dict[str, int]:
        """Helper to get most commonly used features across paths"""
        feature_counts = Counter()
        for path in paths:
            feature_counts.update(path.split_features)
        return dict(feature_counts.most_common(top_k))


def main():
    """Example usage of mechanistic interpreter"""
    import sys

    # Configuration
    MODEL_PATH = "models/loan_xgb_monotonic.joblib"
    DATA_PATH = "data/processed/scored_applications_xgb.csv"

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Get application ID
    if len(sys.argv) > 1:
        application_id = sys.argv[1]
    else:
        application_id = input("Enter application ID (e.g., LP001006): ").strip()

    # Filter to this application
    app_data = df[df['Loan_ID'] == application_id]
    if app_data.empty:
        print(f"Application {application_id} not found")
        sys.exit(1)

    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['Loan_ID', 'Loan_Status', 'Probability']]
    X = app_data[feature_cols]

    print(f"\n=== Mechanistic Interpretability Analysis for {application_id} ===\n")

    # Initialize interpreter
    interpreter = MechanisticInterpreter(MODEL_PATH, feature_cols)

    # 1. Extract decision paths
    print("1. Extracting decision paths...")
    paths = interpreter.extract_decision_path(application_id, X, tree_limit=50)
    print(f"   Analyzed {len(paths)} trees")
    print(f"   Average path length: {np.mean([p.path_length for p in paths]):.2f}")
    print(f"   Average leaf value: {np.mean([p.leaf_value for p in paths]):.4f}")

    # 2. Detect feature interactions
    print("\n2. Detecting feature interactions...")
    interactions = interpreter.detect_feature_interactions(min_co_occurrence=5)
    print(f"   Found {len(interactions)} significant interactions")
    print(f"\n   Top 5 interactions:")
    for i, interaction in enumerate(interactions[:5], 1):
        print(f"   {i}. {interaction.feature_a} × {interaction.feature_b}")
        print(f"      Strength: {interaction.interaction_strength:.3f}, Type: {interaction.interaction_type}")

    # 3. Analyze tree-level patterns
    print("\n3. Analyzing tree-level patterns...")
    tree_insights = interpreter.analyze_tree_level_patterns()
    dominant_features = Counter([t.dominant_feature for t in tree_insights])
    print(f"   Most specialized features across trees:")
    for feat, count in dominant_features.most_common(5):
        print(f"   - {feat}: dominates {count} trees")

    # 4. Interactive Visualizations (Plotly)
    print("\n4. Generating interactive visualizations (Plotly)...")
    if PLOTLY_AVAILABLE:
        output_files = interpreter.create_interactive_visualizations(
            application_id,
            paths,
            interactions,
            tree_insights,
            output_dir="figures/mechanistic_interactive"
        )
        print(f"   Created {len(output_files)} interactive HTML visualizations")
    else:
        print("   Plotly not available - using matplotlib fallback...")
        os.makedirs("figures/mechanistic", exist_ok=True)

        interpreter.visualize_decision_path(
            paths,
            output_path=f"figures/mechanistic/decision_paths_{application_id}.png",
            top_k=20
        )

        interpreter.visualize_feature_interactions(
            interactions,
            output_path=f"figures/mechanistic/feature_interactions.png",
            top_k=15
        )

    # 5. Export to JSON
    print("\n5. Exporting mechanistic insights...")
    os.makedirs("data/evaluation", exist_ok=True)
    interpreter.export_to_json(
        application_id,
        paths,
        interactions,
        tree_insights,
        output_path=f"data/evaluation/mechanistic_{application_id}.json"
    )

    print("\n=== Analysis Complete ===")
    print(f"\nOutputs saved:")
    if PLOTLY_AVAILABLE and output_files:
        for viz_type, filepath in output_files.items():
            print(f"  - {filepath}")
    else:
        print(f"  - figures/mechanistic/decision_paths_{application_id}.png")
        print(f"  - figures/mechanistic/feature_interactions.png")
    print(f"  - data/evaluation/mechanistic_{application_id}.json")


if __name__ == "__main__":
    main()
