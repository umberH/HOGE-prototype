"""
Interactive Tree Visualization Module using Plotly

Provides rich, interactive visualizations of XGBoost decision trees:
1. Individual tree structure (hierarchical layout)
2. Decision paths across trees
3. Feature importance heatmaps
4. Feature interaction networks
5. Tree ensemble analysis
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import defaultdict, Counter


class TreeVisualizer:
    """
    Interactive Plotly-based visualizations for XGBoost decision trees
    """

    def __init__(self):
        self.color_scheme = {
            'approval': '#27ae60',  # Green
            'rejection': '#e74c3c',  # Red
            'neutral': '#95a5a6',   # Gray
            'feature': '#3498db',   # Blue
            'interaction': '#9b59b6'  # Purple
        }

    def visualize_decision_paths_interactive(
        self,
        paths: List,
        output_path: str = None,
        top_k: int = 20
    ) -> go.Figure:
        """
        Create interactive visualization of decision paths across trees

        Args:
            paths: List of DecisionPath objects
            output_path: Where to save HTML file
            top_k: Number of trees to visualize

        Returns:
            Plotly figure object
        """
        if not paths:
            print("No decision paths to visualize")
            return None

        paths_subset = paths[:top_k]

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Decision Path Depths Across Trees',
                'Leaf Values (Tree Contributions)',
                'Feature Usage Frequency',
                'Feature Activation Heatmap',
                'Path Length Distribution',
                'Feature Co-occurrence Network'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "histogram"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )

        # Plot 1: Path depths
        tree_ids = [p.tree_id for p in paths_subset]
        path_lengths = [p.path_length for p in paths_subset]
        leaf_values = [p.leaf_value for p in paths_subset]

        colors_depth = [self.color_scheme['approval'] if v > 0 else self.color_scheme['rejection']
                        for v in leaf_values]

        fig.add_trace(
            go.Bar(
                x=tree_ids,
                y=path_lengths,
                marker=dict(color=colors_depth),
                name='Path Depth',
                text=path_lengths,
                textposition='outside',
                hovertemplate='Tree %{x}<br>Depth: %{y}<br>Contribution: %{customdata:.4f}<extra></extra>',
                customdata=leaf_values
            ),
            row=1, col=1
        )

        # Plot 2: Leaf values (tree contributions)
        fig.add_trace(
            go.Bar(
                x=tree_ids,
                y=leaf_values,
                marker=dict(
                    color=leaf_values,
                    colorscale='RdYlGn',
                    showscale=False
                ),
                name='Leaf Value',
                text=[f"{v:.4f}" for v in leaf_values],
                textposition='outside',
                hovertemplate='Tree %{x}<br>Contribution: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )

        # Plot 3: Feature usage frequency
        all_features = []
        for path in paths_subset:
            all_features.extend(path.split_features)

        feature_counts = Counter(all_features)
        top_features = dict(feature_counts.most_common(15))

        fig.add_trace(
            go.Bar(
                x=list(top_features.values()),
                y=list(top_features.keys()),
                orientation='h',
                marker=dict(color=self.color_scheme['feature']),
                name='Feature Usage',
                text=list(top_features.values()),
                textposition='outside',
                hovertemplate='%{y}<br>Used in %{x} splits<extra></extra>'
            ),
            row=2, col=1
        )

        # Plot 4: Feature activation heatmap
        feature_usage = defaultdict(list)
        for path in paths_subset:
            for feat in path.split_features:
                feature_usage[feat].append(path.tree_id)

        if feature_usage:
            features = list(feature_counts.most_common(15))
            features = [f[0] for f in features]
            usage_matrix = np.zeros((len(features), len(paths_subset)))

            for i, feat in enumerate(features):
                for j, tree_id in enumerate(tree_ids):
                    if tree_id in feature_usage[feat]:
                        usage_matrix[i, j] = 1

            fig.add_trace(
                go.Heatmap(
                    z=usage_matrix,
                    x=tree_ids,
                    y=features,
                    colorscale='YlOrRd',
                    showscale=True,
                    hovertemplate='Tree: %{x}<br>Feature: %{y}<br>Used: %{z}<extra></extra>'
                ),
                row=2, col=2
            )

        # Plot 5: Path length distribution
        fig.add_trace(
            go.Histogram(
                x=[p.path_length for p in paths],
                nbinsx=15,
                marker=dict(color=self.color_scheme['feature'], opacity=0.7),
                name='Path Length',
                hovertemplate='Depth: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=3, col=1
        )

        # Plot 6: Feature co-occurrence network (scatter plot proxy)
        # For each pair of features that appear together in paths, plot their relationship
        feature_pairs = []
        for path in paths_subset:
            feats = path.split_features
            for i in range(len(feats) - 1):
                if i < len(feats) - 1:
                    feature_pairs.append((feats[i], feats[i+1], path.leaf_value))

        if feature_pairs:
            # Create a simple scatter showing feature sequence importance
            x_vals = []
            y_vals = []
            colors = []
            texts = []

            for f1, f2, value in feature_pairs[:30]:  # Limit to 30 for readability
                x_vals.append(feature_counts.get(f1, 0))
                y_vals.append(feature_counts.get(f2, 0))
                colors.append(value)
                texts.append(f"{f1} → {f2}")

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colors,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Contribution", x=1.15)
                    ),
                    text=texts,
                    hovertemplate='%{text}<br>F1 usage: %{x}<br>F2 usage: %{y}<extra></extra>',
                    name='Feature Pairs'
                ),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            title_text=f"Decision Path Analysis Across {len(paths_subset)} Trees",
            showlegend=False,
            height=1200,
            width=1600,
            template='plotly_white'
        )

        # Update axes
        fig.update_xaxes(title_text="Tree ID", row=1, col=1)
        fig.update_yaxes(title_text="Path Depth", row=1, col=1)

        fig.update_xaxes(title_text="Tree ID", row=1, col=2)
        fig.update_yaxes(title_text="Contribution", row=1, col=2)

        fig.update_xaxes(title_text="Usage Count", row=2, col=1)
        fig.update_yaxes(title_text="Feature", row=2, col=1)

        fig.update_xaxes(title_text="Tree ID", row=2, col=2)
        fig.update_yaxes(title_text="Feature", row=2, col=2)

        fig.update_xaxes(title_text="Path Depth", row=3, col=1)
        fig.update_yaxes(title_text="Frequency", row=3, col=1)

        fig.update_xaxes(title_text="First Feature Usage", row=3, col=2)
        fig.update_yaxes(title_text="Second Feature Usage", row=3, col=2)

        if output_path:
            fig.write_html(output_path)
            print(f"✓ Interactive decision path visualization saved to {output_path}")

        return fig

    def visualize_single_tree(
        self,
        tree_df: pd.DataFrame,
        tree_id: int,
        output_path: str = None
    ) -> go.Figure:
        """
        Visualize a single decision tree with hierarchical structure

        Args:
            tree_df: Trees dataframe from booster.trees_to_dataframe()
            tree_id: ID of tree to visualize
            output_path: Where to save HTML file

        Returns:
            Plotly figure object
        """
        # Filter for specific tree
        tree_data = tree_df[tree_df['Tree'] == tree_id].copy()

        if tree_data.empty:
            print(f"No data found for tree {tree_id}")
            return None

        # Build tree structure
        nodes = []
        edges_x = []
        edges_y = []
        node_texts = []
        node_colors = []

        # Parse tree structure
        for idx, row in tree_data.iterrows():
            node_id = row['Node']
            feature = row['Feature']
            split_val = row.get('Split', None)
            gain = row.get('Gain', 0)

            # Determine node type and color
            if feature == 'Leaf':
                node_type = 'Leaf'
                color = self.color_scheme['approval'] if split_val and split_val > 0 else self.color_scheme['rejection']
                label = f"Leaf {node_id}<br>Value: {split_val:.4f}" if split_val else f"Leaf {node_id}"
            else:
                node_type = 'Split'
                color = self.color_scheme['feature']
                label = f"{feature}<br>≤ {split_val:.2f}<br>Gain: {gain:.4f}" if split_val else f"{feature}"

            nodes.append({
                'id': node_id,
                'label': label,
                'type': node_type,
                'color': color
            })

        # Create tree layout (simple vertical layout)
        # This is a simplified version - for complex trees, consider using networkx or graphviz
        fig = go.Figure()

        # For now, create a simple tabular view
        node_ids = [n['id'] for n in nodes]
        node_labels = [n['label'] for n in nodes]
        node_colors_list = [n['color'] for n in nodes]
        node_types = [n['type'] for n in nodes]

        fig.add_trace(go.Bar(
            x=node_ids,
            y=[1] * len(node_ids),
            marker=dict(color=node_colors_list),
            text=node_labels,
            textposition='outside',
            hovertemplate='%{text}<extra></extra>',
            name='Tree Nodes'
        ))

        fig.update_layout(
            title=f"Decision Tree {tree_id} Structure",
            xaxis_title="Node ID",
            yaxis_title="",
            height=600,
            width=1200,
            showlegend=False,
            template='plotly_white'
        )

        fig.update_yaxes(showticklabels=False)

        if output_path:
            fig.write_html(output_path)
            print(f"✓ Tree {tree_id} visualization saved to {output_path}")

        return fig

    def visualize_feature_interactions_network(
        self,
        interactions: List,
        output_path: str = None,
        top_k: int = 20
    ) -> go.Figure:
        """
        Create interactive network visualization of feature interactions

        Args:
            interactions: List of FeatureInteraction objects
            output_path: Where to save HTML file
            top_k: Number of top interactions to show

        Returns:
            Plotly figure object
        """
        if not interactions:
            print("No interactions to visualize")
            return None

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Feature Interaction Strengths',
                'Interaction Type Distribution'
            ),
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            horizontal_spacing=0.15
        )

        # Plot 1: Interaction strengths
        top_interactions = interactions[:top_k]
        labels = [f"{i.feature_a} × {i.feature_b}" for i in top_interactions]
        strengths = [i.interaction_strength for i in top_interactions]

        # Color by interaction type
        color_map = {
            'sequential': '#3498db',
            'parallel': '#e67e22',
            'nested': '#2ecc71'
        }
        colors = [color_map.get(i.interaction_type, '#95a5a6') for i in top_interactions]

        fig.add_trace(
            go.Bar(
                y=labels,
                x=strengths,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{s:.3f}" for s in strengths],
                textposition='outside',
                hovertemplate='%{y}<br>Strength: %{x:.3f}<br>Type: %{customdata}<extra></extra>',
                customdata=[i.interaction_type for i in top_interactions],
                name='Interaction Strength'
            ),
            row=1, col=1
        )

        # Plot 2: Interaction type distribution
        type_counts = Counter([i.interaction_type for i in interactions])

        fig.add_trace(
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                marker=dict(colors=[color_map.get(t, '#95a5a6') for t in type_counts.keys()]),
                hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                name='Interaction Types'
            ),
            row=1, col=2
        )

        fig.update_layout(
            title_text=f"Feature Interaction Analysis (Top {top_k})",
            showlegend=True,
            height=700,
            width=1400,
            template='plotly_white'
        )

        fig.update_xaxes(title_text="Interaction Strength", row=1, col=1)
        fig.update_yaxes(title_text="Feature Pair", row=1, col=1)

        # Add legend for interaction types
        fig.add_annotation(
            text="<b>Interaction Types:</b><br>Sequential: Features used in sequence<br>Parallel: Features used at same depth<br>Nested: One feature nested in another's subtree",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10),
            align='left',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )

        if output_path:
            fig.write_html(output_path)
            print(f"✓ Feature interaction visualization saved to {output_path}")

        return fig

    def visualize_tree_ensemble_summary(
        self,
        tree_insights: List,
        output_path: str = None
    ) -> go.Figure:
        """
        Create comprehensive summary visualization of tree ensemble

        Args:
            tree_insights: List of TreeLevelInsight objects
            output_path: Where to save HTML file

        Returns:
            Plotly figure object
        """
        if not tree_insights:
            print("No tree insights to visualize")
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Tree Depths Distribution',
                'Number of Splits per Tree',
                'Dominant Features by Tree',
                'Leaf Value Statistics'
            ),
            specs=[
                [{"type": "box"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        tree_ids = [t.tree_id for t in tree_insights]
        max_depths = [t.max_depth for t in tree_insights]
        num_splits = [t.num_splits for t in tree_insights]
        leaf_means = [t.leaf_value_mean for t in tree_insights]
        leaf_stds = [t.leaf_value_std for t in tree_insights]

        # Plot 1: Tree depths box plot
        fig.add_trace(
            go.Box(
                y=max_depths,
                name='Max Depth',
                marker=dict(color=self.color_scheme['feature']),
                boxmean='sd',
                hovertemplate='Max Depth: %{y}<extra></extra>'
            ),
            row=1, col=1
        )

        # Plot 2: Number of splits scatter
        fig.add_trace(
            go.Scatter(
                x=tree_ids,
                y=num_splits,
                mode='markers+lines',
                marker=dict(size=8, color=self.color_scheme['interaction']),
                line=dict(width=1, dash='dot'),
                name='Num Splits',
                hovertemplate='Tree %{x}<br>Splits: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        # Plot 3: Dominant features
        dominant_features = [t.dominant_feature for t in tree_insights[:30]]  # First 30 trees
        feature_counts = Counter(dominant_features)
        top_dominant = dict(feature_counts.most_common(10))

        fig.add_trace(
            go.Bar(
                x=list(top_dominant.keys()),
                y=list(top_dominant.values()),
                marker=dict(color=self.color_scheme['approval']),
                text=list(top_dominant.values()),
                textposition='outside',
                name='Dominant Features',
                hovertemplate='%{x}<br>Times dominant: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

        # Plot 4: Leaf value statistics
        fig.add_trace(
            go.Scatter(
                x=tree_ids,
                y=leaf_means,
                error_y=dict(type='data', array=leaf_stds, visible=True),
                mode='markers',
                marker=dict(
                    size=8,
                    color=leaf_means,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Mean Value")
                ),
                name='Leaf Values',
                hovertemplate='Tree %{x}<br>Mean: %{y:.4f}<br>Std: %{error_y.array:.4f}<extra></extra>'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text=f"Tree Ensemble Summary ({len(tree_insights)} trees)",
            showlegend=False,
            height=1000,
            width=1400,
            template='plotly_white'
        )

        # Update axes
        fig.update_yaxes(title_text="Max Depth", row=1, col=1)

        fig.update_xaxes(title_text="Tree ID", row=1, col=2)
        fig.update_yaxes(title_text="Number of Splits", row=1, col=2)

        fig.update_xaxes(title_text="Feature", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        fig.update_xaxes(title_text="Tree ID", row=2, col=2)
        fig.update_yaxes(title_text="Leaf Value Mean", row=2, col=2)

        if output_path:
            fig.write_html(output_path)
            print(f"✓ Tree ensemble summary saved to {output_path}")

        return fig


# Convenience function to replace matplotlib visualizations
def create_all_interactive_visualizations(
    mechanistic_results: Dict[str, Any],
    output_dir: str = "figures/mechanistic_interactive"
) -> Dict[str, str]:
    """
    Create all interactive visualizations for mechanistic interpretability results

    Args:
        mechanistic_results: Dictionary containing paths, interactions, tree_insights
        output_dir: Directory to save HTML visualizations

    Returns:
        Dictionary mapping visualization type to file path
    """
    os.makedirs(output_dir, exist_ok=True)

    visualizer = TreeVisualizer()
    output_files = {}

    application_id = mechanistic_results.get('application_id', 'unknown')

    # 1. Decision paths
    if 'decision_paths' in mechanistic_results:
        output_path = f"{output_dir}/decision_paths_{application_id}.html"
        fig = visualizer.visualize_decision_paths_interactive(
            mechanistic_results['decision_paths'],
            output_path=output_path,
            top_k=30
        )
        if fig:
            output_files['decision_paths'] = output_path

    # 2. Feature interactions
    if 'feature_interactions' in mechanistic_results:
        output_path = f"{output_dir}/feature_interactions_{application_id}.html"
        fig = visualizer.visualize_feature_interactions_network(
            mechanistic_results['feature_interactions'],
            output_path=output_path,
            top_k=20
        )
        if fig:
            output_files['feature_interactions'] = output_path

    # 3. Tree ensemble summary
    if 'tree_insights' in mechanistic_results:
        output_path = f"{output_dir}/tree_ensemble_{application_id}.html"
        fig = visualizer.visualize_tree_ensemble_summary(
            mechanistic_results['tree_insights'],
            output_path=output_path
        )
        if fig:
            output_files['tree_ensemble'] = output_path

    print(f"\n✓ Created {len(output_files)} interactive visualizations in {output_dir}/")
    return output_files
