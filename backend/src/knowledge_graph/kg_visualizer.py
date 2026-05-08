"""
Knowledge Graph Visualizer for HOGE Framework

Provides interactive visualization of KG evidence used in explanations:
1. Evidence table (policy rules, counterfactuals, SHAP values)
2. Interactive network graph showing loan-centric subgraph
"""

import os
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from dotenv import load_dotenv
import graphviz

# Optional Neo4j import
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

load_dotenv()


class KGVisualizer:
    """
    Visualize Knowledge Graph evidence for a specific loan explanation
    """

    def __init__(self):
        """Initialize with Neo4j connection if available"""
        self.neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver = None
        if NEO4J_AVAILABLE and self.neo4j_password:
            try:
                self.driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password)
                )
            except Exception as e:
                print(f"Warning: Could not connect to Neo4j: {e}")

        self.color_scheme = {
            'loan': '#3498db',      # Blue
            'rule': '#e74c3c',      # Red
            'shap': '#2ecc71',      # Green
            'counterfactual': '#f39c12',  # Orange
            'feature': '#9b59b6',   # Purple
            'concept': '#1abc9c',   # Teal
        }

    def extract_evidence_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract KG evidence from explanation context

        Args:
            context: The context dictionary used to generate explanation

        Returns:
            Dictionary with categorized evidence
        """
        evidence = {
            'loan_id': context.get('application_id', 'Unknown'),
            'prediction': context.get('model_prediction', 'Unknown'),
            'probability': context.get('approval_probability', 0.0),
            'shap_values': [],
            'violated_rules': context.get('violated_rules', []),
            'all_policy_rules': context.get('all_policy_rules', []),
            'counterfactuals': context.get('counterfactual_scenarios', []),
            'top_features': [],
        }

        # Extract SHAP values
        shap_details = context.get('shap_details', [])
        shap_sorted = sorted(
            [s for s in shap_details if s.get('importance') is not None],
            key=lambda x: x['importance'],
            reverse=True
        )

        evidence['shap_values'] = shap_sorted[:10]  # Top 10
        evidence['top_features'] = [s['feature'] for s in shap_sorted[:5]]

        return evidence

    def create_evidence_table(self, evidence: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a structured table of all KG evidence

        Returns:
            DataFrame with columns: Type, ID, Description, Value, Impact
        """
        rows = []

        # SHAP values
        for shap in evidence.get('shap_values', []):
            rows.append({
                'Type': 'SHAP Value',
                'ID': shap.get('feature', 'N/A'),
                'Description': f"{shap.get('feature')} = {shap.get('value')}",
                'Value': f"{shap.get('shap', 0):.4f}",
                'Impact': shap.get('direction', 'neutral')
            })

        # Violated rules
        for rule in evidence.get('violated_rules', []):
            rows.append({
                'Type': 'Policy Rule (Violated)',
                'ID': rule.get('rule_id', 'N/A'),
                'Description': rule.get('description', 'N/A'),
                'Value': f"{rule.get('severity', 'N/A')}",
                'Impact': 'negative'
            })

        # Counterfactuals
        for cf in evidence.get('counterfactuals', []):
            impact = 'positive' if cf.get('probability_shift', 0) > 0 else 'negative'
            rows.append({
                'Type': 'Counterfactual',
                'ID': cf.get('feature', 'N/A'),
                'Description': f"Change from {cf.get('current_value')} to {cf.get('changed_to')}",
                'Value': f"Δ{cf.get('probability_shift', 0)*100:.1f}%",
                'Impact': impact
            })

        return pd.DataFrame(rows)

    def query_loan_subgraph(self, loan_id: str, max_depth: int = 2) -> Optional[Dict[str, Any]]:
        """
        Query Neo4j for the subgraph around a specific loan

        Args:
            loan_id: The loan application ID
            max_depth: Maximum relationship depth to traverse

        Returns:
            Dictionary with nodes and relationships
        """
        if not self.driver:
            return None

        query = """
        MATCH path = (l:LoanApplication {application_id: $loan_id})-[*0..2]-(connected)
        WITH l, collect(DISTINCT connected) as nodes, collect(DISTINCT relationships(path)) as rels
        UNWIND nodes as node
        WITH l, collect(DISTINCT {
            id: id(node),
            labels: labels(node),
            properties: properties(node)
        }) as node_list, rels
        UNWIND rels as rel_list
        UNWIND rel_list as rel
        RETURN
            node_list,
            collect(DISTINCT {
                source: id(startNode(rel)),
                target: id(endNode(rel)),
                type: type(rel),
                properties: properties(rel)
            }) as relationships
        """

        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(query, loan_id=loan_id)
                record = result.single()

                if record:
                    return {
                        'nodes': record['node_list'],
                        'relationships': record['relationships']
                    }
        except Exception as e:
            print(f"Error querying Neo4j: {e}")
            return None

        return None

    def create_network_from_context(self, evidence: Dict[str, Any]) -> go.Figure:
        """
        Create interactive network visualization from evidence (without Neo4j query)

        Args:
            evidence: Evidence dictionary from extract_evidence_from_context()

        Returns:
            Plotly figure with network graph
        """
        # Build networkx graph
        G = nx.Graph()

        loan_id = evidence['loan_id']

        # Central node: Loan
        G.add_node(loan_id,
                   node_type='loan',
                   label=f"Loan: {loan_id}",
                   prediction=evidence['prediction'],
                   probability=evidence['probability'])

        # Add SHAP value nodes (top 5)
        for shap in evidence.get('shap_values', [])[:5]:
            feature = shap['feature']
            node_id = f"SHAP_{feature}"
            G.add_node(node_id,
                       node_type='shap',
                       label=f"{feature}\nSHAP: {shap['shap']:.3f}",
                       value=shap['value'],
                       impact=shap['shap'])
            G.add_edge(loan_id, node_id,
                       edge_type='has_shap',
                       label='has_SHAP_value')

        # Add violated rules
        for rule in evidence.get('violated_rules', []):
            rule_id = rule['rule_id']
            G.add_node(rule_id,
                       node_type='rule',
                       label=f"{rule_id}\n{rule['severity']}",
                       description=rule['description'])
            G.add_edge(loan_id, rule_id,
                       edge_type='violates',
                       label='violates_rule')

        # Add counterfactuals
        for i, cf in enumerate(evidence.get('counterfactuals', [])[:3]):
            cf_id = f"CF_{i+1}_{cf['feature']}"
            G.add_node(cf_id,
                       node_type='counterfactual',
                       label=f"What-if: {cf['feature']}\n→ {cf['changed_to']}",
                       shift=cf.get('probability_shift', 0))
            G.add_edge(loan_id, cf_id,
                       edge_type='has_counterfactual',
                       label='counterfactual')

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Create Plotly traces
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=2, color='#95a5a6'),
            hoverinfo='text',
            mode='lines',
            text=[],
            showlegend=False
        )

        # Add edges
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
            edge_trace['text'] += tuple([edge[2].get('label', '')])

        # Node traces (one per type for legend)
        node_traces = {}

        for node, data in G.nodes(data=True):
            node_type = data.get('node_type', 'unknown')

            if node_type not in node_traces:
                node_traces[node_type] = go.Scatter(
                    x=[],
                    y=[],
                    mode='markers+text',
                    marker=dict(
                        size=[],
                        color=self.color_scheme.get(node_type, '#95a5a6'),
                        line=dict(width=2, color='white')
                    ),
                    text=[],
                    textposition='top center',
                    hovertext=[],
                    hoverinfo='text',
                    name=node_type.title(),
                    showlegend=True
                )

            x, y = pos[node]
            node_traces[node_type]['x'] += tuple([x])
            node_traces[node_type]['y'] += tuple([y])

            # Node size based on type
            size = 40 if node_type == 'loan' else 25
            node_traces[node_type]['marker']['size'] += tuple([size])

            # Node label
            label = data.get('label', node)
            node_traces[node_type]['text'] += tuple([label.split('\n')[0]])  # First line only

            # Hover text
            hover_lines = [f"<b>{label}</b>"]
            for key, val in data.items():
                if key not in ['node_type', 'label']:
                    hover_lines.append(f"{key}: {val}")
            node_traces[node_type]['hovertext'] += tuple(['<br>'.join(hover_lines)])

        # Create figure
        fig = go.Figure(
            data=[edge_trace] + list(node_traces.values()),
            layout=go.Layout(
                title=dict(
                    text=f"Knowledge Graph Evidence for {loan_id}",
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='rgba(240,240,240,0.5)',
                height=500
            )
        )

        return fig

    def create_full_network_from_neo4j(self, loan_id: str) -> Optional[go.Figure]:
        """
        Create network visualization by querying Neo4j directly

        Args:
            loan_id: The loan application ID

        Returns:
            Plotly figure or None if Neo4j unavailable
        """
        subgraph = self.query_loan_subgraph(loan_id)

        if not subgraph:
            return None

        # Build networkx graph from Neo4j results
        G = nx.Graph()

        # Add nodes
        node_id_map = {}
        for node in subgraph['nodes']:
            neo_id = node['id']
            labels = node['labels']
            props = node['properties']

            # Create readable label
            if 'LoanApplication' in labels:
                label = f"Loan: {props.get('application_id', neo_id)}"
            elif 'Feature' in labels:
                label = props.get('name', str(neo_id))
            elif 'PolicyRule' in labels:
                label = f"Rule: {props.get('rule_id', neo_id)}"
            else:
                label = labels[0] if labels else str(neo_id)

            node_type = labels[0].lower() if labels else 'unknown'

            G.add_node(neo_id,
                       node_type=node_type,
                       label=label,
                       **props)

            node_id_map[neo_id] = label

        # Add relationships
        for rel in subgraph['relationships']:
            G.add_edge(rel['source'], rel['target'],
                       edge_type=rel['type'],
                       **rel.get('properties', {}))

        # Use same visualization logic as create_network_from_context
        # (simplified for brevity - similar Plotly code)

        pos = nx.spring_layout(G, k=2, iterations=50)

        # Create edge trace
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=1, color='#95a5a6'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        # Node trace
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            marker=dict(
                size=20,
                color='#3498db',
                line=dict(width=2, color='white')
            ),
            text=[],
            textposition='top center',
            hovertext=[],
            hoverinfo='text'
        )

        for node, data in G.nodes(data=True):
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([data.get('label', str(node))])
            node_trace['hovertext'] += tuple([str(data)])

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"Neo4j Subgraph for {loan_id}",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
        )

        return fig

    def create_graphviz_network(self, evidence: Dict[str, Any]) -> graphviz.Digraph:
        """
        Create Graphviz network visualization from evidence

        Args:
            evidence: Evidence dictionary from extract_evidence_from_context()

        Returns:
            Graphviz Digraph object
        """
        dot = graphviz.Digraph(comment='Knowledge Graph Evidence')
        dot.attr(rankdir='TB', size='10,10')
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        dot.attr('edge', fontname='Arial', fontsize='10')

        loan_id = evidence['loan_id']
        prediction = evidence['prediction']
        probability = evidence['probability']

        # Central node: Loan Application
        dot.node(
            loan_id,
            label=f"{loan_id}\n{prediction}\n{probability:.1%}",
            fillcolor='#3498db',
            fontcolor='white',
            shape='ellipse',
            width='2.5',
            height='1.5',
            fontsize='14'
        )

        # Add SHAP value nodes (top 5)
        for i, shap in enumerate(evidence.get('shap_values', [])[:5]):
            feature = shap['feature']
            shap_val = shap['shap']
            feature_val = shap['value']

            node_id = f"shap_{i}"
            color = '#2ecc71' if shap_val > 0 else '#e74c3c'
            direction = '↑' if shap_val > 0 else '↓'

            dot.node(
                node_id,
                label=f"{feature}\nValue: {feature_val}\nSHAP: {shap_val:.3f} {direction}",
                fillcolor=color,
                fontcolor='white',
                fontsize='11'
            )

            dot.edge(
                loan_id,
                node_id,
                label='has_feature',
                color='#95a5a6'
            )

        # Add violated rules
        for i, rule in enumerate(evidence.get('violated_rules', [])):
            rule_id = f"rule_{i}"

            dot.node(
                rule_id,
                label=f"{rule['rule_id']}\n{rule['description']}\nSeverity: {rule['severity']}",
                fillcolor='#e74c3c',
                fontcolor='white',
                shape='box',
                fontsize='10'
            )

            dot.edge(
                loan_id,
                rule_id,
                label='violates',
                color='#e74c3c',
                style='dashed'
            )

        # Add counterfactuals (top 3)
        for i, cf in enumerate(evidence.get('counterfactuals', [])[:3]):
            cf_id = f"cf_{i}"
            feature = cf['feature']
            current = cf['current_value']
            changed = cf['changed_to']
            shift = cf.get('probability_shift', 0)

            dot.node(
                cf_id,
                label=f"What-if: {feature}\n{current} → {changed}\nΔP: {shift:+.3f}",
                fillcolor='#f39c12',
                fontcolor='white',
                fontsize='10'
            )

            dot.edge(
                loan_id,
                cf_id,
                label='counterfactual',
                color='#f39c12',
                style='dotted'
            )

        return dot

    def create_graphviz_from_neo4j(self, loan_id: str) -> Optional[graphviz.Digraph]:
        """
        Create Graphviz network by querying Neo4j directly

        Args:
            loan_id: The loan application ID

        Returns:
            Graphviz Digraph or None if Neo4j unavailable
        """
        subgraph = self.query_loan_subgraph(loan_id)

        if not subgraph:
            return None

        dot = graphviz.Digraph(comment=f'Knowledge Graph for {loan_id}')
        dot.attr(rankdir='TB', size='12,12')
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        dot.attr('edge', fontname='Arial', fontsize='9')

        # Add nodes
        for node in subgraph['nodes']:
            neo_id = str(node['id'])
            labels = node['labels']
            props = node['properties']

            # Determine node type and styling
            if 'LoanApplication' in labels:
                label = f"Loan: {props.get('application_id', neo_id)}"
                color = '#3498db'
                shape = 'ellipse'
            elif 'Feature' in labels:
                label = props.get('name', neo_id)
                color = '#9b59b6'
                shape = 'box'
            elif 'PolicyRule' in labels:
                label = f"{props.get('rule_id', neo_id)}\n{props.get('description', '')[:30]}..."
                color = '#e74c3c'
                shape = 'box'
            elif 'Counterfactual' in labels:
                label = f"What-if: {props.get('feature', '')}"
                color = '#f39c12'
                shape = 'box'
            else:
                label = labels[0] if labels else neo_id
                color = '#95a5a6'
                shape = 'box'

            dot.node(
                neo_id,
                label=label,
                fillcolor=color,
                fontcolor='white',
                shape=shape
            )

        # Add relationships
        for rel in subgraph['relationships']:
            source = str(rel['source'])
            target = str(rel['target'])
            rel_type = rel['type']

            dot.edge(
                source,
                target,
                label=rel_type,
                color='#95a5a6'
            )

        return dot

    def close(self):
        """Close Neo4j driver if open"""
        if self.driver:
            self.driver.close()
