"""
Generate Neo4j Knowledge Graph visualizations for the HOGE paper.
Produces two PNG images:
  1. neo4j_kg_schema.png    – ontology / schema-level view (node labels + relationship types)
  2. neo4j_kg_instance.png  – instance-level subgraph for a single loan application
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI",  "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "test1234")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ── colour palette per node label ──────────────────────────
LABEL_COLORS = {
    "Applicant":              "#4C72B0",
    "LoanApplication":        "#DD8452",
    "Feature":                "#55A868",
    "FeatureValue":           "#C44E52",
    "ModelExplanation":       "#8172B3",
    "FeatureContribution":    "#937860",
    "PolicyRule":             "#DA8BC3",
    "RiskFactor":             "#8C8C8C",
    "CounterfactualScenario": "#64B5CD",
}
DEFAULT_COLOR = "#CCB974"


# ════════════════════════════════════════════════════════════
# 1.  SCHEMA / ONTOLOGY VIEW
# ════════════════════════════════════════════════════════════
def generate_schema_view():
    """Query db.schema.visualization() or fall back to manual meta-query."""
    G = nx.DiGraph()

    with driver.session(database=NEO4J_DATABASE) as session:
        # Get all relationship types connecting label pairs
        result = session.run("""
            MATCH (a)-[r]->(b)
            WITH labels(a)[0] AS from_label, type(r) AS rel, labels(b)[0] AS to_label
            RETURN DISTINCT from_label, rel, to_label
        """)
        for rec in result:
            G.add_node(rec["from_label"])
            G.add_node(rec["to_label"])
            G.add_edge(rec["from_label"], rec["to_label"], label=rec["rel"])

    if len(G.nodes) == 0:
        print("⚠  No schema data found – is the KG loaded?")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=80)

    node_colors = [LABEL_COLORS.get(n, DEFAULT_COLOR) for n in G.nodes]
    node_sizes  = [3000 for _ in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.92, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    nx.draw_networkx_edges(G, pos, edge_color="#555555",
                           arrows=True, arrowsize=18,
                           connectionstyle="arc3,rad=0.1",
                           width=1.5, ax=ax)

    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                 font_size=7, font_color="#333333",
                                 label_pos=0.5, ax=ax)

    ax.set_title("HOGE Knowledge Graph – Ontology Schema", fontsize=14, fontweight="bold")
    ax.axis("off")

    patches = [mpatches.Patch(color=c, label=l) for l, c in LABEL_COLORS.items() if l in G.nodes]
    ax.legend(handles=patches, loc="lower left", fontsize=8, title="Node Labels")

    fig.tight_layout()
    fig.savefig("neo4j_kg_schema.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✔ Saved neo4j_kg_schema.png")


# ════════════════════════════════════════════════════════════
# 2.  INSTANCE-LEVEL SUBGRAPH FOR ONE APPLICATION
# ════════════════════════════════════════════════════════════
def generate_instance_view(app_id="LP001006"):
    """Visualise the full subgraph around a single loan application."""
    G = nx.DiGraph()

    with driver.session(database=NEO4J_DATABASE) as session:
        # Fetch all nodes and relationships within 2 hops of the application
        result = session.run("""
            MATCH path = (la:LoanApplication {application_id: $aid})-[*1..2]-(n)
            UNWIND relationships(path) AS r
            WITH startNode(r) AS s, r, endNode(r) AS e
            RETURN
              id(s)      AS sid, labels(s)[0] AS slabel,
              COALESCE(s.application_id, s.applicant_id, s.name,
                       s.explanation_id, s.rule_id, s.id,
                       s.feature_name, toString(id(s))) AS sname,
              type(r)    AS rel,
              id(e)      AS eid, labels(e)[0] AS elabel,
              COALESCE(e.application_id, e.applicant_id, e.name,
                       e.explanation_id, e.rule_id, e.id,
                       e.feature_name, toString(id(e))) AS ename
        """, aid=app_id)

        for rec in result:
            sid, slabel, sname = rec["sid"], rec["slabel"], rec["sname"]
            eid, elabel, ename = rec["eid"], rec["elabel"], rec["ename"]
            G.add_node(sid, label=slabel, name=_short(sname, slabel))
            G.add_node(eid, label=elabel, name=_short(ename, elabel))
            G.add_edge(sid, eid, label=rec["rel"])

    if len(G.nodes) == 0:
        print(f"⚠  No data for application {app_id}")
        return

    fig, ax = plt.subplots(figsize=(18, 14))

    # Use hierarchical layout: group by label
    pos = nx.spring_layout(G, k=1.8, seed=7, iterations=100)

    node_colors = [LABEL_COLORS.get(G.nodes[n].get("label", ""), DEFAULT_COLOR) for n in G.nodes]
    node_sizes  = [1800 if G.nodes[n].get("label") in ("LoanApplication", "Applicant", "ModelExplanation")
                   else 1000 for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.88, ax=ax)

    labels = {n: G.nodes[n].get("name", str(n)) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)

    nx.draw_networkx_edges(G, pos, edge_color="#888888",
                           arrows=True, arrowsize=12,
                           connectionstyle="arc3,rad=0.05",
                           width=1.0, ax=ax)

    # Only draw edge labels for a subset to avoid clutter
    edge_labels = {}
    seen_rels = set()
    for u, v, d in G.edges(data=True):
        rel = d["label"]
        if rel not in seen_rels:
            edge_labels[(u, v)] = rel
            seen_rels.add(rel)

    nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                 font_size=6, font_color="#444444", ax=ax)

    ax.set_title(f"HOGE Knowledge Graph – Instance View ({app_id})",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    patches = [mpatches.Patch(color=c, label=l)
               for l, c in LABEL_COLORS.items() if l in {G.nodes[n].get("label") for n in G.nodes}]
    ax.legend(handles=patches, loc="lower left", fontsize=8, title="Node Labels")

    fig.tight_layout()
    fig.savefig("neo4j_kg_instance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Saved neo4j_kg_instance.png")


def _short(name, label):
    """Shorten long node names for display."""
    if name is None:
        return "?"
    s = str(name)
    # Shorten FeatureContribution / FeatureValue / CounterfactualScenario ids
    if label in ("FeatureContribution", "FeatureValue", "CounterfactualScenario") and "_" in s:
        parts = s.split("_")
        # e.g. EXP_LP001006_DTI -> DTI
        if len(parts) >= 3:
            return parts[-1]
        if len(parts) == 2:
            return parts[-1]
    if len(s) > 20:
        return s[:18] + "…"
    return s


if __name__ == "__main__":
    print("Generating KG visualizations...")
    generate_schema_view()
    generate_instance_view("LP001006")
    driver.close()
    print("Done.")
