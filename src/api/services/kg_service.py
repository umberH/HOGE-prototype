"""
Knowledge Graph Service
=======================
Service for Knowledge Graph operations (visualization, queries).
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.knowledge_graph.kg_visualizer import KGVisualizer


class KnowledgeGraphService:
    """Service for KG-related operations."""

    def __init__(self):
        """Initialize KG service."""
        self.visualizer = None

    def _get_visualizer(self) -> KGVisualizer:
        """Get or create KG visualizer."""
        if self.visualizer is None:
            self.visualizer = KGVisualizer()
        return self.visualizer

    def create_network_visualization(
        self,
        application_id: str,
        use_neo4j: bool = True
    ) -> Optional[Any]:
        """
        Create a network visualization for an application.

        Args:
            application_id: Application ID to visualize
            use_neo4j: Whether to try Neo4j first (fallback to context-based)

        Returns:
            Graphviz object for rendering, or None if failed
        """
        visualizer = self._get_visualizer()

        try:
            if use_neo4j:
                # Try Neo4j first
                graph_viz = visualizer.create_graphviz_from_neo4j(application_id)
                if graph_viz:
                    return graph_viz

            # Fallback: Would need context passed in
            # For now, return None if Neo4j fails
            return None

        except Exception as e:
            print(f"Error creating KG visualization: {e}")
            return None

    def create_network_from_context(
        self,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Create network visualization from explanation context.

        Args:
            context: Explanation context dict

        Returns:
            Graphviz object for rendering
        """
        visualizer = self._get_visualizer()

        try:
            evidence = visualizer.extract_evidence_from_context(context)
            graph_viz = visualizer.create_graphviz_network(evidence)
            return graph_viz

        except Exception as e:
            print(f"Error creating KG visualization from context: {e}")
            return None

    def close(self):
        """Close KG connections."""
        if self.visualizer:
            self.visualizer.close()
            self.visualizer = None
