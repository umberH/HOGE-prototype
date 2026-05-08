"""
Provenance Service
==================
Service for accessing system and instance-level provenance.
"""

import sys
from pathlib import Path
import os
import json
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from backend.src.api.models.common_dto import ProvenanceInfo


class ProvenanceService:
    """Service for provenance-related operations."""

    def __init__(self):
        """Initialize provenance service."""
        pass

    def load_model_provenance(self) -> Dict[str, Any]:
        """
        Load model training provenance.

        Returns:
            Model provenance dict
        """
        return self._load_provenance_file(".resources/models/model_provenance.json").get("model_provenance", {})

    def load_kg_provenance(self) -> Dict[str, Any]:
        """
        Load knowledge graph provenance.

        Returns:
            KG provenance dict
        """
        return self._load_provenance_file(".resources/data/provenance/kg_provenance.json").get("kg_provenance", {})

    def load_evaluation_provenance(self) -> Dict[str, Any]:
        """
        Load evaluation provenance.

        Returns:
            Evaluation provenance dict
        """
        return self._load_provenance_file(".resources/data/provenance/eval_provenance.json").get("evaluation_provenance", {})

    def load_all_system_provenance(self) -> Dict[str, Any]:
        """
        Load all system-level provenance.

        Returns:
            Dict with model, kg, and evaluation provenance
        """
        return {
            "model": self.load_model_provenance(),
            "knowledge_graph": self.load_kg_provenance(),
            "evaluation": self.load_evaluation_provenance(),
        }

    def _load_provenance_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load provenance JSON file if it exists.

        Args:
            filepath: Path to JSON file

        Returns:
            Provenance dict or empty dict if not found
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

        return {}
