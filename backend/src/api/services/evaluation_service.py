"""
Evaluation Service
==================
Service for accessing evaluation metrics and results.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from backend.src.api.models.evaluation_dto import EvaluationMetrics, EvaluationResult


class EvaluationService:
    """Service for evaluation-related operations."""

    def __init__(self):
        """Initialize evaluation service."""
        pass

    def get_all_metrics(self) -> EvaluationResult:
        """
        Load all evaluation metrics from files.

        Returns:
            EvaluationResult with all metrics and details
        """
        # Delegate to DTO's static method
        return EvaluationResult.load_from_files()

    def get_summary_metrics(self) -> EvaluationMetrics:
        """
        Get summary metrics only (no detailed results).

        Returns:
            EvaluationMetrics with summary statistics
        """
        result = self.get_all_metrics()
        return result.metrics

    def get_faithfulness_details(self) -> list:
        """Get faithfulness evaluation details."""
        result = self.get_all_metrics()
        return result.faithfulness_details

    def get_hallucination_details(self) -> list:
        """Get hallucination evaluation details."""
        result = self.get_all_metrics()
        return result.hallucination_details

    def get_retrieval_details(self) -> list:
        """Get retrieval evaluation details."""
        result = self.get_all_metrics()
        return result.retrieval_details

    def get_human_evaluation_details(self) -> list:
        """Get human evaluation details."""
        result = self.get_all_metrics()
        return result.human_details
