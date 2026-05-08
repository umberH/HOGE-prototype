"""
Evaluation DTOs
===============
Data models for evaluation metrics and results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class EvaluationMetrics:
    """Summary metrics for evaluation."""

    # Faithfulness
    avg_faithfulness: float = 0.0
    faithfulness_samples: int = 0

    # Hallucination & Grounding
    avg_semantic_fidelity: float = 0.0
    avg_hallucination_rate: float = 0.0
    avg_evidence_coverage: float = 0.0
    hallucination_samples: int = 0

    # Retrieval
    avg_precision_at_3: float = 0.0
    avg_precision_at_5: float = 0.0
    avg_precision_at_10: float = 0.0
    retrieval_samples: int = 0

    # Human evaluation
    avg_alignment: float = 0.0
    avg_traceability: float = 0.0
    human_samples: int = 0

    # Timestamp
    evaluation_timestamp: Optional[str] = None


@dataclass
class EvaluationResult:
    """Detailed evaluation results."""

    metrics: EvaluationMetrics
    faithfulness_details: List[Dict[str, Any]] = field(default_factory=list)
    hallucination_details: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_details: List[Dict[str, Any]] = field(default_factory=list)
    human_details: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def load_from_files() -> "EvaluationResult":
        """
        Load evaluation results from JSON files.
        Returns default empty result if files don't exist.
        """
        import os
        import json

        metrics = EvaluationMetrics()
        faithfulness_details = []
        hallucination_details = []
        retrieval_details = []
        human_details = []

        # Load faithfulness
        faith_path = ".resources/data/evaluation/eval_faithfulness.json"
        if os.path.exists(faith_path):
            with open(faith_path, 'r') as f:
                faith_data = json.load(f)
                results = faith_data.get("results", [])
                faithfulness_details = results
                scores = [r.get("faithfulness_score", 0) for r in results
                         if isinstance(r.get("faithfulness_score"), (int, float))]
                if scores:
                    metrics.avg_faithfulness = sum(scores) / len(scores)
                    metrics.faithfulness_samples = len(scores)

        # Load hallucination
        hall_path = ".resources/data/evaluation/eval_hallucination.json"
        if os.path.exists(hall_path):
            with open(hall_path, 'r') as f:
                hall_data = json.load(f)
                results = hall_data.get("results", [])
                hallucination_details = results

                sf_scores = [r.get("semantic_fidelity", 0) for r in results
                           if isinstance(r.get("semantic_fidelity"), (int, float))]
                hr_scores = [r.get("hallucination_rate", 0) for r in results
                           if isinstance(r.get("hallucination_rate"), (int, float))]
                ec_scores = [r.get("evidence_coverage", 0) for r in results
                           if isinstance(r.get("evidence_coverage"), (int, float))]

                if sf_scores:
                    metrics.avg_semantic_fidelity = sum(sf_scores) / len(sf_scores)
                if hr_scores:
                    metrics.avg_hallucination_rate = sum(hr_scores) / len(hr_scores)
                if ec_scores:
                    metrics.avg_evidence_coverage = sum(ec_scores) / len(ec_scores)
                metrics.hallucination_samples = len(results)

        # Load retrieval
        retr_path = ".resources/data/evaluation/eval_retrieval.json"
        if os.path.exists(retr_path):
            with open(retr_path, 'r') as f:
                retr_data = json.load(f)
                results = [r for r in retr_data.get("results", []) if r.get("kg_found")]
                retrieval_details = results

                if results:
                    metrics.avg_precision_at_3 = sum(r.get("precision_at_3", 0) for r in results) / len(results)
                    metrics.avg_precision_at_5 = sum(r.get("precision_at_5", 0) for r in results) / len(results)
                    metrics.avg_precision_at_10 = sum(r.get("precision_at_10", 0) for r in results) / len(results)
                    metrics.retrieval_samples = len(results)

        # Load human evaluation
        human_path = ".resources/data/evaluation/eval_human_results.json"
        if os.path.exists(human_path):
            with open(human_path, 'r') as f:
                human_data = json.load(f)
                summary = human_data.get("summary", {})
                human_details = human_data.get("results", [])

                metrics.avg_alignment = summary.get("avg_alignment_score", 0)
                metrics.avg_traceability = summary.get("avg_traceability_score", 0)
                metrics.human_samples = summary.get("total_samples", 0)

        # Try to get timestamp from combined file
        combined_path = ".resources/data/evaluation/eval_system_all.json"
        if os.path.exists(combined_path):
            with open(combined_path, 'r') as f:
                combined_data = json.load(f)
                metrics.evaluation_timestamp = combined_data.get("evaluation_timestamp")

        return EvaluationResult(
            metrics=metrics,
            faithfulness_details=faithfulness_details,
            hallucination_details=hallucination_details,
            retrieval_details=retrieval_details,
            human_details=human_details,
        )
