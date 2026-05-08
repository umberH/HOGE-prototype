"""
Comprehensive Logging System for LLM Explanations

This module provides:
1. Structured logging of all LLM-generated explanations
2. Drift detection across model versions, audiences, and time
3. Comparison capabilities for A/B testing
4. Metadata tracking for auditability
5. Neo4j integration for semantic queries

HOGE Framework Integration:
- Provenance tracking for all explanations
- Evidence bundle preservation
- Semantic grounding verification
- Temporal drift detection
"""

import os
import json
import hashlib
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import Counter
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


@dataclass
class ExplanationMetrics:
    """Metrics extracted from an explanation for drift detection"""

    # Content metrics
    narrative_length: int
    summary_length: int
    num_positive_drivers: int
    num_negative_drivers: int
    num_policy_violations: int
    num_features_mentioned: int

    # Vocabulary metrics
    unique_words: int
    technical_terms_count: int

    # Feature coverage
    feature_coverage_ratio: float  # features_mentioned / total_features_available
    top_feature_mentioned: Optional[str]

    # Evidence grounding
    evidence_bundle_size: int
    citation_quality_score: float  # 0-1, based on how well features are cited

    # Concept usage (if applicable)
    uses_concepts: bool
    num_concepts_mentioned: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExplanationLogEntry:
    """Complete log entry for a single explanation generation"""

    # Identifiers
    log_id: str
    application_id: str
    timestamp: str

    # Model configuration
    llm_model: str
    llm_temperature: float
    audience: str
    used_concepts: bool

    # Input context
    input_context_hash: str  # Hash of input data for change detection
    model_prediction: str
    approval_probability: float
    num_input_features: int
    num_violated_rules: int

    # Generated explanation
    explanation: Dict[str, Any]  # Full structured output

    # Metrics
    metrics: ExplanationMetrics

    # Provenance
    provenance: Dict[str, Any]
    evidence_bundle: List[Dict[str, Any]]

    # Performance
    generation_time_ms: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result['metrics'] = self.metrics.to_dict()
        return result


class ExplanationLogger:
    """
    Comprehensive logger for LLM explanations with drift detection

    Features:
    - Stores all explanations in Neo4j for semantic queries
    - Tracks metrics over time for drift detection
    - Enables comparison between models, audiences, and versions
    - Provides audit trail for HOGE framework compliance
    """

    def __init__(self, driver=None):
        if driver is None:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
        else:
            self.driver = driver

        # Ensure schema is initialized
        self._initialize_schema()

    def _initialize_schema(self):
        """Create indexes and constraints for explanation logging"""
        with self.driver.session(database=NEO4J_DATABASE) as session:
            # Constraints
            try:
                session.run("""
                    CREATE CONSTRAINT explanation_log_id IF NOT EXISTS
                    FOR (e:ExplanationLog) REQUIRE e.log_id IS UNIQUE
                """)
            except Exception:
                pass  # Constraint may already exist

            # Indexes
            indexes = [
                "CREATE INDEX explanation_timestamp IF NOT EXISTS FOR (e:ExplanationLog) ON (e.timestamp)",
                "CREATE INDEX explanation_app_id IF NOT EXISTS FOR (e:ExplanationLog) ON (e.application_id)",
                "CREATE INDEX explanation_model IF NOT EXISTS FOR (e:ExplanationLog) ON (e.llm_model)",
                "CREATE INDEX explanation_audience IF NOT EXISTS FOR (e:ExplanationLog) ON (e.audience)",
            ]

            for index_query in indexes:
                try:
                    session.run(index_query)
                except Exception:
                    pass  # Index may already exist

    def _compute_context_hash(self, context: dict) -> str:
        """
        Compute hash of input context to detect when same input produces different output
        """
        # Include only the deterministic parts of context
        hashable = {
            "application_id": context.get("application_id"),
            "model_prediction": context.get("model_prediction"),
            "approval_probability": context.get("approval_probability"),
            "shap_details": sorted(
                [
                    (s.get("feature"), s.get("value"), s.get("shap"))
                    for s in context.get("shap_details", [])
                ],
                key=lambda x: x[0] if x[0] else ""
            ),
            "violated_rules": sorted(
                [r.get("rule_id") for r in context.get("violated_rules", [])],
            ),
        }

        json_str = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _extract_metrics(self, explanation: dict, context: dict) -> ExplanationMetrics:
        """Extract quantitative metrics from explanation for drift detection"""

        narrative = explanation.get("narrative", "")
        summary = explanation.get("summary", "")

        # Word analysis
        words = narrative.lower().split()
        unique_words = len(set(words))

        # Technical terms (example list - can be expanded)
        technical_terms = {
            'shap', 'feature', 'model', 'probability', 'prediction',
            'contribution', 'importance', 'coefficient', 'score',
            'algorithm', 'correlation', 'variance', 'threshold'
        }
        technical_count = sum(1 for word in words if word.strip('.,!?') in technical_terms)

        # Feature coverage
        features_mentioned = explanation.get("features_used", [])
        total_features = len(context.get("shap_details", []))
        coverage_ratio = len(features_mentioned) / total_features if total_features > 0 else 0.0

        # Citation quality: check if features_used actually appear in narrative
        if features_mentioned:
            citations_found = sum(
                1 for feat in features_mentioned
                if feat.lower() in narrative.lower()
            )
            citation_quality = citations_found / len(features_mentioned)
        else:
            citation_quality = 0.0

        # Concept usage
        uses_concepts = explanation.get("used_concepts", False)
        num_concepts = len(explanation.get("business_concepts", []))

        return ExplanationMetrics(
            narrative_length=len(narrative),
            summary_length=len(summary),
            num_positive_drivers=len(explanation.get("positive_drivers", [])),
            num_negative_drivers=len(explanation.get("negative_drivers", [])),
            num_policy_violations=len(explanation.get("policy_violations", [])),
            num_features_mentioned=len(features_mentioned),
            unique_words=unique_words,
            technical_terms_count=technical_count,
            feature_coverage_ratio=coverage_ratio,
            top_feature_mentioned=features_mentioned[0] if features_mentioned else None,
            evidence_bundle_size=len(explanation.get("evidence_bundle", [])),
            citation_quality_score=citation_quality,
            uses_concepts=uses_concepts,
            num_concepts_mentioned=num_concepts,
        )

    def log_explanation(
        self,
        context: dict,
        explanation: dict,
        llm_model: str,
        llm_temperature: float,
        audience: str,
        generation_time_ms: float,
        used_concepts: bool = False
    ) -> str:
        """
        Log a complete explanation with all metadata

        Returns:
            log_id: Unique identifier for this explanation log entry
        """

        # Generate unique log ID
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        log_id = f"expl_{context['application_id']}_{audience}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

        # Compute context hash
        context_hash = self._compute_context_hash(context)

        # Extract metrics
        metrics = self._extract_metrics(explanation, context)

        # Create log entry
        log_entry = ExplanationLogEntry(
            log_id=log_id,
            application_id=context["application_id"],
            timestamp=timestamp.isoformat(),
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            audience=audience,
            used_concepts=used_concepts,
            input_context_hash=context_hash,
            model_prediction=context.get("model_prediction", ""),
            approval_probability=context.get("approval_probability", 0.0),
            num_input_features=len(context.get("shap_details", [])),
            num_violated_rules=len(context.get("violated_rules", [])),
            explanation=explanation,
            metrics=metrics,
            provenance=explanation.get("provenance", {}),
            evidence_bundle=explanation.get("evidence_bundle", []),
            generation_time_ms=generation_time_ms,
        )

        # Store in Neo4j
        self._store_in_neo4j(log_entry)

        # Also store in JSON for backup/offline analysis
        self._store_in_json(log_entry)

        return log_id

    def _store_in_neo4j(self, log_entry: ExplanationLogEntry):
        """Store explanation log in Neo4j knowledge graph"""

        with self.driver.session(database=NEO4J_DATABASE) as session:
            # Create ExplanationLog node
            query = """
            MERGE (app:LoanApplication {application_id: $application_id})
            CREATE (log:ExplanationLog {
                log_id: $log_id,
                application_id: $application_id,
                timestamp: datetime($timestamp),
                llm_model: $llm_model,
                llm_temperature: $llm_temperature,
                audience: $audience,
                used_concepts: $used_concepts,
                input_context_hash: $input_context_hash,
                model_prediction: $model_prediction,
                approval_probability: $approval_probability,
                num_input_features: $num_input_features,
                num_violated_rules: $num_violated_rules,

                // Explanation content
                summary: $summary,
                narrative: $narrative,
                recommendation: $recommendation,
                counterfactual_what_if: $counterfactual_what_if,

                // Metrics
                narrative_length: $narrative_length,
                summary_length: $summary_length,
                num_positive_drivers: $num_positive_drivers,
                num_negative_drivers: $num_negative_drivers,
                num_policy_violations: $num_policy_violations,
                num_features_mentioned: $num_features_mentioned,
                unique_words: $unique_words,
                technical_terms_count: $technical_terms_count,
                feature_coverage_ratio: $feature_coverage_ratio,
                top_feature_mentioned: $top_feature_mentioned,
                evidence_bundle_size: $evidence_bundle_size,
                citation_quality_score: $citation_quality_score,
                uses_concepts: $uses_concepts,
                num_concepts_mentioned: $num_concepts_mentioned,

                // Performance
                generation_time_ms: $generation_time_ms,

                // Full data (JSON stringified for complex structures)
                positive_drivers_json: $positive_drivers_json,
                negative_drivers_json: $negative_drivers_json,
                policy_violations_json: $policy_violations_json,
                features_used_json: $features_used_json,
                provenance_json: $provenance_json
            })
            CREATE (app)-[:HAS_EXPLANATION_LOG]->(log)

            // Link to features that were mentioned
            WITH log
            UNWIND $features_used AS feature_name
            MATCH (f:Feature {name: feature_name})
            MERGE (log)-[:MENTIONS_FEATURE]->(f)

            RETURN log.log_id as log_id
            """

            params = {
                "log_id": log_entry.log_id,
                "application_id": log_entry.application_id,
                "timestamp": log_entry.timestamp,
                "llm_model": log_entry.llm_model,
                "llm_temperature": log_entry.llm_temperature,
                "audience": log_entry.audience,
                "used_concepts": log_entry.used_concepts,
                "input_context_hash": log_entry.input_context_hash,
                "model_prediction": log_entry.model_prediction,
                "approval_probability": log_entry.approval_probability,
                "num_input_features": log_entry.num_input_features,
                "num_violated_rules": log_entry.num_violated_rules,

                # Explanation content
                "summary": log_entry.explanation.get("summary", ""),
                "narrative": log_entry.explanation.get("narrative", ""),
                "recommendation": log_entry.explanation.get("recommendation", ""),
                "counterfactual_what_if": log_entry.explanation.get("counterfactual_what_if", ""),

                # Metrics
                "narrative_length": log_entry.metrics.narrative_length,
                "summary_length": log_entry.metrics.summary_length,
                "num_positive_drivers": log_entry.metrics.num_positive_drivers,
                "num_negative_drivers": log_entry.metrics.num_negative_drivers,
                "num_policy_violations": log_entry.metrics.num_policy_violations,
                "num_features_mentioned": log_entry.metrics.num_features_mentioned,
                "unique_words": log_entry.metrics.unique_words,
                "technical_terms_count": log_entry.metrics.technical_terms_count,
                "feature_coverage_ratio": log_entry.metrics.feature_coverage_ratio,
                "top_feature_mentioned": log_entry.metrics.top_feature_mentioned,
                "evidence_bundle_size": log_entry.metrics.evidence_bundle_size,
                "citation_quality_score": log_entry.metrics.citation_quality_score,
                "uses_concepts": log_entry.metrics.uses_concepts,
                "num_concepts_mentioned": log_entry.metrics.num_concepts_mentioned,

                # Performance
                "generation_time_ms": log_entry.generation_time_ms,

                # JSON fields
                "positive_drivers_json": json.dumps(log_entry.explanation.get("positive_drivers", [])),
                "negative_drivers_json": json.dumps(log_entry.explanation.get("negative_drivers", [])),
                "policy_violations_json": json.dumps(log_entry.explanation.get("policy_violations", [])),
                "features_used_json": json.dumps(log_entry.explanation.get("features_used", [])),
                "features_used": log_entry.explanation.get("features_used", []),
                "provenance_json": json.dumps(log_entry.provenance),
            }

            session.run(query, params)

    def _store_in_json(self, log_entry: ExplanationLogEntry):
        """Store explanation log as JSON file for backup/offline analysis"""

        # Create directory if it doesn't exist
        log_dir = "data/explanation_logs"
        os.makedirs(log_dir, exist_ok=True)

        # Save to timestamped file
        filename = f"{log_dir}/{log_entry.log_id}.json"
        with open(filename, "w") as f:
            json.dump(log_entry.to_dict(), f, indent=2, default=str)

    def get_explanation_history(
        self,
        application_id: Optional[str] = None,
        audience: Optional[str] = None,
        llm_model: Optional[str] = None,
        limit: int = 100
    ) -> List[dict]:
        """
        Retrieve explanation history with optional filters
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            where_clauses = []
            params = {"limit": limit}

            if application_id:
                where_clauses.append("log.application_id = $application_id")
                params["application_id"] = application_id

            if audience:
                where_clauses.append("log.audience = $audience")
                params["audience"] = audience

            if llm_model:
                where_clauses.append("log.llm_model = $llm_model")
                params["llm_model"] = llm_model

            where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

            query = f"""
            MATCH (log:ExplanationLog)
            {where_clause}
            RETURN log
            ORDER BY log.timestamp DESC
            LIMIT $limit
            """

            result = session.run(query, params)
            return [dict(record["log"]) for record in result]

    def detect_explanation_drift(
        self,
        application_id: str,
        audience: str,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Detect drift in explanations for the same application/audience over time

        Returns metrics showing how explanations have changed:
        - Narrative length drift
        - Feature coverage drift
        - Vocabulary drift
        - Citation quality drift
        """

        history = self.get_explanation_history(
            application_id=application_id,
            audience=audience,
            limit=window_size
        )

        if len(history) < 2:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 2 explanations, found {len(history)}"
            }

        # Extract metric time series
        narrative_lengths = [h["narrative_length"] for h in history]
        feature_coverages = [h["feature_coverage_ratio"] for h in history]
        unique_words = [h["unique_words"] for h in history]
        citation_qualities = [h["citation_quality_score"] for h in history]
        technical_terms = [h["technical_terms_count"] for h in history]

        # Compute drift metrics (coefficient of variation)
        def coeff_variation(values):
            if not values or np.mean(values) == 0:
                return 0.0
            return np.std(values) / np.mean(values)

        return {
            "status": "success",
            "num_explanations": len(history),
            "time_range": {
                "earliest": history[-1]["timestamp"],
                "latest": history[0]["timestamp"],
            },
            "drift_metrics": {
                "narrative_length": {
                    "mean": float(np.mean(narrative_lengths)),
                    "std": float(np.std(narrative_lengths)),
                    "coefficient_of_variation": float(coeff_variation(narrative_lengths)),
                    "trend": "increasing" if narrative_lengths[0] > narrative_lengths[-1] else "decreasing"
                },
                "feature_coverage": {
                    "mean": float(np.mean(feature_coverages)),
                    "std": float(np.std(feature_coverages)),
                    "coefficient_of_variation": float(coeff_variation(feature_coverages)),
                    "trend": "increasing" if feature_coverages[0] > feature_coverages[-1] else "decreasing"
                },
                "vocabulary_diversity": {
                    "mean": float(np.mean(unique_words)),
                    "std": float(np.std(unique_words)),
                    "coefficient_of_variation": float(coeff_variation(unique_words)),
                },
                "citation_quality": {
                    "mean": float(np.mean(citation_qualities)),
                    "std": float(np.std(citation_qualities)),
                    "coefficient_of_variation": float(coeff_variation(citation_qualities)),
                },
                "technical_terms": {
                    "mean": float(np.mean(technical_terms)),
                    "std": float(np.std(technical_terms)),
                }
            },
            "alerts": self._generate_drift_alerts(
                narrative_lengths, feature_coverages, citation_qualities
            )
        }

    def _generate_drift_alerts(
        self,
        narrative_lengths: List[int],
        feature_coverages: List[float],
        citation_qualities: List[float]
    ) -> List[str]:
        """Generate alerts for significant drift"""
        alerts = []

        # Alert if narrative length varies by >50%
        if len(narrative_lengths) >= 2:
            length_cv = np.std(narrative_lengths) / np.mean(narrative_lengths) if np.mean(narrative_lengths) > 0 else 0
            if length_cv > 0.5:
                alerts.append(f"HIGH narrative length variation (CV={length_cv:.2f})")

        # Alert if feature coverage drops below 50%
        if any(fc < 0.5 for fc in feature_coverages):
            alerts.append("LOW feature coverage detected in some explanations (<50%)")

        # Alert if citation quality drops below 0.7
        if any(cq < 0.7 for cq in citation_qualities):
            alerts.append("LOW citation quality detected (features not properly referenced)")

        return alerts

    def compare_audiences(
        self,
        application_id: str,
        audiences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare explanations across different audience types for same application
        """
        if audiences is None:
            audiences = ["technical", "non_technical", "executive", "business"]

        comparisons = {}

        for audience in audiences:
            history = self.get_explanation_history(
                application_id=application_id,
                audience=audience,
                limit=5  # Get most recent 5
            )

            if history:
                latest = history[0]
                comparisons[audience] = {
                    "timestamp": latest["timestamp"],
                    "narrative_length": latest["narrative_length"],
                    "technical_terms_count": latest["technical_terms_count"],
                    "feature_coverage_ratio": latest["feature_coverage_ratio"],
                    "num_features_mentioned": latest["num_features_mentioned"],
                    "citation_quality_score": latest["citation_quality_score"],
                    "used_concepts": latest.get("uses_concepts", False),
                    "summary": latest["summary"][:200] + "..." if len(latest["summary"]) > 200 else latest["summary"],
                }

        return {
            "application_id": application_id,
            "audiences_compared": list(comparisons.keys()),
            "comparisons": comparisons,
            "insights": self._generate_audience_insights(comparisons)
        }

    def _generate_audience_insights(self, comparisons: Dict[str, dict]) -> List[str]:
        """Generate insights from audience comparison"""
        insights = []

        if len(comparisons) < 2:
            return ["Need at least 2 audiences to compare"]

        # Compare technical vs non-technical
        if "technical" in comparisons and "non_technical" in comparisons:
            tech_terms_tech = comparisons["technical"]["technical_terms_count"]
            tech_terms_non_tech = comparisons["non_technical"]["technical_terms_count"]

            if tech_terms_non_tech > tech_terms_tech * 0.5:
                insights.append(
                    f"WARNING: Non-technical explanation uses {tech_terms_non_tech} technical terms "
                    f"(should be much lower than technical's {tech_terms_tech})"
                )
            else:
                insights.append(
                    f"GOOD: Non-technical explanation properly reduces jargon "
                    f"({tech_terms_non_tech} vs {tech_terms_tech} technical terms)"
                )

        # Check if executive uses concepts
        if "executive" in comparisons:
            if not comparisons["executive"]["used_concepts"]:
                insights.append("INFO: Executive explanation could benefit from concept mapping")

        # Compare lengths
        lengths = {aud: comp["narrative_length"] for aud, comp in comparisons.items()}
        if lengths:
            max_aud = max(lengths, key=lengths.get)
            min_aud = min(lengths, key=lengths.get)
            insights.append(
                f"Length variation: {max_aud} is longest ({lengths[max_aud]} chars), "
                f"{min_aud} is shortest ({lengths[min_aud]} chars)"
            )

        return insights

    def compare_models(
        self,
        application_id: str,
        audience: str = "technical"
    ) -> Dict[str, Any]:
        """
        Compare explanations from different LLM models for same application
        """
        history = self.get_explanation_history(
            application_id=application_id,
            audience=audience,
            limit=50
        )

        # Group by model
        by_model = {}
        for entry in history:
            model = entry["llm_model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(entry)

        model_stats = {}
        for model, entries in by_model.items():
            model_stats[model] = {
                "count": len(entries),
                "avg_narrative_length": float(np.mean([e["narrative_length"] for e in entries])),
                "avg_citation_quality": float(np.mean([e["citation_quality_score"] for e in entries])),
                "avg_feature_coverage": float(np.mean([e["feature_coverage_ratio"] for e in entries])),
                "avg_generation_time_ms": float(np.mean([e["generation_time_ms"] for e in entries])),
            }

        return {
            "application_id": application_id,
            "audience": audience,
            "models_compared": list(model_stats.keys()),
            "model_statistics": model_stats,
        }

    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()
