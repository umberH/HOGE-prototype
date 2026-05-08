"""
Explanation Analysis and Comparison Utilities

Provides advanced analysis capabilities for logged LLM explanations:
- Multi-dimensional drift analysis
- A/B testing between models and audiences
- Temporal trend analysis
- Quality assurance metrics
- Semantic similarity detection
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class ExplanationAnalyzer:
    """
    Advanced analysis of logged explanations for quality assurance,
    drift detection, and comparative evaluation
    """

    def __init__(self, driver=None):
        if driver is None:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
        else:
            self.driver = driver

    def analyze_temporal_drift(
        self,
        days: int = 7,
        audience: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Analyze how explanations have drifted over time

        Args:
            days: Number of days to look back
            audience: Optional filter by audience type

        Returns:
            Comprehensive drift analysis with trends and alerts
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            # Get explanations from the past N days
            cutoff = datetime.now() - timedelta(days=days)

            query = """
            MATCH (log:ExplanationLog)
            WHERE log.timestamp >= datetime($cutoff)
            """

            params = {"cutoff": cutoff.isoformat()}

            if audience:
                query += " AND log.audience = $audience"
                params["audience"] = audience

            query += """
            RETURN log.timestamp as timestamp,
                   log.llm_model as model,
                   log.audience as audience,
                   log.narrative_length as narrative_length,
                   log.feature_coverage_ratio as feature_coverage,
                   log.citation_quality_score as citation_quality,
                   log.technical_terms_count as technical_terms,
                   log.generation_time_ms as generation_time
            ORDER BY log.timestamp ASC
            """

            result = session.run(query, params)
            records = [dict(r) for r in result]

        if not records:
            return {
                "status": "no_data",
                "message": f"No explanations found in the past {days} days"
            }

        # Convert to time series
        timestamps = [r["timestamp"] for r in records]
        narrative_lengths = [r["narrative_length"] for r in records]
        feature_coverages = [r["feature_coverage"] for r in records]
        citation_qualities = [r["citation_quality"] for r in records]
        technical_terms = [r["technical_terms"] for r in records]
        generation_times = [r["generation_time"] for r in records]

        # Compute trends (simple linear regression slope)
        def compute_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)

        return {
            "status": "success",
            "period_days": days,
            "num_explanations": len(records),
            "date_range": {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat()
            },
            "trends": {
                "narrative_length": {
                    "current_avg": float(np.mean(narrative_lengths[-5:])),
                    "historical_avg": float(np.mean(narrative_lengths)),
                    "trend_slope": compute_trend(narrative_lengths),
                    "direction": "increasing" if compute_trend(narrative_lengths) > 0 else "decreasing"
                },
                "feature_coverage": {
                    "current_avg": float(np.mean(feature_coverages[-5:])),
                    "historical_avg": float(np.mean(feature_coverages)),
                    "trend_slope": compute_trend(feature_coverages),
                    "direction": "increasing" if compute_trend(feature_coverages) > 0 else "decreasing"
                },
                "citation_quality": {
                    "current_avg": float(np.mean(citation_qualities[-5:])),
                    "historical_avg": float(np.mean(citation_qualities)),
                    "trend_slope": compute_trend(citation_qualities),
                    "direction": "increasing" if compute_trend(citation_qualities) > 0 else "decreasing"
                },
                "technical_terms": {
                    "current_avg": float(np.mean(technical_terms[-5:])),
                    "historical_avg": float(np.mean(technical_terms)),
                    "trend_slope": compute_trend(technical_terms),
                    "direction": "increasing" if compute_trend(technical_terms) > 0 else "decreasing"
                },
                "generation_time_ms": {
                    "current_avg": float(np.mean(generation_times[-5:])),
                    "historical_avg": float(np.mean(generation_times)),
                    "trend_slope": compute_trend(generation_times),
                }
            },
            "alerts": self._generate_temporal_alerts(records)
        }

    def _generate_temporal_alerts(self, records: List[dict]) -> List[str]:
        """Generate alerts for temporal drift issues"""
        alerts = []

        if len(records) < 5:
            return ["Insufficient data for temporal analysis"]

        # Check for sudden quality drops
        recent_quality = np.mean([r["citation_quality"] for r in records[-5:]])
        historical_quality = np.mean([r["citation_quality"] for r in records[:-5]])

        if recent_quality < historical_quality * 0.8:
            alerts.append(
                f"ALERT: Citation quality dropped by {(1 - recent_quality/historical_quality)*100:.1f}% "
                f"in recent explanations"
            )

        # Check for performance degradation
        recent_time = np.mean([r["generation_time"] for r in records[-5:]])
        historical_time = np.mean([r["generation_time"] for r in records[:-5]])

        if recent_time > historical_time * 1.5:
            alerts.append(
                f"ALERT: Generation time increased by {(recent_time/historical_time - 1)*100:.1f}% "
                f"(avg {recent_time:.0f}ms vs {historical_time:.0f}ms)"
            )

        return alerts if alerts else ["No significant temporal drift detected"]

    def compare_model_versions(
        self,
        model1: str,
        model2: str,
        audience: str = "technical",
        sample_size: int = 50
    ) -> Dict[str, any]:
        """
        A/B comparison between two different LLM models

        Args:
            model1: First model name (e.g., "gpt-4")
            model2: Second model name (e.g., "gpt-4o")
            audience: Audience type to compare
            sample_size: Number of samples per model

        Returns:
            Statistical comparison with significance tests
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            query = """
            MATCH (log:ExplanationLog)
            WHERE log.llm_model = $model AND log.audience = $audience
            RETURN log.narrative_length as narrative_length,
                   log.feature_coverage_ratio as feature_coverage,
                   log.citation_quality_score as citation_quality,
                   log.technical_terms_count as technical_terms,
                   log.generation_time_ms as generation_time,
                   log.num_features_mentioned as features_mentioned
            ORDER BY log.timestamp DESC
            LIMIT $sample_size
            """

            # Get samples for both models
            model1_data = list(session.run(
                query,
                {"model": model1, "audience": audience, "sample_size": sample_size}
            ))

            model2_data = list(session.run(
                query,
                {"model": model2, "audience": audience, "sample_size": sample_size}
            ))

        if not model1_data or not model2_data:
            return {
                "status": "insufficient_data",
                "model1_samples": len(model1_data),
                "model2_samples": len(model2_data)
            }

        # Extract metrics
        metrics = [
            "narrative_length", "feature_coverage", "citation_quality",
            "technical_terms", "generation_time", "features_mentioned"
        ]

        comparison = {}
        for metric in metrics:
            m1_values = [r[metric] for r in model1_data if r[metric] is not None]
            m2_values = [r[metric] for r in model2_data if r[metric] is not None]

            if not m1_values or not m2_values:
                continue

            m1_mean = float(np.mean(m1_values))
            m2_mean = float(np.mean(m2_values))

            # Simple effect size (Cohen's d approximation)
            pooled_std = np.sqrt((np.var(m1_values) + np.var(m2_values)) / 2)
            effect_size = (m2_mean - m1_mean) / pooled_std if pooled_std > 0 else 0

            comparison[metric] = {
                f"{model1}_mean": m1_mean,
                f"{model2}_mean": m2_mean,
                "difference": m2_mean - m1_mean,
                "percent_change": ((m2_mean - m1_mean) / m1_mean * 100) if m1_mean != 0 else 0,
                "effect_size": float(effect_size),
                "better_model": model2 if m2_mean > m1_mean else model1
            }

        return {
            "status": "success",
            "model1": model1,
            "model2": model2,
            "audience": audience,
            "sample_sizes": {
                model1: len(model1_data),
                model2: len(model2_data)
            },
            "comparison": comparison,
            "recommendation": self._generate_model_recommendation(comparison, model1, model2)
        }

    def _generate_model_recommendation(
        self,
        comparison: dict,
        model1: str,
        model2: str
    ) -> str:
        """Generate recommendation based on model comparison"""

        # Score each model
        model1_score = 0
        model2_score = 0

        # Citation quality is most important
        if "citation_quality" in comparison:
            if comparison["citation_quality"]["better_model"] == model1:
                model1_score += 3
            else:
                model2_score += 3

        # Feature coverage
        if "feature_coverage" in comparison:
            if comparison["feature_coverage"]["better_model"] == model1:
                model1_score += 2
            else:
                model2_score += 2

        # Generation time (lower is better, so invert)
        if "generation_time" in comparison:
            if comparison["generation_time"][f"{model1}_mean"] < comparison["generation_time"][f"{model2}_mean"]:
                model1_score += 1
            else:
                model2_score += 1

        if model1_score > model2_score:
            return f"Recommend {model1} - better overall quality metrics"
        elif model2_score > model1_score:
            return f"Recommend {model2} - better overall quality metrics"
        else:
            return "Models perform similarly - choose based on cost/speed preferences"

    def detect_consistency_issues(
        self,
        application_id: str,
        threshold: float = 0.3
    ) -> Dict[str, any]:
        """
        Detect when same application gets inconsistent explanations

        Args:
            application_id: Application to analyze
            threshold: Allowed variation (as fraction, e.g., 0.3 = 30%)

        Returns:
            Analysis of consistency issues
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            query = """
            MATCH (log:ExplanationLog {application_id: $application_id})
            RETURN log.input_context_hash as context_hash,
                   log.audience as audience,
                   log.llm_model as model,
                   log.timestamp as timestamp,
                   log.narrative_length as narrative_length,
                   log.features_used_json as features_used_json,
                   log.summary as summary
            ORDER BY log.timestamp DESC
            """

            result = session.run(query, {"application_id": application_id})
            records = [dict(r) for r in result]

        if len(records) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 explanations for consistency analysis"
            }

        # Group by context hash and audience (should be consistent within group)
        groups = defaultdict(list)
        for record in records:
            key = (record["context_hash"], record["audience"])
            groups[key].append(record)

        issues = []

        for (context_hash, audience), group_records in groups.items():
            if len(group_records) < 2:
                continue

            # Check narrative length consistency
            lengths = [r["narrative_length"] for r in group_records]
            length_cv = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0

            if length_cv > threshold:
                issues.append({
                    "type": "narrative_length_inconsistency",
                    "audience": audience,
                    "coefficient_of_variation": float(length_cv),
                    "threshold": threshold,
                    "samples": len(group_records),
                    "message": f"Narrative length varies by {length_cv*100:.1f}% (threshold: {threshold*100:.0f}%)"
                })

            # Check feature usage consistency
            try:
                feature_sets = [
                    set(json.loads(r["features_used_json"]))
                    for r in group_records
                ]

                # Jaccard similarity between consecutive explanations
                similarities = []
                for i in range(len(feature_sets) - 1):
                    intersection = len(feature_sets[i] & feature_sets[i+1])
                    union = len(feature_sets[i] | feature_sets[i+1])
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)

                avg_similarity = np.mean(similarities) if similarities else 1.0

                if avg_similarity < (1 - threshold):
                    issues.append({
                        "type": "feature_coverage_inconsistency",
                        "audience": audience,
                        "avg_jaccard_similarity": float(avg_similarity),
                        "threshold": 1 - threshold,
                        "samples": len(group_records),
                        "message": f"Feature coverage only {avg_similarity*100:.1f}% consistent across explanations"
                    })
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "status": "success",
            "application_id": application_id,
            "total_explanations": len(records),
            "unique_contexts": len(groups),
            "issues_found": len(issues),
            "issues": issues,
            "verdict": "PASS" if len(issues) == 0 else "FAIL"
        }

    def get_feature_popularity_trends(
        self,
        days: int = 30,
        top_n: int = 10
    ) -> Dict[str, any]:
        """
        Analyze which features are most commonly mentioned in explanations
        and track their popularity over time
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            cutoff = datetime.now() - timedelta(days=days)

            query = """
            MATCH (log:ExplanationLog)-[:MENTIONS_FEATURE]->(f:Feature)
            WHERE log.timestamp >= datetime($cutoff)
            RETURN f.name as feature_name,
                   count(*) as mention_count,
                   avg(log.citation_quality_score) as avg_citation_quality,
                   collect(DISTINCT log.audience) as audiences
            ORDER BY mention_count DESC
            LIMIT $top_n
            """

            result = session.run(query, {"cutoff": cutoff.isoformat(), "top_n": top_n})
            features = [dict(r) for r in result]

        return {
            "status": "success",
            "period_days": days,
            "top_features": [
                {
                    "feature": f["feature_name"],
                    "mentions": f["mention_count"],
                    "avg_citation_quality": float(f["avg_citation_quality"]) if f["avg_citation_quality"] else 0.0,
                    "audiences": f["audiences"]
                }
                for f in features
            ]
        }

    def generate_quality_report(
        self,
        days: int = 7
    ) -> Dict[str, any]:
        """
        Generate comprehensive quality assurance report for explanations
        """
        temporal_drift = self.analyze_temporal_drift(days=days)
        feature_trends = self.get_feature_popularity_trends(days=days)

        # Get overall statistics
        with self.driver.session(database=NEO4J_DATABASE) as session:
            cutoff = datetime.now() - timedelta(days=days)

            query = """
            MATCH (log:ExplanationLog)
            WHERE log.timestamp >= datetime($cutoff)
            RETURN
                count(*) as total_explanations,
                avg(log.citation_quality_score) as avg_citation_quality,
                avg(log.feature_coverage_ratio) as avg_feature_coverage,
                avg(log.generation_time_ms) as avg_generation_time,
                collect(DISTINCT log.audience) as audiences_used,
                collect(DISTINCT log.llm_model) as models_used
            """

            result = session.run(query, {"cutoff": cutoff.isoformat()})
            stats = dict(result.single())

        return {
            "report_date": datetime.now().isoformat(),
            "period_days": days,
            "overview": {
                "total_explanations": stats["total_explanations"],
                "avg_citation_quality": float(stats["avg_citation_quality"]) if stats["avg_citation_quality"] else 0.0,
                "avg_feature_coverage": float(stats["avg_feature_coverage"]) if stats["avg_feature_coverage"] else 0.0,
                "avg_generation_time_ms": float(stats["avg_generation_time"]) if stats["avg_generation_time"] else 0.0,
                "audiences_used": stats["audiences_used"],
                "models_used": stats["models_used"]
            },
            "temporal_drift": temporal_drift,
            "feature_trends": feature_trends,
            "quality_grade": self._compute_quality_grade(stats, temporal_drift)
        }

    def _compute_quality_grade(self, stats: dict, drift: dict) -> str:
        """Compute overall quality grade"""
        citation_quality = stats.get("avg_citation_quality", 0) or 0
        feature_coverage = stats.get("avg_feature_coverage", 0) or 0

        score = (citation_quality * 0.6 + feature_coverage * 0.4)

        if score >= 0.9:
            return "A (Excellent)"
        elif score >= 0.8:
            return "B (Good)"
        elif score >= 0.7:
            return "C (Acceptable)"
        elif score >= 0.6:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"

    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()
