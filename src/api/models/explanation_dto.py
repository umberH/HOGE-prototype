"""
Explanation DTOs
================
Data models for explanation requests and responses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .common_dto import ProvenanceInfo


@dataclass
class ShapFeature:
    """Single SHAP feature contribution."""

    feature: str
    value: Any
    shap: float
    importance: float
    direction: str  # "positive" or "negative"
    rank: int


@dataclass
class CounterfactualScenario:
    """What-if scenario for counterfactual analysis."""

    feature: str
    current_value: Any
    changed_to: Any
    change_description: str
    probability_shift: float
    new_probability: float
    flipped_prediction: bool = False


@dataclass
class BusinessConcept:
    """Business concept grouping multiple features."""

    concept_name: str
    concept_category: str
    aggregated_shap_value: float
    overall_direction: str
    related_features: List[str]
    business_interpretation: str
    confidence_level: str = "medium"  # high, medium, low


@dataclass
class ExplanationRequest:
    """Request for generating an explanation."""

    application_id: str
    audience: str = "technical"  # technical, non_technical, executive, business
    use_concepts: bool = False  # Whether to use business concepts
    include_mechanistic: bool = False  # Whether to include deep mechanistic analysis


@dataclass
class ExplanationResponse:
    """Complete explanation response."""

    # Application info
    application_id: str
    decision: str  # "Approved" or "Not Approved"
    approval_probability: float

    # LLM-generated explanations
    summary: str
    narrative: str
    recommendation: str

    # Key drivers
    positive_drivers: List[str]
    negative_drivers: List[str]
    policy_violations: List[str]

    # SHAP details
    shap_features: List[ShapFeature]
    features_used: List[str]  # Features mentioned by LLM

    # Counterfactuals
    counterfactual_what_if: str
    counterfactual_scenarios: List[CounterfactualScenario] = field(default_factory=list)

    # Business concepts (optional)
    used_concepts: bool = False
    business_concepts: Optional[List[BusinessConcept]] = None

    # Provenance
    provenance: ProvenanceInfo = None

    # Evidence (raw context for advanced users)
    evidence_bundle: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "application_id": self.application_id,
            "decision": self.decision,
            "approval_probability": self.approval_probability,
            "summary": self.summary,
            "narrative": self.narrative,
            "recommendation": self.recommendation,
            "positive_drivers": self.positive_drivers,
            "negative_drivers": self.negative_drivers,
            "policy_violations": self.policy_violations,
            "shap_features": [
                {
                    "feature": sf.feature,
                    "value": sf.value,
                    "shap": sf.shap,
                    "importance": sf.importance,
                    "direction": sf.direction,
                    "rank": sf.rank,
                }
                for sf in self.shap_features
            ],
            "features_used": self.features_used,
            "counterfactual_what_if": self.counterfactual_what_if,
            "counterfactual_scenarios": [
                {
                    "feature": cs.feature,
                    "current_value": cs.current_value,
                    "changed_to": cs.changed_to,
                    "change_description": cs.change_description,
                    "probability_shift": cs.probability_shift,
                    "new_probability": cs.new_probability,
                    "flipped_prediction": cs.flipped_prediction,
                }
                for cs in self.counterfactual_scenarios
            ],
            "used_concepts": self.used_concepts,
            "business_concepts": [
                {
                    "concept_name": bc.concept_name,
                    "concept_category": bc.concept_category,
                    "aggregated_shap_value": bc.aggregated_shap_value,
                    "overall_direction": bc.overall_direction,
                    "related_features": bc.related_features,
                    "business_interpretation": bc.business_interpretation,
                    "confidence_level": bc.confidence_level,
                }
                for bc in (self.business_concepts or [])
            ],
            "provenance": self.provenance.to_dict() if self.provenance else {},
            "evidence_bundle": self.evidence_bundle,
        }
