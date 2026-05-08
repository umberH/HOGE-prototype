"""
Unit tests for Explanation DTOs.
"""

# Standard library
from typing import List

# Third-party
import pytest

# Local application
from backend.src.api.models.explanation_dto import (
    ExplanationRequest,
    ExplanationResponse,
    ShapFeature,
    CounterfactualScenario,
    BusinessConcept,
)
from backend.src.api.models.common_dto import ProvenanceInfo


class TestExplanationRequest:
    """Test ExplanationRequest DTO."""

    def test_create_basic_request(self):
        """Test creating a basic explanation request."""
        request = ExplanationRequest(
            application_id="LP001006",
            audience="technical"
        )

        assert request.application_id == "LP001006"
        assert request.audience == "technical"
        assert request.use_concepts is False
        assert request.include_mechanistic is False

    def test_create_with_concepts(self):
        """Test creating request with business concepts enabled."""
        request = ExplanationRequest(
            application_id="LP001006",
            audience="business",
            use_concepts=True
        )

        assert request.use_concepts is True
        assert request.audience == "business"

    def test_create_with_mechanistic(self):
        """Test creating request with mechanistic analysis."""
        request = ExplanationRequest(
            application_id="LP001006",
            audience="technical",
            include_mechanistic=True
        )

        assert request.include_mechanistic is True


class TestShapFeature:
    """Test ShapFeature DTO."""

    def test_create_shap_feature(self):
        """Test creating a SHAP feature."""
        feature = ShapFeature(
            feature="Credit_History",
            value=1.0,
            shap=0.35,
            importance=0.35,
            direction="positive",
            rank=1
        )

        assert feature.feature == "Credit_History"
        assert feature.shap == 0.35
        assert feature.direction == "positive"


class TestCounterfactualScenario:
    """Test CounterfactualScenario DTO."""

    def test_create_counterfactual(self):
        """Test creating a counterfactual scenario."""
        cf = CounterfactualScenario(
            feature="LoanAmount",
            current_value=150,
            changed_to=120,
            change_description="Reduce loan amount by 20%",
            probability_shift=0.15,
            new_probability=0.75,
            flipped_prediction=True
        )

        assert cf.feature == "LoanAmount"
        assert cf.probability_shift == 0.15
        assert cf.flipped_prediction is True


class TestBusinessConcept:
    """Test BusinessConcept DTO."""

    def test_create_business_concept(self):
        """Test creating a business concept."""
        concept = BusinessConcept(
            concept_name="Financial Stability",
            concept_category="financial",
            aggregated_shap_value=0.45,
            overall_direction="positive",
            related_features=["ApplicantIncome", "CoapplicantIncome"],
            business_interpretation="Strong financial position",
            confidence_level="high"
        )

        assert concept.concept_name == "Financial Stability"
        assert len(concept.related_features) == 2
        assert concept.confidence_level == "high"


class TestExplanationResponse:
    """Test ExplanationResponse DTO."""

    def test_create_minimal_response(self):
        """Test creating a minimal explanation response."""
        prov = ProvenanceInfo(llm_model="gpt-4o", evidence_items=5)

        response = ExplanationResponse(
            application_id="LP001006",
            decision="Approved",
            approval_probability=0.85,
            summary="Test summary",
            narrative="Test narrative",
            recommendation="Test recommendation",
            positive_drivers=["Good credit"],
            negative_drivers=[],
            policy_violations=[],
            shap_features=[],
            features_used=["Credit_History"],
            counterfactual_what_if="No changes needed",
            provenance=prov
        )

        assert response.application_id == "LP001006"
        assert response.decision == "Approved"
        assert response.approval_probability == 0.85
        assert len(response.positive_drivers) == 1

    def test_to_dict_serialization(self):
        """Test DTO serialization to dict."""
        prov = ProvenanceInfo(llm_model="gpt-4o", evidence_items=5)

        response = ExplanationResponse(
            application_id="LP001006",
            decision="Approved",
            approval_probability=0.85,
            summary="Test",
            narrative="Test",
            recommendation="Test",
            positive_drivers=[],
            negative_drivers=[],
            policy_violations=[],
            shap_features=[],
            features_used=[],
            counterfactual_what_if="",
            provenance=prov
        )

        result = response.to_dict()

        assert isinstance(result, dict)
        assert result["application_id"] == "LP001006"
        assert result["decision"] == "Approved"
        assert "provenance" in result
