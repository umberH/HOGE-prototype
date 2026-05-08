"""
Explanation Service
===================
Service for generating explanations with abstraction from backend implementation.
"""

# Standard library
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Local application imports
from src.api.models.explanation_dto import (
    ExplanationRequest,
    ExplanationResponse,
    ShapFeature,
    CounterfactualScenario,
    BusinessConcept,
)
from src.api.models.common_dto import ProvenanceInfo
from src.explainability.llm_explainer import (
    get_application_explanation_data,
    call_llm_for_explanation,
)


class ExplanationService:
    """
    Service for generating explanations.
    Wraps backend logic and returns DTOs.
    """

    def __init__(self):
        """Initialize the explanation service."""
        pass

    def generate_explanation(
        self, request: ExplanationRequest
    ) -> ExplanationResponse:
        """
        Generate explanation for a loan application.

        Args:
            request: ExplanationRequest with application_id, audience, use_concepts

        Returns:
            ExplanationResponse DTO with complete explanation

        Raises:
            ValueError: If application not found
            Exception: For other errors
        """
        # 1. Get context from backend (Neo4j + SHAP)
        context = get_application_explanation_data(request.application_id)

        # 2. Call LLM for explanation
        explanation = call_llm_for_explanation(
            context,
            audience=request.audience,
            use_concepts=request.use_concepts
        )

        # 3. Transform to DTO
        return self._transform_to_dto(explanation, context, request)

    def _transform_to_dto(
        self,
        explanation: dict,
        context: dict,
        request: ExplanationRequest
    ) -> ExplanationResponse:
        """
        Transform raw backend response to ExplanationResponse DTO.

        Args:
            explanation: Raw LLM explanation dict
            context: Raw context from knowledge graph
            request: Original request

        Returns:
            ExplanationResponse DTO
        """
        # Transform SHAP features
        shap_features = [
            ShapFeature(
                feature=feat["feature"],
                value=feat["value"],
                shap=feat["shap"],
                importance=feat["importance"],
                direction=feat["direction"],
                rank=feat["rank"]
            )
            for feat in context.get("shap_details", [])
        ]

        # Transform counterfactual scenarios
        counterfactual_scenarios = [
            CounterfactualScenario(
                feature=cf.get("feature", ""),
                current_value=cf.get("current_value"),
                changed_to=cf.get("changed_to"),
                change_description=cf.get("change_description", ""),
                probability_shift=cf.get("probability_shift", 0.0),
                new_probability=cf.get("new_probability", 0.0),
                flipped_prediction=cf.get("flipped_prediction", False)
            )
            for cf in context.get("counterfactual_scenarios", [])
        ]

        # Transform business concepts (if available)
        business_concepts = None
        if explanation.get("business_concepts"):
            business_concepts = [
                BusinessConcept(
                    concept_name=bc["concept_name"],
                    concept_category=bc["concept_category"],
                    aggregated_shap_value=bc["aggregated_shap_value"],
                    overall_direction=bc["overall_direction"],
                    related_features=bc["related_features"],
                    business_interpretation=bc["business_interpretation"],
                    confidence_level=bc.get("confidence_level", "medium")
                )
                for bc in explanation["business_concepts"]
            ]

        # Build provenance info
        provenance = self._build_provenance(explanation, request)

        # Build response DTO
        return ExplanationResponse(
            application_id=request.application_id,
            decision=context["model_prediction"],
            approval_probability=context["approval_probability"],
            summary=explanation["summary"],
            narrative=explanation["narrative"],
            recommendation=explanation["recommendation"],
            positive_drivers=explanation["positive_drivers"],
            negative_drivers=explanation["negative_drivers"],
            policy_violations=explanation["policy_violations"],
            shap_features=shap_features,
            features_used=explanation["features_used"],
            counterfactual_what_if=explanation.get("counterfactual_what_if", ""),
            counterfactual_scenarios=counterfactual_scenarios,
            used_concepts=explanation.get("used_concepts", False),
            business_concepts=business_concepts,
            provenance=provenance,
            evidence_bundle=explanation.get("evidence_bundle", [])
        )

    def _build_provenance(
        self,
        explanation: dict,
        request: ExplanationRequest
    ) -> ProvenanceInfo:
        """Build provenance metadata."""
        prov_compact = explanation.get("provenance_compact", {})
        prov_full = explanation.get("provenance_full", {})
        prov_basic = explanation.get("provenance", {})

        return ProvenanceInfo(
            llm_model=prov_compact.get("llm_model") or prov_basic.get("llm_model", "gpt-4o"),
            llm_tokens=prov_compact.get("llm_tokens"),
            evidence_items=prov_compact.get("evidence_items", len(explanation.get("evidence_bundle", []))),
            target_audience=request.audience,
            generation_timestamp=prov_compact.get("generation_timestamp") or datetime.now().isoformat(),
            model_name=prov_basic.get("model_name", "xgb_monotonic_v1"),
            xai_method=prov_basic.get("xai_method", "TreeSHAP"),
            kg_loader_version=prov_basic.get("kg_loader_version", "neo_loader_v1"),
            hoge_version=prov_basic.get("hoge_version", "1.0.0"),
            full_provenance=prov_full if prov_full else None
        )

    def get_cached_explanation(
        self,
        application_id: str,
        audience: str,
        use_concepts: bool,
        cache: dict
    ) -> Optional[ExplanationResponse]:
        """
        Retrieve cached explanation if available.

        Args:
            application_id: Application ID
            audience: Target audience
            use_concepts: Whether concepts were used
            cache: Cache dictionary

        Returns:
            ExplanationResponse if cached, else None
        """
        cache_key = f"{application_id}_{audience}_{use_concepts}"

        if cache_key in cache:
            cached_data = cache[cache_key]
            # Cached data is raw dict, transform to DTO
            # (In practice, you'd cache the DTO directly or serialize it)
            return cached_data  # Return as-is for now

        return None

    def cache_explanation(
        self,
        response: ExplanationResponse,
        cache: dict
    ) -> None:
        """
        Cache an explanation response.

        Args:
            response: ExplanationResponse to cache
            cache: Cache dictionary to update
        """
        cache_key = f"{response.application_id}_{response.provenance.target_audience}_{response.used_concepts}"

        cache[cache_key] = {
            'explanation': response.to_dict(),
            'application_id': response.application_id,
            'audience': response.provenance.target_audience,
            'use_concepts': response.used_concepts,
            'cached_at': datetime.now().isoformat()
        }
