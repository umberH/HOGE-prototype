"""
Concept Mapper for HOGE Framework
==================================

This module provides high-level business concept abstraction over technical features.
Instead of explaining individual features, it groups them into business concepts
that are more meaningful to executives and business users.

Example:
    Instead of: "ApplicantIncome=5000, CoapplicantIncome=2000, LoanAmount=150"
    Use: "Financial Capacity: Strong (combined income covers loan comfortably)"

Usage:
    mapper = ConceptMapper()
    concepts = mapper.map_features_to_concepts(shap_details)
    narrative = mapper.generate_concept_narrative(concepts, audience="executive")
"""

import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class BusinessConcept:
    """Represents a high-level business concept"""
    concept_name: str
    concept_category: str  # "risk", "capacity", "profile", "collateral"
    description: str
    related_features: List[str]
    aggregated_importance: float
    aggregated_shap_value: float
    overall_direction: str  # "positive", "negative", "neutral"
    confidence_level: str  # "high", "medium", "low"
    business_interpretation: str


class ConceptMapper:
    """
    Maps technical features to business concepts for executive/business audiences.

    This creates a semantic layer that bridges ML features and business understanding.
    """

    def __init__(self):
        """Initialize concept mapper with predefined mappings"""
        self.feature_to_concept_map = self._build_feature_concept_map()
        self.concept_definitions = self._build_concept_definitions()

    def _build_feature_concept_map(self) -> Dict[str, str]:
        """
        Define mapping from technical features to business concepts.

        Returns:
            Dictionary mapping feature names to concept names
        """
        return {
            # Financial Capacity Concept
            "ApplicantIncome": "Financial Capacity",
            "CoapplicantIncome": "Financial Capacity",
            "LoanAmount": "Financial Capacity",
            "Loan_Amount_Term": "Financial Capacity",

            # Credit Profile Concept
            "Credit_History": "Credit Profile",
            "Credit_History_Available": "Credit Profile",

            # Employment Stability Concept
            "Self_Employed": "Employment Stability",
            "Education": "Employment Stability",

            # Applicant Demographics Concept
            "Gender": "Applicant Profile",
            "Married": "Applicant Profile",
            "Dependents": "Applicant Profile",

            # Property & Collateral Concept
            "Property_Area": "Property & Location",

            # Add more mappings as your features expand
        }

    def _build_concept_definitions(self) -> Dict[str, Dict[str, str]]:
        """
        Define business concepts with descriptions and interpretations.

        Returns:
            Dictionary of concept metadata
        """
        return {
            "Financial Capacity": {
                "category": "capacity",
                "description": "Ability to repay the loan based on income and loan amount",
                "business_meaning": "Measures whether the applicant can afford the loan",
                "technical_note": "Combines income sources and loan size metrics",
                "executive_summary": "Debt servicing ability"
            },
            "Credit Profile": {
                "category": "risk",
                "description": "Historical creditworthiness and repayment behavior",
                "business_meaning": "Indicates likelihood of timely repayment based on past behavior",
                "technical_note": "Based on credit history records",
                "executive_summary": "Repayment risk indicator"
            },
            "Employment Stability": {
                "category": "risk",
                "description": "Job security and income stability factors",
                "business_meaning": "Assesses consistency and reliability of income source",
                "technical_note": "Considers employment type and education level",
                "executive_summary": "Income stability assessment"
            },
            "Applicant Profile": {
                "category": "profile",
                "description": "Demographic and family situation characteristics",
                "business_meaning": "Context about applicant's life situation and responsibilities",
                "technical_note": "Demographic features (non-discriminatory use only)",
                "executive_summary": "Applicant background"
            },
            "Property & Location": {
                "category": "collateral",
                "description": "Property type and geographic location factors",
                "business_meaning": "Affects collateral value and regional risk assessment",
                "technical_note": "Property area classification",
                "executive_summary": "Collateral context"
            }
        }

    def map_features_to_concepts(self,
                                  shap_details: List[Dict],
                                  context: Dict = None) -> List[BusinessConcept]:
        """
        Aggregate features into business concepts with combined importance.

        Args:
            shap_details: List of feature contributions from SHAP
            context: Optional additional context (e.g., feature values)

        Returns:
            List of BusinessConcept objects
        """
        # Group features by concept
        concept_groups = defaultdict(list)

        for feature_data in shap_details:
            feature_name = feature_data.get('feature')
            if not feature_name:
                continue

            concept_name = self.feature_to_concept_map.get(feature_name, "Other Factors")
            concept_groups[concept_name].append(feature_data)

        # Aggregate into concepts
        concepts = []

        for concept_name, features in concept_groups.items():
            concept = self._aggregate_concept(concept_name, features, context)
            if concept:
                concepts.append(concept)

        # Sort by aggregated importance
        concepts.sort(key=lambda x: x.aggregated_importance, reverse=True)

        return concepts

    def _aggregate_concept(self,
                          concept_name: str,
                          features: List[Dict],
                          context: Dict = None) -> BusinessConcept:
        """
        Aggregate multiple features into a single business concept.

        Args:
            concept_name: Name of the concept
            features: List of feature data to aggregate
            context: Additional context

        Returns:
            BusinessConcept object
        """
        if not features:
            return None

        # Get concept definition
        concept_def = self.concept_definitions.get(
            concept_name,
            {
                "category": "other",
                "description": "Additional factors",
                "business_meaning": "Other considerations",
                "executive_summary": "Other factors"
            }
        )

        # Aggregate SHAP values and importance
        shap_values = [f.get('shap', 0) for f in features if f.get('shap') is not None]
        importances = [f.get('importance', 0) for f in features if f.get('importance') is not None]

        aggregated_shap = sum(shap_values)
        aggregated_importance = sum(importances)

        # Determine overall direction
        if aggregated_shap > 0.01:
            direction = "positive"
        elif aggregated_shap < -0.01:
            direction = "negative"
        else:
            direction = "neutral"

        # Determine confidence based on number of features and importance
        if len(features) >= 2 and aggregated_importance > 0.1:
            confidence = "high"
        elif aggregated_importance > 0.05:
            confidence = "medium"
        else:
            confidence = "low"

        # Generate business interpretation
        business_interp = self._generate_business_interpretation(
            concept_name, concept_def, direction, features
        )

        return BusinessConcept(
            concept_name=concept_name,
            concept_category=concept_def["category"],
            description=concept_def["description"],
            related_features=[f.get('feature') for f in features],
            aggregated_importance=float(aggregated_importance),
            aggregated_shap_value=float(aggregated_shap),
            overall_direction=direction,
            confidence_level=confidence,
            business_interpretation=business_interp
        )

    def _generate_business_interpretation(self,
                                          concept_name: str,
                                          concept_def: Dict,
                                          direction: str,
                                          features: List[Dict]) -> str:
        """
        Generate human-readable business interpretation of a concept.

        Args:
            concept_name: Name of the concept
            concept_def: Concept definition metadata
            direction: Overall impact direction
            features: Contributing features

        Returns:
            Business interpretation string
        """
        impact_word = {
            "positive": "supports approval",
            "negative": "raises concerns",
            "neutral": "has neutral impact"
        }.get(direction, "has mixed impact")

        # Concept-specific interpretations
        if concept_name == "Financial Capacity":
            if direction == "positive":
                return f"Strong financial capacity: Income adequately covers loan requirements"
            elif direction == "negative":
                return f"Limited financial capacity: Loan amount is high relative to income"
            else:
                return f"Adequate financial capacity: Income-to-loan ratio is acceptable"

        elif concept_name == "Credit Profile":
            if direction == "positive":
                return f"Strong credit history: Demonstrated reliable repayment behavior"
            elif direction == "negative":
                return f"Weak credit history: Past repayment issues raise risk concerns"
            else:
                return f"Average credit profile: Standard repayment history"

        elif concept_name == "Employment Stability":
            if direction == "positive":
                return f"Stable employment: Reliable and consistent income source"
            elif direction == "negative":
                return f"Employment concerns: Income stability may be limited"
            else:
                return f"Standard employment profile"

        elif concept_name == "Applicant Profile":
            return f"Applicant demographics and family situation {impact_word}"

        elif concept_name == "Property & Location":
            if direction == "positive":
                return f"Favorable property location: Lower risk area"
            else:
                return f"Property location: Standard risk assessment"

        # Default interpretation
        return f"{concept_name} {impact_word}"

    def generate_concept_narrative(self,
                                   concepts: List[BusinessConcept],
                                   audience: str = "executive",
                                   decision: str = None,
                                   probability: float = None) -> str:
        """
        Generate narrative explanation using business concepts instead of features.

        Args:
            concepts: List of BusinessConcept objects
            audience: Target audience ("executive" or "business")
            decision: Model decision (Approved/Rejected)
            probability: Approval probability

        Returns:
            Narrative explanation string
        """
        if audience == "executive":
            return self._generate_executive_narrative(concepts, decision, probability)
        else:
            return self._generate_business_narrative(concepts, decision, probability)

    def _generate_executive_narrative(self,
                                     concepts: List[BusinessConcept],
                                     decision: str,
                                     probability: float) -> str:
        """Generate executive-level narrative using concepts"""

        # Separate by direction
        positive_concepts = [c for c in concepts if c.overall_direction == "positive"]
        negative_concepts = [c for c in concepts if c.overall_direction == "negative"]

        # Build narrative
        narrative_parts = []

        # Opening
        decision_str = decision or "the decision"
        prob_str = f" ({probability*100:.0f}% confidence)" if probability else ""
        narrative_parts.append(
            f"**Decision Summary**: {decision_str}{prob_str}\n"
        )

        # Key drivers
        narrative_parts.append("\n**Primary Decision Drivers:**\n")

        top_concepts = concepts[:3]  # Top 3 most important
        for i, concept in enumerate(top_concepts, 1):
            category_emoji = {
                "capacity": "💰",
                "risk": "⚠️",
                "profile": "👤",
                "collateral": "🏠"
            }.get(concept.concept_category, "📊")

            narrative_parts.append(
                f"{i}. {category_emoji} **{concept.concept_name}**: "
                f"{concept.business_interpretation}\n"
            )

        # Risk assessment
        if negative_concepts:
            narrative_parts.append("\n**Risk Considerations:**\n")
            for concept in negative_concepts[:2]:  # Top 2 risks
                narrative_parts.append(f"- {concept.business_interpretation}\n")

        # Strengths
        if positive_concepts:
            narrative_parts.append("\n**Strengths:**\n")
            for concept in positive_concepts[:2]:  # Top 2 strengths
                narrative_parts.append(f"- {concept.business_interpretation}\n")

        return "".join(narrative_parts)

    def _generate_business_narrative(self,
                                    concepts: List[BusinessConcept],
                                    decision: str,
                                    probability: float) -> str:
        """Generate business analyst-level narrative using concepts"""

        narrative_parts = []

        # Overview
        decision_str = decision or "the decision"
        prob_str = f" with {probability*100:.1f}% approval probability" if probability else ""
        narrative_parts.append(
            f"**Business Analysis**: The model reached a decision of {decision_str}{prob_str} "
            f"based on {len(concepts)} key business factors.\n\n"
        )

        # Detailed concept breakdown
        narrative_parts.append("**Factor Analysis:**\n\n")

        for concept in concepts[:5]:  # Top 5 concepts
            impact_emoji = {
                "positive": "✅",
                "negative": "❌",
                "neutral": "➖"
            }.get(concept.overall_direction, "•")

            narrative_parts.append(
                f"{impact_emoji} **{concept.concept_name}** "
                f"(Impact: {concept.aggregated_shap_value:+.3f}, "
                f"Confidence: {concept.confidence_level})\n"
                f"   {concept.business_interpretation}\n"
                f"   *Based on: {', '.join(concept.related_features[:3])}*\n\n"
            )

        # Summary insight
        positive_count = sum(1 for c in concepts if c.overall_direction == "positive")
        negative_count = sum(1 for c in concepts if c.overall_direction == "negative")

        narrative_parts.append(
            f"\n**Overall Assessment**: {positive_count} factors support approval, "
            f"{negative_count} factors raise concerns."
        )

        return "".join(narrative_parts)

    def export_concept_map(self,
                          concepts: List[BusinessConcept],
                          output_path: str = None) -> Dict:
        """
        Export concept map to JSON for visualization/KG integration.

        Args:
            concepts: List of BusinessConcept objects
            output_path: Optional path to save JSON

        Returns:
            Dictionary with concept map data
        """
        export_data = {
            "concept_map": {
                "concepts": [asdict(c) for c in concepts],
                "summary": {
                    "total_concepts": len(concepts),
                    "positive_concepts": sum(1 for c in concepts if c.overall_direction == "positive"),
                    "negative_concepts": sum(1 for c in concepts if c.overall_direction == "negative"),
                    "high_confidence_concepts": sum(1 for c in concepts if c.confidence_level == "high")
                },
                "feature_to_concept_mapping": self.feature_to_concept_map,
                "concept_definitions": self.concept_definitions
            },
            "provenance": {
                "component": "ConceptMapper",
                "version": "1.0.0",
                "method": "feature_concept_aggregation"
            }
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Concept map exported to {output_path}")

        return export_data

    def get_concept_for_feature(self, feature_name: str) -> str:
        """Get the business concept for a given feature"""
        return self.feature_to_concept_map.get(feature_name, "Other Factors")

    def get_concept_definition(self, concept_name: str) -> Dict:
        """Get the definition and metadata for a concept"""
        return self.concept_definitions.get(concept_name, {})


# Standalone utility functions

def create_concept_enhanced_context(context: Dict,
                                    use_concepts: bool = True) -> Dict:
    """
    Enhance explanation context with concept-level aggregations.

    This creates an augmented context that includes both feature-level
    and concept-level information for flexible explanation generation.

    Args:
        context: Original explanation context from get_application_explanation_data
        use_concepts: Whether to add concept mappings

    Returns:
        Enhanced context with concept information
    """
    if not use_concepts:
        return context

    mapper = ConceptMapper()

    # Map features to concepts
    shap_details = context.get('shap_details', [])
    concepts = mapper.map_features_to_concepts(shap_details, context)

    # Add concept information to context
    enhanced_context = context.copy()
    enhanced_context['business_concepts'] = [asdict(c) for c in concepts]
    enhanced_context['concept_narrative'] = mapper.generate_concept_narrative(
        concepts,
        audience="executive",
        decision=context.get('model_prediction'),
        probability=context.get('approval_probability')
    )
    enhanced_context['use_concepts'] = True

    return enhanced_context


if __name__ == "__main__":
    """Demo/test of concept mapper"""

    # Sample SHAP details (simulated)
    sample_shap = [
        {"feature": "ApplicantIncome", "value": 5000, "shap": 0.15, "importance": 0.25},
        {"feature": "CoapplicantIncome", "value": 2000, "shap": 0.08, "importance": 0.12},
        {"feature": "LoanAmount", "value": 150, "shap": -0.05, "importance": 0.18},
        {"feature": "Credit_History", "value": 1.0, "shap": 0.22, "importance": 0.35},
        {"feature": "Property_Area", "value": "Urban", "shap": 0.03, "importance": 0.08},
    ]

    # Create mapper
    mapper = ConceptMapper()

    # Map to concepts
    concepts = mapper.map_features_to_concepts(sample_shap)

    print("\n" + "="*80)
    print("CONCEPT MAPPING DEMO")
    print("="*80 + "\n")

    print("Individual Concepts:")
    print("-"*80)
    for concept in concepts:
        print(f"\n📊 {concept.concept_name}")
        print(f"   Category: {concept.concept_category}")
        print(f"   Direction: {concept.overall_direction}")
        print(f"   Impact: {concept.aggregated_shap_value:+.3f}")
        print(f"   Importance: {concept.aggregated_importance:.3f}")
        print(f"   Features: {', '.join(concept.related_features)}")
        print(f"   Interpretation: {concept.business_interpretation}")

    print("\n" + "="*80)
    print("EXECUTIVE NARRATIVE")
    print("="*80 + "\n")

    narrative = mapper.generate_concept_narrative(
        concepts,
        audience="executive",
        decision="Approved",
        probability=0.78
    )
    print(narrative)

    print("\n" + "="*80)
    print("BUSINESS NARRATIVE")
    print("="*80 + "\n")

    narrative = mapper.generate_concept_narrative(
        concepts,
        audience="business",
        decision="Approved",
        probability=0.78
    )
    print(narrative)
