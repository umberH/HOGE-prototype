"""
Data Transfer Objects (DTOs)
=============================
These classes define the data contracts between frontend and backend.
"""

from .explanation_dto import (
    ExplanationRequest,
    ExplanationResponse,
    ShapFeature,
    CounterfactualScenario,
    BusinessConcept,
)
from .application_dto import (
    ApplicationInfo,
    ApplicationListResponse,
)
from .evaluation_dto import (
    EvaluationMetrics,
    EvaluationResult,
)
from .common_dto import (
    ProvenanceInfo,
    SystemStatus,
)

__all__ = [
    "ExplanationRequest",
    "ExplanationResponse",
    "ShapFeature",
    "CounterfactualScenario",
    "BusinessConcept",
    "ApplicationInfo",
    "ApplicationListResponse",
    "EvaluationMetrics",
    "EvaluationResult",
    "ProvenanceInfo",
    "SystemStatus",
]
