"""
Common Data Transfer Objects
=============================
Shared DTOs used across multiple services.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class SystemStatus:
    """System connection and health status."""

    neo4j_connected: bool
    neo4j_message: Optional[str] = None
    openai_configured: bool = False
    openai_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ProvenanceInfo:
    """Provenance metadata for explanations."""

    # Instance-level provenance
    llm_model: str
    llm_tokens: Optional[int] = None
    evidence_items: int = 0
    target_audience: str = "technical"
    generation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # System-level provenance
    model_name: str = "xgb_monotonic_v1"
    xai_method: str = "TreeSHAP"
    kg_loader_version: str = "neo_loader_v1"
    hoge_version: str = "1.0.0"

    # Full provenance (optional)
    full_provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "llm_model": self.llm_model,
            "llm_tokens": self.llm_tokens,
            "evidence_items": self.evidence_items,
            "target_audience": self.target_audience,
            "generation_timestamp": self.generation_timestamp,
            "model_name": self.model_name,
            "xai_method": self.xai_method,
            "kg_loader_version": self.kg_loader_version,
            "hoge_version": self.hoge_version,
        }
