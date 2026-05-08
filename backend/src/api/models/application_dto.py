"""
Application DTOs
================
Data models for loan application information.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ApplicationInfo:
    """Information about a single loan application."""

    application_id: str
    decision: str  # "Approved" or "Not Approved"
    probability: Optional[float] = None
    probability_str: str = "N/A"
    label: Optional[str] = None  # Formatted display label

    def __post_init__(self):
        """Generate display label and probability string if not provided."""
        # Update probability_str based on probability
        if self.probability is not None:
            self.probability_str = f"{self.probability * 100:.1f}%"

        # Generate label if not provided
        if self.label is None:
            self.label = f"{self.application_id} - {self.decision} ({self.probability_str})"


@dataclass
class ApplicationListResponse:
    """Response containing list of applications."""

    applications: List[ApplicationInfo]
    total_count: int

    @classmethod
    def from_list(cls, apps: List[ApplicationInfo]) -> "ApplicationListResponse":
        """Create from a list of ApplicationInfo objects."""
        return cls(applications=apps, total_count=len(apps))
