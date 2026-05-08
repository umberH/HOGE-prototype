"""
Unit tests for Application DTOs.
"""

# Third-party
import pytest

# Local application
from src.api.models.application_dto import ApplicationInfo, ApplicationListResponse


class TestApplicationInfo:
    """Test ApplicationInfo DTO."""

    def test_create_with_probability(self):
        """Test creating application info with probability."""
        app = ApplicationInfo(
            application_id="LP001006",
            decision="Approved",
            probability=0.85
        )

        assert app.application_id == "LP001006"
        assert app.decision == "Approved"
        assert app.probability == 0.85
        assert app.probability_str == "85.0%"

    def test_auto_generate_label(self):
        """Test automatic label generation."""
        app = ApplicationInfo(
            application_id="LP001006",
            decision="Approved",
            probability=0.85
        )

        assert "LP001006" in app.label
        assert "Approved" in app.label
        assert "85.0%" in app.label

    def test_create_without_probability(self):
        """Test creating application without probability."""
        app = ApplicationInfo(
            application_id="LP001006",
            decision="Not Approved"
        )

        assert app.probability is None
        assert app.probability_str == "N/A"
        assert "N/A" in app.label


class TestApplicationListResponse:
    """Test ApplicationListResponse DTO."""

    def test_create_from_list(self):
        """Test creating response from list of applications."""
        apps = [
            ApplicationInfo("LP001006", "Approved", 0.85),
            ApplicationInfo("LP001003", "Not Approved", 0.25),
        ]

        response = ApplicationListResponse.from_list(apps)

        assert response.total_count == 2
        assert len(response.applications) == 2
        assert response.applications[0].application_id == "LP001006"

    def test_empty_list(self):
        """Test creating response from empty list."""
        response = ApplicationListResponse.from_list([])

        assert response.total_count == 0
        assert len(response.applications) == 0
