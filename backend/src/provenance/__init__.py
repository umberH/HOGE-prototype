"""
HOGE Framework - Enhanced Provenance Tracking Module
====================================================

This module provides comprehensive provenance tracking for all components
of the HOGE framework to ensure full reproducibility and auditability.

Usage:
    from src.provenance.enhanced_provenance import ProvenanceTracker

    tracker = ProvenanceTracker()
    tracker.add_system_environment()
    tracker.add_llm_provenance(response, config)
    provenance = tracker.export_provenance()
"""

from .enhanced_provenance import ProvenanceTracker

__all__ = ['ProvenanceTracker']
__version__ = '1.0.0'
