"""
Configuration Module
====================
Centralized configuration for the entire project.

Modules:
    - paths: File paths and directory structure
    - adapters.config: Environment and API configuration (Neo4j, OpenAI)

Usage:
    from backend.src.config import get_paths, get_neo4j_config

    paths = get_paths()
    neo4j = get_neo4j_config()
"""

from backend.src.config.paths import PathConfig, get_paths, reset_paths

__all__ = ["PathConfig", "get_paths", "reset_paths"]
