"""
Configuration Adapter
=====================
Unified configuration reader supporting both .env and Streamlit secrets.
"""

# Standard library
import os
from typing import Any, Optional, Dict

# Third-party
from dotenv import load_dotenv

load_dotenv()


def get_config(key: str, default: Any = None) -> Any:
    """
    Read config from environment or Streamlit secrets.

    Supports both local development (.env) and Streamlit Cloud deployment.

    Args:
        key: Config key (supports nested keys like "neo4j.uri")
        default: Default value if key not found

    Returns:
        Config value

    Examples:
        >>> get_config("NEO4J_URI")
        'neo4j://localhost:7687'

        >>> get_config("neo4j.uri")  # On Streamlit Cloud
        'neo4j+s://xxxxx.databases.neo4j.io'
    """
    # Try environment variable first
    value = os.getenv(key, default)

    # If running on Streamlit Cloud, try secrets
    try:
        import streamlit as st

        # Direct key
        if key in st.secrets:
            value = st.secrets[key]

        # Nested key (e.g., "neo4j.uri")
        elif "." in key:
            parts = key.split(".")
            if parts[0] in st.secrets:
                nested = st.secrets[parts[0]]
                if len(parts) == 2 and parts[1] in nested:
                    value = nested[parts[1]]

    except (ImportError, FileNotFoundError):
        # Not running in Streamlit context
        pass

    return value


def get_neo4j_config() -> Dict[str, str]:
    """
    Get Neo4j configuration.

    Returns:
        Dict with uri, user, password, database

    Examples:
        >>> config = get_neo4j_config()
        >>> config['uri']
        'neo4j://localhost:7687'
    """
    return {
        "uri": get_config("NEO4J_URI") or get_config("neo4j.uri", "neo4j://127.0.0.1:7687"),
        "user": get_config("NEO4J_USER") or get_config("neo4j.user", "neo4j"),
        "password": get_config("NEO4J_PASSWORD") or get_config("neo4j.password", "test1234"),
        "database": get_config("NEO4J_DATABASE") or get_config("neo4j.database", "neo4j"),
    }


def get_openai_config() -> Dict[str, str]:
    """
    Get OpenAI configuration.

    Returns:
        Dict with api_key and model

    Examples:
        >>> config = get_openai_config()
        >>> config['model']
        'gpt-4o'
    """
    return {
        "api_key": get_config("OPENAI_API_KEY") or get_config("openai.api_key", ""),
        "model": get_config("OPENAI_MODEL") or get_config("openai.model", "gpt-4o"),
    }


def is_streamlit_cloud() -> bool:
    """
    Check if running on Streamlit Cloud.

    Returns:
        True if on Streamlit Cloud, False otherwise
    """
    try:
        import streamlit as st
        return hasattr(st, 'secrets') and len(st.secrets) > 0
    except (ImportError, FileNotFoundError):
        return False
