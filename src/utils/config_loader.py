"""
Configuration loader for HOGE framework.
Loads JSON configuration files from config/ directory.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration files."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

    def load(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration file by name.

        Args:
            config_name: Name of config file (e.g., 'model_hyperparameters')

        Returns:
            Dictionary with configuration data
        """
        config_path = self.config_dir / f"{config_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            return json.load(f)

    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """Load configuration for a specific model.

        Args:
            model_name: Name of model (e.g., 'xgboost_monotonic')

        Returns:
            Dictionary with model configuration
        """
        all_configs = self.load('model_hyperparameters')
        if model_name not in all_configs:
            raise ValueError(f"Model '{model_name}' not found in config. "
                           f"Available: {list(all_configs.keys())}")
        return all_configs[model_name]

    def get_pipeline_config(self) -> Dict[str, Any]:
        """Load pipeline configuration."""
        return self.load('pipeline_config')

    def get_neo4j_config(self) -> Dict[str, Any]:
        """Load Neo4j configuration."""
        return self.load('neo4j_config')

    def get_llm_config(self) -> Dict[str, Any]:
        """Load LLM configuration."""
        return self.load('llm_config')

    def get_env_or_default(self, env_var: str, default: str) -> str:
        """Get environment variable or default value.

        Args:
            env_var: Name of environment variable
            default: Default value if env var not set

        Returns:
            Value from environment or default
        """
        return os.getenv(env_var, default)


# Singleton instance
_config_loader = None


def get_config_loader(config_dir: str = "config") -> ConfigLoader:
    """Get or create ConfigLoader singleton instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader
