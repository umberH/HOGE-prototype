"""
Centralized Path Configuration
================================
All file paths and directory structures for the project.
Modify paths here instead of hardcoding them in individual scripts.

Usage:
    from backend.src.config.paths import PathConfig

    paths = PathConfig()
    df = pd.read_csv(paths.RAW_DATA_LOAN)
    model = joblib.load(paths.MODEL_XGB_MONOTONIC)
"""

import os
from pathlib import Path
from typing import Optional


class PathConfig:
    """
    Centralized path configuration for the entire project.

    All paths are configurable via environment variables with sensible defaults.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize path configuration.

        Args:
            project_root: Project root directory. If None, auto-detected.
        """
        # Auto-detect project root (4 levels up from this file)
        if project_root is None:
            self.PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
        else:
            self.PROJECT_ROOT = Path(project_root)

        # ============================================================
        # MAIN DIRECTORIES
        # ============================================================

        # Resource directory (contains data, models, etc.)
        self.RESOURCES_DIR = self.PROJECT_ROOT / ".resources"

        # Backend directories
        self.BACKEND_DIR = self.PROJECT_ROOT / "backend"
        self.SRC_DIR = self.BACKEND_DIR / "src"

        # Frontend directory
        self.FRONTEND_DIR = self.PROJECT_ROOT / "frontend"

        # Development directory
        self.DEV_DIR = self.PROJECT_ROOT / ".development"

        # Documentation directory
        self.DOCS_DIR = self.PROJECT_ROOT / ".documentation"

        # ============================================================
        # DATA DIRECTORIES
        # ============================================================

        self.DATA_DIR = self.RESOURCES_DIR / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.EVALUATION_DIR = self.DATA_DIR / "evaluation"
        self.PROVENANCE_DIR = self.DATA_DIR / "provenance"

        # ============================================================
        # MODEL DIRECTORIES
        # ============================================================

        self.MODELS_DIR = self.RESOURCES_DIR / "models"
        self.MULTIMODAL_DIR = self.MODELS_DIR / "multimodal"

        # ============================================================
        # RAW DATA FILES
        # ============================================================

        # Primary loan dataset
        self.RAW_DATA_LOAN = self.RAW_DATA_DIR / "df1_loan.csv"

        # New applications for prediction
        self.RAW_DATA_NEW_APPLICATIONS = self.RAW_DATA_DIR / "new_applications.csv"

        # ============================================================
        # PROCESSED DATA FILES
        # ============================================================

        # SHAP values
        self.SHAP_LONG = self.PROCESSED_DATA_DIR / "shap_long.csv"
        self.SHAP_WIDE = self.PROCESSED_DATA_DIR / "shap_wide.csv"

        # Scored applications
        self.SCORED_APPLICATIONS = self.PROCESSED_DATA_DIR / "scored_applications_xgb.csv"

        # ============================================================
        # MODEL FILES
        # ============================================================

        # Main XGBoost model
        self.MODEL_XGB_MONOTONIC = self.MODELS_DIR / "loan_xgb_monotonic.joblib"

        # Model provenance
        self.MODEL_PROVENANCE = self.MODELS_DIR / "model_provenance.json"

        # ============================================================
        # EVALUATION FILES
        # ============================================================

        # Counterfactual evaluation results
        self.EVAL_COUNTERFACTUAL = self.EVALUATION_DIR / "eval_counterfactual.json"

        # Human evaluation results
        self.EVAL_HUMAN = self.EVALUATION_DIR / "eval_human.json"

        # System evaluation results
        self.EVAL_SYSTEM = self.EVALUATION_DIR / "eval_system.json"

        # Fidelity metrics
        self.EVAL_FIDELITY = self.EVALUATION_DIR / "eval_fidelity.json"

        # ============================================================
        # PROVENANCE FILES
        # ============================================================

        # Knowledge graph provenance
        self.KG_PROVENANCE = self.PROVENANCE_DIR / "kg_provenance.json"

        # ============================================================
        # ENVIRONMENT-SPECIFIC OVERRIDES
        # ============================================================

        # Allow overriding paths via environment variables
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides to paths."""
        # Data directories
        if data_dir := os.getenv("DATA_DIR"):
            self.DATA_DIR = Path(data_dir)
            self.RAW_DATA_DIR = self.DATA_DIR / "raw"
            self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"

        # Model directory
        if models_dir := os.getenv("MODELS_DIR"):
            self.MODELS_DIR = Path(models_dir)

        # Specific file overrides
        if raw_loan := os.getenv("RAW_DATA_LOAN"):
            self.RAW_DATA_LOAN = Path(raw_loan)

        if model_path := os.getenv("MODEL_XGB_MONOTONIC"):
            self.MODEL_XGB_MONOTONIC = Path(model_path)

    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.RESOURCES_DIR,
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.EVALUATION_DIR,
            self.PROVENANCE_DIR,
            self.MODELS_DIR,
            self.MULTIMODAL_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_relative_path(self, path: Path) -> Path:
        """
        Get path relative to project root.

        Args:
            path: Absolute path

        Returns:
            Path relative to project root
        """
        try:
            return path.relative_to(self.PROJECT_ROOT)
        except ValueError:
            # Path is not relative to project root
            return path

    def __repr__(self):
        """String representation showing key paths."""
        return f"""PathConfig(
    PROJECT_ROOT={self.PROJECT_ROOT}
    DATA_DIR={self.DATA_DIR}
    MODELS_DIR={self.MODELS_DIR}
    RAW_DATA_LOAN={self.RAW_DATA_LOAN}
    MODEL_XGB_MONOTONIC={self.MODEL_XGB_MONOTONIC}
)"""


# Global singleton instance
_paths_instance: Optional[PathConfig] = None


def get_paths(project_root: Optional[Path] = None) -> PathConfig:
    """
    Get the global PathConfig instance.

    Args:
        project_root: Project root directory. Only used on first call.

    Returns:
        PathConfig instance

    Examples:
        >>> from backend.src.config.paths import get_paths
        >>> paths = get_paths()
        >>> df = pd.read_csv(paths.RAW_DATA_LOAN)
    """
    global _paths_instance

    if _paths_instance is None:
        _paths_instance = PathConfig(project_root)

    return _paths_instance


# Convenience function to reset the singleton (useful for testing)
def reset_paths():
    """Reset the global PathConfig instance."""
    global _paths_instance
    _paths_instance = None


if __name__ == "__main__":
    # Test the configuration
    paths = PathConfig()
    print(paths)
    print("\nChecking if key files exist:")
    print(f"  RAW_DATA_LOAN: {paths.RAW_DATA_LOAN.exists()}")
    print(f"  MODEL_XGB_MONOTONIC: {paths.MODEL_XGB_MONOTONIC.exists()}")
    print(f"  SHAP_LONG: {paths.SHAP_LONG.exists()}")
