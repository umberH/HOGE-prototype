"""
Configuration Migration Helper
==============================
Helps identify and migrate hardcoded paths to centralized configuration.

Usage:
    python .development/scripts/migrate_to_config.py <file_path>

Example:
    python .development/scripts/migrate_to_config.py backend/src/explainability/counterfactual_explainer.py
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple


# Path patterns to search for
PATH_PATTERNS = [
    (r'"data/raw/([^"]+)"', 'paths.RAW_DATA_DIR / "{}"'),
    (r'"data/processed/([^"]+)"', 'paths.PROCESSED_DATA_DIR / "{}"'),
    (r'"data/evaluation/([^"]+)"', 'paths.EVALUATION_DIR / "{}"'),
    (r'"models/([^"]+)"', 'paths.MODELS_DIR / "{}"'),
    (r'"\.resources/data/raw/([^"]+)"', 'paths.RAW_DATA_DIR / "{}"'),
    (r'"\.resources/data/processed/([^"]+)"', 'paths.PROCESSED_DATA_DIR / "{}"'),
    (r'"\.resources/models/([^"]+)"', 'paths.MODELS_DIR / "{}"'),
]

# Known file mappings
KNOWN_FILES = {
    "df1_loan.csv": "paths.RAW_DATA_LOAN",
    "new_applications.csv": "paths.RAW_DATA_NEW_APPLICATIONS",
    "shap_long.csv": "paths.SHAP_LONG",
    "shap_wide.csv": "paths.SHAP_WIDE",
    "scored_applications_xgb.csv": "paths.SCORED_APPLICATIONS",
    "loan_xgb_monotonic.joblib": "paths.MODEL_XGB_MONOTONIC",
    "model_provenance.json": "paths.MODEL_PROVENANCE",
    "eval_counterfactual.json": "paths.EVAL_COUNTERFACTUAL",
    "eval_human.json": "paths.EVAL_HUMAN",
    "eval_system.json": "paths.EVAL_SYSTEM",
    "eval_fidelity.json": "paths.EVAL_FIDELITY",
    "kg_provenance.json": "paths.KG_PROVENANCE",
}


def find_hardcoded_paths(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Find hardcoded paths in a Python file.

    Returns:
        List of (line_number, original_line, suggestion)
    """
    findings = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        for pattern, replacement_template in PATH_PATTERNS:
            matches = re.finditer(pattern, line)
            for match in matches:
                original = match.group(0)
                filename = match.group(1)

                # Check if we have a known mapping
                if filename in KNOWN_FILES:
                    suggestion = KNOWN_FILES[filename]
                else:
                    suggestion = replacement_template.format(filename)

                findings.append((line_num, line.strip(), original, suggestion))

    return findings


def print_migration_report(file_path: Path, findings: List[Tuple[int, str, str, str]]):
    """Print a migration report for the file."""
    print(f"\n{'='*70}")
    print(f"Migration Report: {file_path}")
    print(f"{'='*70}\n")

    if not findings:
        print("✓ No hardcoded paths found! This file may already be migrated.\n")
        return

    print(f"Found {len(findings)} hardcoded path(s) to migrate:\n")

    for line_num, line, original, suggestion in findings:
        print(f"Line {line_num}:")
        print(f"  Current:  {line}")
        print(f"  Replace:  {original}")
        print(f"  With:     {suggestion}")
        print()

    print(f"{'='*70}")
    print("Migration Steps:")
    print(f"{'='*70}\n")

    print("1. Add imports at the top of the file:")
    print("   import sys")
    print("   from pathlib import Path")
    print()
    print("   # Add project root to Python path")
    print("   sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))")
    print()
    print("   from backend.src.config import get_paths")
    print()
    print("2. Get paths instance:")
    print("   paths = get_paths()")
    print()
    print("3. Replace the hardcoded paths with the suggestions above")
    print()
    print("4. Test the script to ensure it works")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_to_config.py <file_path>")
        print("\nExample:")
        print("  python migrate_to_config.py backend/src/explainability/counterfactual_explainer.py")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    if file_path.suffix != '.py':
        print(f"Warning: {file_path} is not a Python file")

    findings = find_hardcoded_paths(file_path)
    print_migration_report(file_path, findings)


if __name__ == "__main__":
    main()
