"""
Update file paths in existing Python scripts to match new directory structure.
Run this script after reorganizing the repository.
"""

import re
from pathlib import Path

# Define path replacements
PATH_REPLACEMENTS = {
    # Data files
    '"df1_loan.csv"': '"data/raw/df1_loan.csv"',
    '"new_applications.csv"': '"data/raw/new_applications.csv"',
    '"shap_long.csv"': '"data/processed/shap_long.csv"',
    '"shap_wide.csv"': '"data/processed/shap_wide.csv"',
    '"scored_applications_xgb.csv"': '"data/processed/scored_applications_xgb.csv"',

    # Model files
    '"loan_xgb_monotonic.joblib"': '"models/loan_xgb_monotonic.joblib"',

    # Evaluation files
    '"eval_counterfactual.json"': '"data/evaluation/eval_counterfactual.json"',
    '"eval_faithfulness.json"': '"data/evaluation/eval_faithfulness.json"',
    '"eval_hallucination.json"': '"data/evaluation/eval_hallucination.json"',
    '"eval_retrieval.json"': '"data/evaluation/eval_retrieval.json"',
    '"eval_system_all.json"': '"data/evaluation/eval_system_all.json"',
    '"eval_human_results.json"': '"data/evaluation/eval_human_results.json"',
    '"hoge_evaluation_workbook.xlsx"': '"data/evaluation/hoge_evaluation_workbook.xlsx"',

    # Explanation output pattern
    r'f"explanation_{application_id}\.json"': r'f"data/evaluation/explanation_{application_id}.json"',
}

# Files to update
FILES_TO_UPDATE = [
    'train_model.py',
    'extract_shap_values.py',
    'neo_loader.py',
    'llm_explainer.py',
    'evaluation_system.py',
    'evaluation_human.py',
    'counterfactual_explainer.py',
    'predict_new.py',
    'explain_xgb_shap.py',
    'app.py',
]


def update_file(file_path: Path):
    """Update file paths in a Python script."""
    if not file_path.exists():
        print(f"[!] File not found: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes_made = 0

    # Apply replacements
    for old_path, new_path in PATH_REPLACEMENTS.items():
        if old_path.startswith('r"') or 'f"' in old_path:
            # Regex pattern
            pattern = old_path.replace('\\', '\\\\')
            replacement = new_path
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                changes_made += content.count(old_path.replace('r"', '').replace('"', ''))
                content = new_content
        else:
            # Simple string replacement
            if old_path in content:
                content = content.replace(old_path, new_path)
                changes_made += original_content.count(old_path) - content.count(old_path)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[+] Updated {file_path.name} ({changes_made} changes)")
        return True
    else:
        print(f"[-] No changes needed for {file_path.name}")
        return False


def main():
    """Update all Python scripts."""
    print("Updating file paths in Python scripts...\n")

    updated_count = 0

    for filename in FILES_TO_UPDATE:
        file_path = Path(filename)
        if update_file(file_path):
            updated_count += 1

    print(f"\n{'='*60}")
    print(f"Updated {updated_count}/{len(FILES_TO_UPDATE)} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
