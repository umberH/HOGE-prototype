# Migration Guide - HOGE Framework Restructuring

## Summary of Changes

We've completely restructured the HOGE repository to follow best practices, add configuration management, hyperparameter tuning, and a web interface.

## What Changed?

### ✅ New Directory Structure

**Before:**
```
Phd-Experiment1/
├── df1_loan.csv
├── train_model.py
├── llm_explainer.py
└── (all files in root)
```

**After:**
```
Phd-Experiment1/
├── config/                    # JSON configuration files
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Processed data
│   └── evaluation/           # Evaluation results
├── models/                   # Trained models
├── src/                      # Source code (modular)
├── figures/                  # Images and diagrams
├── paper/                    # LaTeX paper
├── scripts/                  # Utility scripts
├── app.py                    # Streamlit web interface
└── requirements.txt          # Dependencies
```

### ✅ New Features

1. **JSON Configuration System**
   - `config/model_hyperparameters.json` - Define models & tuning grids
   - `config/pipeline_config.json` - Data processing settings
   - `config/neo4j_config.json` - Knowledge graph config
   - `config/llm_config.json` - LLM prompt templates

2. **Hyperparameter Tuning**
   - `src/models/hyperparameter_tuner.py` - Optuna/GridSearch/RandomSearch
   - Support for multiple models: XGBoost, Random Forest, Logistic Regression
   - Automatic tuning with configurable trials

3. **Web Interface**
   - `app.py` - Streamlit app for interactive explanations
   - Features: single explanations, batch analysis, settings

4. **Config-Driven Training**
   - `src/models/train_config.py` - New training script using configs
   - Old `train_model.py` still works with updated paths

5. **Better Documentation**
   - `README_NEW.md` - Comprehensive guide
   - `QUICKSTART.md` - Step-by-step setup
   - This migration guide

## How to Use the New Structure

### Option 1: Use Old Scripts (Still Work!)

```bash
# Old way still works with updated paths
python train_model.py
python extract_shap_values.py
python llm_explainer.py LP001006
```

### Option 2: Use New Config-Driven Approach

```bash
# Train with default config
python src/models/train_config.py --model xgboost_monotonic

# Train with hyperparameter tuning
python src/models/train_config.py --model xgboost_monotonic --tune

# Train different models
python src/models/train_config.py --model random_forest --tune
```

### Option 3: Use Web Interface

```bash
streamlit run app.py
```

Open http://localhost:8501 and use the interactive UI.

## Configuration Files Explained

### `config/model_hyperparameters.json`

Defines all models with their hyperparameters:

```json
{
  "xgboost_monotonic": {
    "model_type": "XGBClassifier",
    "default_params": {...},
    "tuning_grid": {...},
    "monotonic_constraints": {...}
  }
}
```

**To add a new model:**
1. Add entry to this JSON file
2. Run: `python src/models/train_config.py --model your_model --tune`

### `config/pipeline_config.json`

Controls data processing and tuning:

```json
{
  "hyperparameter_tuning": {
    "method": "optuna",      // or "grid" or "random"
    "n_trials": 100,
    "timeout_seconds": 3600
  }
}
```

## Breaking Changes

### File Paths Updated

All scripts now use new paths:

| Old Path | New Path |
|----------|----------|
| `df1_loan.csv` | `data/raw/df1_loan.csv` |
| `shap_long.csv` | `data/processed/shap_long.csv` |
| `loan_xgb_monotonic.joblib` | `models/loan_xgb_monotonic.joblib` |
| `eval_*.json` | `data/evaluation/eval_*.json` |

### Scripts Updated Automatically

The following scripts have been updated with new paths:
- ✅ `train_model.py`
- ✅ `extract_shap_values.py`
- ✅ `neo_loader.py`
- ✅ `llm_explainer.py`
- ✅ `evaluation_system.py`
- ✅ `evaluation_human.py`
- ✅ `counterfactual_explainer.py`
- ✅ `predict_new.py`
- ✅ `explain_xgb_shap.py`

## Testing the New Structure

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Run Quick Test

```bash
# Test config loading
python -c "from src.utils.config_loader import get_config_loader; print(get_config_loader().get_pipeline_config())"

# Test web interface
streamlit run app.py
```

### 4. Run Full Pipeline (Optional)

```bash
# 1. Train model
python src/models/train_config.py --model xgboost_monotonic

# 2. Extract SHAP
python extract_shap_values.py

# 3. Load KG
python neo_loader.py

# 4. Generate explanations
streamlit run app.py
```

## Git Workflow

### Current State

- **Branch:** `develop`
- **Status:** All changes committed
- **Main branch:** Unchanged (safe!)

### To Merge to Main (After Testing)

```bash
# Test thoroughly on develop branch first!
streamlit run app.py  # Test web interface
python src/models/train_config.py --model xgboost_monotonic  # Test training

# When ready, merge to main:
git checkout main
git merge develop
git push origin main
```

### To Revert to Old Structure (If Needed)

```bash
git checkout main  # Go back to original structure
```

The main branch still has the old structure, so it's safe to test!

## What to Document in Your Paper

### New Contributions

1. **Configuration Management**
   - "We implement a JSON-based configuration system enabling reproducible experiments with different hyperparameters"

2. **Hyperparameter Optimization**
   - "We employ Optuna for Bayesian hyperparameter optimization, achieving [X]% improvement in ROC-AUC"

3. **Interactive Interface**
   - "We provide a web-based interface using Streamlit, enabling non-technical stakeholders to explore explanations"

4. **Model Comparison**
   - "We compare XGBoost with monotonic constraints against baseline models (Random Forest, Logistic Regression)"

## Common Tasks

### Change Tuning Strategy

Edit `config/model_hyperparameters.json`:

```json
{
  "xgboost_monotonic": {
    "tuning_strategy": "optuna"  // Change to "grid" or "random"
  }
}
```

### Add New Model

1. Add to `config/model_hyperparameters.json`:

```json
{
  "my_new_model": {
    "model_type": "XGBClassifier",
    "default_params": {...},
    "tuning_grid": {...}
  }
}
```

2. Train:

```bash
python src/models/train_config.py --model my_new_model --tune
```

### Customize LLM Prompts

Edit `config/llm_config.json`:

```json
{
  "prompting": {
    "grounding_rules": [
      "Add your custom grounding rules here"
    ]
  }
}
```

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the repository root
cd Phd-Experiment1

# Try running:
python -m src.models.train_config --model xgboost_monotonic
```

### Path Errors

All paths should now be relative to repository root. If you get "file not found":

1. Check you're in repository root: `pwd` should show `.../Phd-Experiment1`
2. Verify file exists: `ls data/raw/df1_loan.csv`

### Config Not Found

```bash
# Verify config directory exists
ls config/

# Should show:
# - model_hyperparameters.json
# - pipeline_config.json
# - neo4j_config.json
# - llm_config.json
```

## Support

- **Old scripts:** Still work! Use if you prefer the original approach
- **New scripts:** Use for config-driven, reproducible experiments
- **Web interface:** Use for interactive exploration

## Next Steps

1. ✅ Test the new structure on develop branch
2. ⏳ Run hyperparameter tuning experiments
3. ⏳ Compare multiple models
4. ⏳ Use web interface for explanation exploration
5. ⏳ Merge to main when satisfied

## Files Changed

**Added:**
- `config/` directory (4 JSON files)
- `src/` directory (modular code)
- `app.py` (web interface)
- `requirements.txt`
- `README_NEW.md`, `QUICKSTART.md`, this file

**Moved:**
- All data files → `data/raw/` or `data/processed/`
- Evaluation results → `data/evaluation/`
- Model file → `models/`
- Paper → `paper/`
- Images → `figures/`

**Modified:**
- 10 Python scripts updated with new file paths

**Unchanged:**
- Original scripts still work!
- Main branch unchanged (safe fallback)

---

**Questions?** Check `README_NEW.md` or `QUICKSTART.md`
