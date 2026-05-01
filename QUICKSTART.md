# 🚀 HOGE Framework - Quick Start Guide

## What's New?

✨ **JSON-Based Configuration** - All hyperparameters now in `config/` directory
✨ **Hyperparameter Tuning** - Built-in Optuna, GridSearch, and RandomSearch support
✨ **Web Interface** - Interactive Streamlit app for exploring explanations
✨ **Better Structure** - Organized `src/` directory with proper modules

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- XGBoost, scikit-learn, SHAP
- Neo4j driver
- OpenAI API
- Optuna (hyperparameter tuning)
- Streamlit (web interface)

### 2. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env file with your credentials:
# - NEO4J_URI=neo4j://127.0.0.1:7687
# - NEO4J_PASSWORD=your_password
# - OPENAI_API_KEY=sk-your-key
```

### 3. Start Neo4j Database

**Using Docker:**
```bash
docker run -d -p 7687:7687 -p 7474:7474 \
  --name neo4j \
  -e NEO4J_AUTH=neo4j/test1234 \
  neo4j:latest
```

**Or use Neo4j Desktop** (download from neo4j.com)

### 4. Train Your First Model

**Option A: Quick Training (default params)**
```bash
python src/models/train_config.py --model xgboost_monotonic
```

**Option B: With Hyperparameter Tuning (recommended)**
```bash
python src/models/train_config.py --model xgboost_monotonic --tune
```

This will:
- Load data from `data/raw/df1_loan.csv`
- Run Optuna optimization (100 trials)
- Save best model to `models/xgboost_monotonic.joblib`
- Save predictions to `data/processed/scored_applications_xgboost_monotonic.csv`

### 5. Extract SHAP Values

```bash
python extract_shap_values.py
```

Outputs:
- `data/processed/shap_long.csv` - SHAP values in long format
- `data/processed/shap_wide.csv` - SHAP values in wide format

### 6. Load Knowledge Graph

```bash
python neo_loader.py
```

This creates 23,000+ nodes in Neo4j:
- LoanApplication, Applicant, Feature nodes
- SHAP explanations
- Policy rules and risk factors

### 7. Generate Counterfactuals (Optional)

```bash
python counterfactual_explainer.py --all
```

Then reload KG to include counterfactuals:
```bash
python neo_loader.py
```

### 8. Launch Web Interface 🌐

```bash
streamlit run app.py
```

Open http://localhost:8501 and:
1. Enter a Loan ID (e.g., `LP001006`)
2. Click "Generate Explanation"
3. Explore SHAP values, policy violations, and counterfactuals

---

## 📋 Configuration Files

All settings are in `config/` directory:

### `model_hyperparameters.json`
Defines 4 models with tuning grids:
- `xgboost_monotonic` - XGBoost with monotonic constraints ⭐
- `xgboost_baseline` - XGBoost without constraints
- `random_forest` - Random Forest baseline
- `logistic_regression` - Logistic Regression baseline

### `pipeline_config.json`
Controls:
- Data splitting (test_size, stratification)
- Feature engineering (DTI calculation)
- Hyperparameter tuning (Optuna settings)
- Evaluation metrics

### `neo4j_config.json`
Knowledge graph settings:
- Connection details
- Policy rules
- Risk factors

### `llm_config.json`
LLM explanation settings:
- Model selection (GPT-4o, GPT-4, etc.)
- Temperature, max_tokens
- Prompt templates

---

## 🎯 Common Tasks

### Compare Multiple Models

```bash
# Train all 4 models
python src/models/train_config.py --model xgboost_monotonic --tune
python src/models/train_config.py --model xgboost_baseline --tune
python src/models/train_config.py --model random_forest --tune
python src/models/train_config.py --model logistic_regression --tune

# Compare results in data/evaluation/tuning_results_*.json
```

### Change Tuning Strategy

Edit `config/model_hyperparameters.json`:

```json
{
  "xgboost_monotonic": {
    "tuning_strategy": "optuna",  // or "grid" or "random"
    ...
  }
}
```

Or edit `config/pipeline_config.json` for global Optuna settings:

```json
{
  "hyperparameter_tuning": {
    "enabled": true,
    "method": "optuna",
    "n_trials": 100,         // increase for better results
    "timeout_seconds": 3600,
    "parallel_jobs": 4
  }
}
```

### Run Full Evaluation

```bash
# System evaluation (100 samples)
python evaluation_system.py --all --n-samples 100

# Human-centric evaluation
python evaluation_human.py --n-samples 100

# Outputs:
# - eval_faithfulness.json
# - eval_hallucination.json
# - eval_retrieval.json
# - hoge_evaluation_workbook.xlsx
```

### Generate Single Explanation

```bash
# Command line
python llm_explainer.py LP001006

# Output: explanation_LP001006.json
```

---

## 🐛 Troubleshooting

### Neo4j Connection Error
```
Error: Unable to connect to Neo4j
```

**Solution:**
1. Check Neo4j is running: `docker ps` or Neo4j Desktop
2. Verify `.env` credentials match Neo4j
3. Try browser: http://localhost:7474

### OpenAI API Error
```
Error: OPENAI_API_KEY not set
```

**Solution:**
1. Add key to `.env`: `OPENAI_API_KEY=sk-...`
2. Check balance: https://platform.openai.com/usage

### Import Error (config_loader)
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
Run from repository root:
```bash
cd Phd-Experiment1
python src/models/train_config.py ...
```

### Streamlit Not Found
```
streamlit: command not found
```

**Solution:**
```bash
pip install streamlit
```

---

## 📚 Next Steps

1. **Read the Paper**: `paper_HOGE_clean.tex`
2. **Explore Visualizations**: `python generate_kg_visualizations.py`
3. **Customize Prompts**: Edit `config/llm_config.json`
4. **Add New Models**: Extend `config/model_hyperparameters.json`
5. **Deploy API**: Create FastAPI wrapper for production

---

## 💡 Pro Tips

1. **Start with small n_trials (20-30)** to test configuration, then increase to 100+
2. **Use Optuna pruning** to save time on bad trials
3. **Monitor GPU usage** for XGBoost (set `tree_method='gpu_hist'` if available)
4. **Cache SHAP values** - they're expensive to recompute
5. **Use batch explanations** in Streamlit for faster analysis

---

## 🆘 Need Help?

- GitHub Issues: [Report bugs/request features]
- Documentation: See `README_NEW.md`
- Config Examples: Check `config/*.json` files

Happy explaining! 🎉
