# HOGE Framework - Complete Step-by-Step Procedure

## Overview

This guide walks you through the entire HOGE pipeline from scratch.

## Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] Git repository cloned
- [ ] On `develop` branch
- [ ] Virtual environment created (`venv/`)
- [ ] Dependencies installed

---

## Phase 1: Environment Setup (One-time)

### Step 1.1: Verify You're on Develop Branch

```bash
git branch
# Should show: * develop
```

If not:
```bash
git checkout develop
```

### Step 1.2: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

You should see `(venv)` in your prompt.

### Step 1.3: Verify Dependencies

```bash
venv\Scripts\python.exe -c "import pandas, sklearn, xgboost, shap; print('Core ML libraries OK')"
```

If this fails, reinstall:
```bash
venv\Scripts\python.exe -m pip install numpy pandas scikit-learn xgboost shap joblib
```

---

## Phase 2: Train Model & Extract SHAP (No external services needed!)

### Step 2.1: Train XGBoost Model

**Option A: Quick training (default hyperparameters)**
```bash
venv\Scripts\python.exe src/models/train_model.py
```

**Option B: Config-driven training**
```bash
venv\Scripts\python.exe src/models/train_config.py --model xgboost_monotonic
```

**Option C: With hyperparameter tuning (takes longer)**
```bash
venv\Scripts\python.exe src/models/train_config.py --model xgboost_monotonic --tune
```

**Expected Output:**
- ✅ `models/loan_xgb_monotonic.joblib` (trained model)
- ✅ `data/processed/scored_applications_xgboost_monotonic.csv` (predictions)
- 📊 ROC-AUC: ~0.839, Accuracy: ~80%

**Time:** 30 seconds - 10 minutes (depending on tuning)

### Step 2.2: Extract SHAP Values

```bash
venv\Scripts\python.exe src/explainability/extract_shap_values.py
```

**Expected Output:**
- ✅ `data/processed/shap_long.csv` (11,000 rows - one per feature per application)
- ✅ `data/processed/shap_wide.csv` (500 rows - one per application)

**Time:** ~2 minutes

---

## Phase 3: Knowledge Graph Setup (Optional but recommended)

### Step 3.1: Start Neo4j Database

**Option A: Docker (Recommended)**
```bash
docker run -d --name neo4j-hoge \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test1234 \
  neo4j:latest
```

**Option B: Neo4j Desktop**
1. Download from https://neo4j.com/download/
2. Create new database
3. Set password to `test1234`
4. Start database

**Verify Neo4j is running:**
- Open browser: http://localhost:7474
- Login: username `neo4j`, password `test1234`
- You should see Neo4j Browser

### Step 3.2: Configure Environment

```bash
# Edit .env file
notepad .env  # Windows
# or
nano .env     # Linux/Mac
```

Set:
```
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=test1234

# Leave OpenAI empty for now
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o
```

### Step 3.3: Load Data into Neo4j

```bash
venv\Scripts\python.exe src/knowledge_graph/neo_loader.py
```

**Expected Output:**
- ✅ Creates 23,000+ nodes
- ✅ Loads: Applications, Features, SHAP values, Policy rules

**Time:** ~30 seconds

**Verify in Neo4j Browser:**
```cypher
MATCH (n) RETURN count(n)
// Should return ~23,000+
```

### Step 3.4: Generate Counterfactual Scenarios (Optional)

```bash
venv\Scripts\python.exe src/explainability/counterfactual_explainer.py --all
```

**Expected Output:**
- ✅ `data/evaluation/eval_counterfactual.json`
- 📊 Shows minimal feature changes to flip predictions

**Then reload KG to include counterfactuals:**
```bash
venv\Scripts\python.exe src/knowledge_graph/neo_loader.py
```

---

## Phase 4: LLM Explanations (Requires OpenAI API)

### Step 4.1: Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Copy the key (starts with `sk-...`)

### Step 4.2: Add API Key to .env

```bash
notepad .env
```

Update:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Step 4.3: Generate Single Explanation

```bash
venv\Scripts\python.exe src/explainability/llm_explainer.py LP001006
```

**Expected Output:**
- ✅ `data/evaluation/explanation_LP001006.json`
- 📄 JSON with: summary, positive/negative drivers, narrative, counterfactual what-if

**Try different applications:**
```bash
venv\Scripts\python.exe src/explainability/llm_explainer.py LP001002
venv\Scripts\python.exe src/explainability/llm_explainer.py LP001003
```

---

## Phase 5: Evaluation

### Step 5.1: System Evaluation

```bash
venv\Scripts\python.exe src/evaluation/evaluation_system.py --all --n-samples 10
```

**Expected Output:**
- ✅ `data/evaluation/eval_faithfulness.json` (perturbation tests)
- ✅ `data/evaluation/eval_hallucination.json` (grounding metrics)
- ✅ `data/evaluation/eval_retrieval.json` (precision@K)
- ✅ `data/evaluation/eval_system_all.json` (combined)

**Metrics:**
- Faithfulness Score: ~0.760
- Semantic Fidelity: ~0.793
- Hallucination Rate: ~20.7%

**Time:** ~5-10 minutes (10 samples)

### Step 5.2: Human-Centric Evaluation

```bash
venv\Scripts\python.exe src/evaluation/evaluation_human.py --n-samples 10
```

**Expected Output:**
- ✅ `data/evaluation/eval_human_results.json`
- ✅ `data/evaluation/hoge_evaluation_workbook.xlsx` (Excel with 4 sheets)

**Open Excel workbook to review:**
- Sheet 1: Prediction vs SHAP Alignment
- Sheet 2: Intermediate Outcomes
- Sheet 3: LLM Grounding Cross-Check
- Sheet 4: Summary Dashboard

**Time:** ~10-15 minutes (10 samples)

---

## Phase 6: Web Interface (Optional)

### Step 6.1: Install Streamlit (if not already)

```bash
venv\Scripts\python.exe -m pip install streamlit plotly
```

### Step 6.2: Launch App

```bash
venv\Scripts\streamlit.exe run app.py
```

**Expected:**
- 🌐 Opens browser at http://localhost:8501
- Interactive UI for exploring explanations

### Step 6.3: Use the App

1. Navigate to "🔍 Explain Application"
2. Enter Loan ID: `LP001006`
3. Click "Generate Explanation"
4. Explore SHAP values, policy violations, counterfactuals

---

## Phase 7: Compare Multiple Models (Advanced)

### Step 7.1: Train Baseline Models

```bash
# XGBoost without monotonic constraints
venv\Scripts\python.exe src/models/train_config.py --model xgboost_baseline --tune

# Random Forest
venv\Scripts\python.exe src/models/train_config.py --model random_forest --tune

# Logistic Regression
venv\Scripts\python.exe src/models/train_config.py --model logistic_regression --tune
```

### Step 7.2: Compare Results

Check tuning results in:
- `data/evaluation/tuning_results_xgboost_monotonic.json`
- `data/evaluation/tuning_results_xgboost_baseline.json`
- `data/evaluation/tuning_results_random_forest.json`
- `data/evaluation/tuning_results_logistic_regression.json`

---

## Quick Reference: Common Commands

### Training
```bash
# Quick train
venv\Scripts\python.exe src/models/train_model.py

# Config-driven with tuning
venv\Scripts\python.exe src/models/train_config.py --model xgboost_monotonic --tune
```

### SHAP
```bash
venv\Scripts\python.exe src/explainability/extract_shap_values.py
```

### Knowledge Graph
```bash
# Load data
venv\Scripts\python.exe src/knowledge_graph/neo_loader.py

# Visualize schema
venv\Scripts\python.exe src/knowledge_graph/generate_kg_visualizations.py
```

### Explanations
```bash
# Single explanation
venv\Scripts\python.exe src/explainability/llm_explainer.py LP001006

# Counterfactuals
venv\Scripts\python.exe src/explainability/counterfactual_explainer.py --all
```

### Evaluation
```bash
# System metrics
venv\Scripts\python.exe src/evaluation/evaluation_system.py --all --n-samples 10

# Human-centric
venv\Scripts\python.exe src/evaluation/evaluation_human.py --n-samples 10
```

### Web Interface
```bash
venv\Scripts\streamlit.exe run app.py
```

---

## Troubleshooting

### Import Errors
```bash
# Verify you're in repository root
pwd  # Should show .../Phd-Experiment1

# Verify venv is activated
where python  # Should show .../venv/Scripts/python.exe

# Reinstall dependencies
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Neo4j Connection Error
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Or test connection
curl http://localhost:7474

# Verify .env has correct password
cat .env | grep NEO4J_PASSWORD
```

### OpenAI API Error
```bash
# Check API key is set
cat .env | grep OPENAI_API_KEY

# Test API key
venv\Scripts\python.exe -c "import openai; print('API key OK')"
```

### Path Not Found Errors
All paths are relative to repository root. Always run commands from:
```bash
cd c:\Users\dumb_\Documents\GitHub\Phd-Experiment1
```

---

## Minimal Pipeline (No External Services)

If you just want to train and analyze without Neo4j/OpenAI:

```bash
# 1. Train model
venv\Scripts\python.exe src/models/train_model.py

# 2. Extract SHAP
venv\Scripts\python.exe src/explainability/extract_shap_values.py

# 3. Done! You now have:
# - Trained model: models/loan_xgb_monotonic.joblib
# - SHAP values: data/processed/shap_long.csv
# - Predictions: data/processed/scored_applications_*.csv
```

---

## Full Pipeline (Everything)

```bash
# 1. Train
venv\Scripts\python.exe src/models/train_config.py --model xgboost_monotonic --tune

# 2. SHAP
venv\Scripts\python.exe src/explainability/extract_shap_values.py

# 3. Start Neo4j
docker run -d --name neo4j-hoge -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/test1234 neo4j

# 4. Load KG
venv\Scripts\python.exe src/knowledge_graph/neo_loader.py

# 5. Counterfactuals
venv\Scripts\python.exe src/explainability/counterfactual_explainer.py --all
venv\Scripts\python.exe src/knowledge_graph/neo_loader.py  # Reload

# 6. Explanations (add OpenAI key to .env first!)
venv\Scripts\python.exe src/explainability/llm_explainer.py LP001006

# 7. Evaluate
venv\Scripts\python.exe src/evaluation/evaluation_system.py --all --n-samples 100
venv\Scripts\python.exe src/evaluation/evaluation_human.py --n-samples 100

# 8. Launch web interface
venv\Scripts\streamlit.exe run app.py
```

---

## Next Steps After Setup

1. ✅ **Experiment with hyperparameters** - Edit `config/model_hyperparameters.json`
2. ✅ **Compare models** - Train XGBoost, Random Forest, Logistic Regression
3. ✅ **Analyze explanations** - Review Excel workbook from human evaluation
4. ✅ **Customize prompts** - Edit `config/llm_config.json`
5. ✅ **Write your paper** - Use results from `data/evaluation/`

---

**Questions? Check:**
- [QUICKSTART.md](QUICKSTART.md) - Quick setup guide
- [SETUP.md](SETUP.md) - Installation details
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - What changed in restructuring
- [README_NEW.md](README_NEW.md) - Full documentation
