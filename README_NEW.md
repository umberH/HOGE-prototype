# HOGE: Human-Centric Ontology-Grounded Explanation Framework

**PhD Experiment 1** — Prototype implementation and evaluation of the HOGE framework for explainable AI in loan decisioning.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Phd-Experiment1

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### 2. Train Model (Config-Driven)

```bash
# Train with default config
python src/models/train_config.py --model xgboost_monotonic

# Train with hyperparameter tuning
python src/models/train_config.py --model xgboost_monotonic --tune

# Train baseline models
python src/models/train_config.py --model random_forest --tune
python src/models/train_config.py --model logistic_regression
```

### 3. Load Knowledge Graph

```bash
# Start Neo4j (ensure it's running first)
python neo_loader.py
```

### 4. Generate Explanations

```bash
# Command line
python llm_explainer.py LP001006

# Web interface
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 📁 Repository Structure

```
Phd-Experiment1/
├── config/                         # JSON configuration files
│   ├── model_hyperparameters.json  # Model configs & tuning grids
│   ├── pipeline_config.json        # Data pipeline settings
│   ├── neo4j_config.json          # Knowledge graph config
│   └── llm_config.json            # LLM prompting config
│
├── data/
│   ├── raw/                       # Original datasets
│   │   └── df1_loan.csv
│   ├── processed/                 # Processed data
│   │   ├── shap_long.csv
│   │   └── shap_wide.csv
│   └── evaluation/                # Evaluation results
│       ├── eval_*.json
│       └── tuning_results_*.json
│
├── models/                        # Trained models
│   └── *.joblib
│
├── src/                          # Source code
│   ├── models/
│   │   ├── train_config.py       # Config-driven training
│   │   └── hyperparameter_tuner.py  # Optuna/GridSearch tuning
│   ├── explainability/
│   ├── knowledge_graph/
│   ├── evaluation/
│   └── utils/
│       └── config_loader.py      # JSON config management
│
├── app.py                        # Streamlit web interface
├── llm_explainer.py             # LLM explanation generator
├── neo_loader.py                # Neo4j KG loader
├── requirements.txt             # Python dependencies
└── .env.example                 # Environment template
```

## 🎛️ Configuration System

All system behavior is controlled via JSON files in `config/`:

### Model Hyperparameters (`model_hyperparameters.json`)

Define multiple models with tuning grids:

```json
{
  "xgboost_monotonic": {
    "model_type": "XGBClassifier",
    "default_params": {...},
    "tuning_grid": {...},
    "monotonic_constraints": {...}
  },
  "random_forest": {...},
  "logistic_regression": {...}
}
```

### Pipeline Configuration (`pipeline_config.json`)

Control data processing, feature engineering, and evaluation:

```json
{
  "data": {
    "test_size": 0.2,
    "stratify": true
  },
  "hyperparameter_tuning": {
    "enabled": true,
    "method": "optuna",
    "n_trials": 100
  }
}
```

## 🔬 Hyperparameter Tuning

The framework supports three tuning strategies:

1. **Optuna** (recommended): Bayesian optimization with pruning
2. **GridSearchCV**: Exhaustive grid search
3. **RandomizedSearchCV**: Random sampling

Example usage:

```python
from src.models.hyperparameter_tuner import HyperparameterTuner
from src.utils.config_loader import get_config_loader

# Load configs
config_loader = get_config_loader()
model_config = config_loader.load_model_config('xgboost_monotonic')
pipeline_config = config_loader.get_pipeline_config()

# Create tuner
tuner = HyperparameterTuner(model_config, pipeline_config)

# Tune (returns best params and score)
best_params, best_score = tuner.tune(X_train, y_train)
```

## 🌐 Web Interface Features

The Streamlit app (`app.py`) provides:

- **🔍 Single Application Explanations**: Enter a Loan ID and get instant explanations
- **📊 Batch Analysis**: Analyze multiple applications simultaneously
- **⚙️ Settings**: Configure Neo4j and OpenAI connections
- **📥 Export**: Download explanations as JSON

Launch with: `streamlit run app.py`

## 🧪 Evaluation

Run comprehensive evaluations:

```bash
# System evaluation (faithfulness, hallucination, retrieval)
python evaluation_system.py --all --n-samples 100

# Human-centric evaluation (alignment, grounding)
python evaluation_human.py --n-samples 100

# Generate Excel workbook
# Output: hoge_evaluation_workbook.xlsx
```

## 🔄 Workflow

**Full Pipeline:**

1. **Train Model** → `python src/models/train_config.py --model xgboost_monotonic --tune`
2. **Extract SHAP** → `python extract_shap_values.py`
3. **Load KG** → `python neo_loader.py`
4. **Counterfactuals** → `python counterfactual_explainer.py --all`
5. **Reload KG** → `python neo_loader.py` (to include counterfactuals)
6. **Generate Explanations** → `streamlit run app.py`

## 📊 Model Comparison

Train and compare multiple models:

```bash
# Train all models with tuning
python src/models/train_config.py --model xgboost_monotonic --tune
python src/models/train_config.py --model xgboost_baseline --tune
python src/models/train_config.py --model random_forest --tune
python src/models/train_config.py --model logistic_regression --tune
```

Results saved in `data/evaluation/tuning_results_*.json`

## 🐳 Neo4j Setup

You need Neo4j running to use the knowledge graph:

**Option 1: Docker**
```bash
docker run -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/test1234 \
  neo4j:latest
```

**Option 2: Neo4j Desktop**
- Download from https://neo4j.com/download/
- Create a database with password `test1234`
- Start the database

Then load data: `python neo_loader.py`

## 🎯 Key Features

✅ **Config-driven architecture**: All settings in JSON
✅ **Multiple model support**: XGBoost, Random Forest, Logistic Regression
✅ **Hyperparameter tuning**: Optuna, GridSearch, RandomSearch
✅ **Knowledge graph integration**: Neo4j with ontology
✅ **LLM explanations**: GPT-4o with grounding constraints
✅ **Web interface**: Interactive Streamlit app
✅ **Comprehensive evaluation**: Faithfulness, hallucination, retrieval metrics
✅ **Counterfactual analysis**: What-if scenarios

## 📖 Citation

```bibtex
@article{hoge2024,
  title={HOGE: Human-Centric Ontology-Grounded Explanation Framework},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## 📝 License

[Your License Here]
