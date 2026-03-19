# HOGE: Human-Centric Ontology-Grounded Explanation Framework

**PhD Experiment 1** — Prototype implementation and evaluation of the HOGE framework for explainable AI in loan decisioning.

## Overview

HOGE integrates three technologies to produce human-readable, semantically grounded explanations for ML decisions:

- **Component A — Predictive Model + Technical Explainer**: XGBoost with monotonic constraints + SHAP feature attribution
- **Component B — Semantic Knowledge Representation**: Neo4j knowledge graph with domain ontology, policy rules, and risk factors
- **Component C — Narrative Explainer**: GPT-4o constrained by ontology-grounded evidence to generate audience-adaptive explanations

## Pipeline Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌───────────────┐    ┌─────────────────┐
│  train_model │───▶│extract_shap_values│───▶│  neo_loader  │───▶│ llm_explainer │───▶│  Evaluation     │
│     .py      │    │       .py        │    │     .py      │    │     .py       │    │  (system +      │
│              │    │                  │    │              │    │               │    │   human)        │
│  XGBoost +   │    │  SHAP Tree-     │    │  Neo4j KG    │    │  GPT-4o       │    │                 │
│  monotonic   │    │  Explainer      │    │  ingestion   │    │  constrained  │    │                 │
│  constraints │    │  (22 features)  │    │  (23K+ nodes)│    │  narrative    │    │                 │
└─────────────┘    └──────────────────┘    └─────────────┘    └───────────────┘    └─────────────────┘
```

## How to Run (Step by Step)

### Prerequisites

```bash
pip install xgboost scikit-learn shap pandas neo4j openai python-dotenv openpyxl matplotlib networkx
```

Create a `.env` file in the project root:
```
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-password>
OPENAI_API_KEY=<your-key>
OPENAI_MODEL=gpt-4o
```

### Step 1: Train the Model
```bash
python train_model.py
```
**Output**: `loan_xgb_monotonic.joblib`, `scored_applications_xgb.csv`
- Trains XGBoost with monotonic constraints on 500 loan applications
- ROC-AUC: 0.839, Accuracy: 80%

### Step 2: Extract SHAP Values
```bash
python extract_shap_values.py
```
**Output**: `shap_wide.csv`, `shap_long.csv`
- Computes SHAP values for all 500 applications across 22 features
- `shap_long.csv` (11,000 rows) is the primary KG input

### Step 3: Load Knowledge Graph into Neo4j
```bash
python neo_loader.py
```
**Requires**: Running Neo4j instance
- Creates 23,000+ nodes: Applicant, LoanApplication, Feature, FeatureValue, ModelExplanation, FeatureContribution, PolicyRule, RiskFactor
- Automatically detects and links policy violations

### Step 4: Generate LLM Explanations
```bash
python llm_explainer.py LP001006
```
**Output**: `explanation_LP001006.json`
- Queries KG for complete evidence bundle
- Sends constraint-aware prompt to GPT-4o
- Returns structured JSON with summary, drivers, violations, narrative, provenance

### Step 5: Run System Evaluation
```bash
python evaluation_system.py --all --n-samples 10
```
**Output**: `eval_faithfulness.json`, `eval_hallucination.json`, `eval_retrieval.json`, `eval_system_all.json`

| Metric | Result |
|--------|--------|
| Avg Faithfulness Score | 0.760 |
| Semantic Fidelity (SF) | 0.793 |
| Evidence Coverage (EC) | 0.907 |
| Hallucination Rate (HRC) | 0.207 |
| Precision@3/5/10 | 1.000 |

### Step 6: Run Human-Centric Evaluation
```bash
python evaluation_human.py --n-samples 10
```
**Output**: `eval_human_results.json`, `hoge_evaluation_workbook.xlsx`

| Metric | Result |
|--------|--------|
| Prediction Accuracy | 10/10 (100%) |
| SHAP-Prediction Alignment | 10/10 (100%) |
| SHAP Top-5 Coverage in LLM | 96% |
| Overall Feature Coverage | 61.1% |

### Optional: Generate KG Visualizations
```bash
python generate_kg_visualizations.py
```
**Output**: `neo4j_kg_schema.png`, `neo4j_kg_instance.png`

## File Reference

### Source Code

| File | Purpose |
|------|---------|
| `train_model.py` | Train XGBoost model with monotonic constraints |
| `extract_shap_values.py` | Compute SHAP values for all applications |
| `explain_xgb_shap.py` | Standalone SHAP explanation (visual plots) |
| `neo_loader.py` | Ingest data into Neo4j knowledge graph |
| `llm_explainer.py` | Generate constraint-aware LLM narratives (Component C) |
| `predict_new.py` | Score new applications with trained model |
| `evaluation_system.py` | System evaluation: faithfulness, hallucination, retrieval |
| `evaluation_human.py` | Human-centric evaluation: alignment, traceability, Excel workbook |
| `generate_kg_visualizations.py` | Generate KG schema and instance PNG visualizations |

### Data Files

| File | Description |
|------|-------------|
| `df1_loan.csv` | Original loan application dataset (500 rows) |
| `new_applications.csv` | Sample new applications for prediction |
| `scored_applications_xgb.csv` | Model predictions with probabilities |
| `shap_wide.csv` | SHAP values in wide format (500 × 22) |
| `shap_long.csv` | SHAP values in long format (11,000 rows) — primary KG input |

### Evaluation Outputs

| File | Description |
|------|-------------|
| `eval_faithfulness.json` | Per-application perturbation test results |
| `eval_hallucination.json` | Per-application SF, EC, HRC scores |
| `eval_retrieval.json` | Per-application Precision@K scores |
| `eval_system_all.json` | Combined system evaluation summary |
| `eval_human_results.json` | Human-centric evaluation data (alignment, intermediate outcomes) |
| `hoge_evaluation_workbook.xlsx` | 4-sheet Excel workbook for supervisor review |
| `explanation_LP001006.json` | Example structured LLM explanation |

### Paper & Figures

| File | Description |
|------|-------------|
| `paper_HOGE_updated.tex` | Full LaTeX paper (Springer LNBIP format) with 34 references |
| `Architecture.png` | HOGE system architecture diagram |
| `neo4j_kg_schema.png` | KG ontology schema visualization from Neo4j |
| `neo4j_kg_instance.png` | KG instance subgraph for LP001006 |

## Excel Workbook Sheets (`hoge_evaluation_workbook.xlsx`)

The evaluation workbook contains four sheets for supervisor review:

1. **Prediction SHAP Alignment** — Verifies each application's prediction direction matches net SHAP sum
2. **Intermediate Outcomes** — Traces all 8 pipeline stages per application (raw features → model → SHAP → policy → KG → LLM)
3. **LLM Grounding CrossCheck** — Shows which features/rules the LLM referenced vs. what was available
4. **Summary Dashboard** — Aggregated metrics and pass/fail status

## Key Evaluation Results Summary

| Category | Metric | Value |
|----------|--------|-------|
| Model | ROC-AUC | 0.839 |
| Faithfulness | Avg Score | 0.760 (9/10 pass) |
| Grounding | Semantic Fidelity | 0.793 |
| Grounding | Evidence Coverage | 0.907 |
| Grounding | Hallucination Rate | 20.7% (embellishments only) |
| Retrieval | Precision@K | 1.000 (perfect) |
| Human-Centric | Prediction Accuracy | 100% |
| Human-Centric | SHAP-Prediction Alignment | 100% |
| Human-Centric | SHAP Top-5 LLM Coverage | 96% |

## Technology Stack

- **Python 3.14** | XGBoost | scikit-learn | SHAP | pandas
- **Neo4j** — Graph database for knowledge graph
- **OpenAI GPT-4o** — Narrative explanation generation
- **openpyxl** — Excel workbook generation
- **matplotlib + networkx** — KG visualizations
- **LaTeX** (Springer LNBIP `svmultln`) — Paper format
