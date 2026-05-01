# HOGE Framework - Setup Instructions

## Quick Setup (5 minutes)

### 1. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate venv
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env and add your credentials:
# - NEO4J_PASSWORD
# - OPENAI_API_KEY
```

### 4. Launch Web Interface

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Full Setup (if running entire pipeline)

### 5. Start Neo4j (Optional - only if using Knowledge Graph)

**Option A: Docker**
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test1234 \
  neo4j:latest
```

**Option B: Neo4j Desktop**
- Download from https://neo4j.com/download/
- Create database with password `test1234`
- Start database

### 6. Load Data into Neo4j (Optional)

```bash
python src/knowledge_graph/neo_loader.py
```

## Verify Installation

```bash
# Test imports
python -c "import streamlit; import pandas; import neo4j; import openai; print('All dependencies installed!')"

# List installed packages
pip list
```

## Troubleshooting

### Virtual Environment Not Activating

**Windows:**
```bash
# Use PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

### Module Not Found Errors

```bash
# Make sure venv is activated (you should see (venv) in your prompt)
# Then reinstall
pip install -r requirements.txt
```

### Streamlit Command Not Found

```bash
# Install streamlit explicitly
pip install streamlit

# Or use Python module syntax
python -m streamlit run app.py
```

## Next Steps

After setup:
1. ✅ Verify web interface loads: `streamlit run app.py`
2. ⏳ (Optional) Train a model: `python src/models/train_config.py --model xgboost_monotonic`
3. ⏳ (Optional) Extract SHAP: `python src/explainability/extract_shap_values.py`
4. ⏳ (Optional) Load KG: `python src/knowledge_graph/neo_loader.py`

## Using the App Without Neo4j

The Streamlit app has a "Home" page that works without Neo4j.

To generate explanations, you'll need:
- ✅ Neo4j running with data loaded
- ✅ OpenAI API key in `.env`

## Development Setup (Optional)

For development with code quality tools:

```bash
pip install black flake8 mypy pytest
```

Format code:
```bash
black src/
```

Run tests:
```bash
pytest tests/
```
