# Centralized Configuration System

## Overview

This module provides centralized configuration for all file paths and environment settings across the project. Instead of hardcoding paths in individual scripts, all paths are defined in one place.

## Features

- **Centralized Path Management**: All file paths defined in `paths.py`
- **Environment Variable Overrides**: Paths can be overridden via `.env` file
- **Auto-Detection**: Project root is automatically detected
- **Type Safety**: Uses `pathlib.Path` for cross-platform compatibility
- **Singleton Pattern**: One configuration instance shared across the application

## Quick Start

### Basic Usage

```python
from backend.src.config import get_paths

# Get the global paths instance
paths = get_paths()

# Use paths in your code
import pandas as pd
df = pd.read_csv(paths.RAW_DATA_LOAN)

import joblib
model = joblib.load(paths.MODEL_XGB_MONOTONIC)
```

### Available Paths

#### Data Files
```python
paths.RAW_DATA_LOAN              # .resources/data/raw/df1_loan.csv
paths.RAW_DATA_NEW_APPLICATIONS  # .resources/data/raw/new_applications.csv
paths.SHAP_LONG                  # .resources/data/processed/shap_long.csv
paths.SHAP_WIDE                  # .resources/data/processed/shap_wide.csv
paths.SCORED_APPLICATIONS        # .resources/data/processed/scored_applications_xgb.csv
```

#### Model Files
```python
paths.MODEL_XGB_MONOTONIC        # .resources/models/loan_xgb_monotonic.joblib
paths.MODEL_PROVENANCE           # .resources/models/model_provenance.json
```

#### Evaluation Files
```python
paths.EVAL_COUNTERFACTUAL        # .resources/data/evaluation/eval_counterfactual.json
paths.EVAL_HUMAN                 # .resources/data/evaluation/eval_human.json
paths.EVAL_SYSTEM                # .resources/data/evaluation/eval_system.json
paths.EVAL_FIDELITY              # .resources/data/evaluation/eval_fidelity.json
```

#### Directories
```python
paths.PROJECT_ROOT               # Project root directory
paths.DATA_DIR                   # .resources/data/
paths.RAW_DATA_DIR               # .resources/data/raw/
paths.PROCESSED_DATA_DIR         # .resources/data/processed/
paths.MODELS_DIR                 # .resources/models/
paths.EVALUATION_DIR             # .resources/data/evaluation/
paths.PROVENANCE_DIR             # .resources/data/provenance/
```

## Environment Variable Overrides

You can override any path using environment variables in your `.env` file:

```bash
# Override entire directories
DATA_DIR=/custom/path/to/data
MODELS_DIR=/custom/path/to/models

# Override specific files
RAW_DATA_LOAN=/custom/path/to/loan_data.csv
MODEL_XGB_MONOTONIC=/custom/path/to/model.joblib
```

## API Configuration

For Neo4j and OpenAI configuration, use the existing adapter:

```python
from backend.src.api.adapters.config import get_neo4j_config, get_openai_config

# Get Neo4j configuration
neo4j = get_neo4j_config()
# Returns: {uri, user, password, database, environment}

# Get OpenAI configuration
openai = get_openai_config()
# Returns: {api_key, model}
```

## Migration Guide

### Before (Hardcoded Paths)

```python
import pandas as pd
import joblib

# Bad: Hardcoded paths
df = pd.read_csv("data/raw/df1_loan.csv")
model = joblib.load("models/loan_xgb_monotonic.joblib")
output_df.to_csv("data/processed/output.csv")
```

### After (Centralized Config)

```python
import pandas as pd
import joblib
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.src.config import get_paths

# Good: Centralized configuration
paths = get_paths()
df = pd.read_csv(paths.RAW_DATA_LOAN)
model = joblib.load(paths.MODEL_XGB_MONOTONIC)
output_df.to_csv(paths.PROCESSED_DATA_DIR / "output.csv")
```

## Adding New Paths

To add a new path to the configuration:

1. **Open `backend/src/config/paths.py`**
2. **Add the path in the appropriate section**:

```python
class PathConfig:
    def __init__(self, project_root: Optional[Path] = None):
        # ... existing code ...

        # Add your new path here
        self.MY_NEW_FILE = self.DATA_DIR / "my_new_file.csv"
```

3. **Use it in your scripts**:

```python
from backend.src.config import get_paths

paths = get_paths()
df = pd.read_csv(paths.MY_NEW_FILE)
```

## Utility Methods

### Check if Files Exist

```python
paths = get_paths()

if paths.RAW_DATA_LOAN.exists():
    df = pd.read_csv(paths.RAW_DATA_LOAN)
else:
    print(f"File not found: {paths.RAW_DATA_LOAN}")
```

### Create Missing Directories

```python
paths = get_paths()
paths.ensure_directories()  # Creates all necessary directories
```

### Get Relative Paths

```python
paths = get_paths()
relative = paths.get_relative_path(paths.MODEL_XGB_MONOTONIC)
print(relative)  # .resources/models/loan_xgb_monotonic.joblib
```

## Testing

To test the path configuration:

```bash
python backend/src/config/paths.py
```

This will print all paths and check if key files exist.

## Updated Scripts

The following scripts have been migrated to use centralized configuration:

- ✅ `backend/src/models/train_model.py`
- ✅ `backend/src/models/predict_new.py`
- ✅ `backend/src/explainability/extract_shap_values.py`
- ✅ `backend/src/knowledge_graph/neo_loader.py`

## Benefits

1. **Single Source of Truth**: All paths in one place
2. **Easy Maintenance**: Change paths in one file, not dozens
3. **Environment Flexibility**: Override paths for different environments
4. **Cross-Platform**: Works on Windows, Mac, and Linux
5. **Type Safety**: IDE autocomplete and type hints
6. **Error Prevention**: Reduces path-related bugs

## Troubleshooting

### Import Error: No module named 'backend'

Make sure you add the project root to `sys.path`:

```python
import sys
from pathlib import Path

# Add this at the top of your script
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.src.config import get_paths
```

### Path Not Found

Check if the path exists:

```python
paths = get_paths()
print(f"Looking for: {paths.RAW_DATA_LOAN}")
print(f"Exists: {paths.RAW_DATA_LOAN.exists()}")
```

### Override Not Working

Make sure your `.env` file is in the project root and loaded:

```python
from dotenv import load_dotenv
load_dotenv()  # Call this before importing get_paths
```

## Support

For issues or questions, please refer to the project documentation or create an issue in the GitHub repository.
