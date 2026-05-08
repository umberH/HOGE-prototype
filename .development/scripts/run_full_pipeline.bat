@echo off
REM ========================================================================
REM HOGE Framework - Complete End-to-End Pipeline Execution Script
REM ========================================================================
REM This script runs the entire HOGE pipeline from start to finish
REM Author: Mostaqim Murshed
REM Date: 2026-05-01
REM ========================================================================

echo.
echo ========================================================================
echo HOGE FRAMEWORK - FULL PIPELINE EXECUTION
echo ========================================================================
echo.

REM Check if we're in the right directory
if not exist "src\" (
    echo ERROR: Please run this script from the repository root directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/9] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please ensure .venv exists: python -m venv .venv
    pause
    exit /b 1
)
echo     ✓ Virtual environment activated
echo.

REM ========================================================================
REM PHASE 1: MODEL TRAINING & SHAP EXTRACTION
REM ========================================================================

echo [2/9] Training XGBoost model with monotonic constraints...
python src\models\train_model.py
if errorlevel 1 (
    echo ERROR: Model training failed
    pause
    exit /b 1
)
echo     ✓ Model trained successfully
echo     ✓ Saved: models\loan_xgb_monotonic.joblib
echo.

echo [3/9] Extracting SHAP values for all applications...
python src\explainability\extract_shap_values.py
if errorlevel 1 (
    echo ERROR: SHAP extraction failed
    pause
    exit /b 1
)
echo     ✓ SHAP values extracted
echo     ✓ Saved: data\processed\shap_long.csv (11,000 rows)
echo     ✓ Saved: data\processed\shap_wide.csv (500 rows)
echo.

REM ========================================================================
REM PHASE 2: KNOWLEDGE GRAPH (Optional - requires Neo4j)
REM ========================================================================

echo [4/9] Checking Neo4j connection...
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Neo4j URI:', os.getenv('NEO4J_URI', 'NOT SET'))"

set /p RUN_NEO4J="Do you want to load data into Neo4j? (y/n): "
if /i "%RUN_NEO4J%"=="y" (
    echo.
    echo     Loading data into Neo4j knowledge graph...
    python src\knowledge_graph\neo_loader.py
    if errorlevel 1 (
        echo WARNING: Neo4j loading failed - continuing without KG
        echo Make sure Neo4j is running and .env is configured
    ) else (
        echo     ✓ Knowledge graph loaded (23,000+ nodes)
    )
) else (
    echo     ⊘ Skipping Neo4j knowledge graph loading
)
echo.

REM ========================================================================
REM PHASE 3: COUNTERFACTUAL ANALYSIS
REM ========================================================================

echo [5/9] Generating counterfactual what-if scenarios...
python src\explainability\counterfactual_explainer.py --all
if errorlevel 1 (
    echo WARNING: Counterfactual generation failed - continuing
) else (
    echo     ✓ Counterfactuals generated
    echo     ✓ Saved: data\evaluation\eval_counterfactual.json

    REM Reload Neo4j if it was loaded earlier
    if /i "%RUN_NEO4J%"=="y" (
        echo     Reloading Neo4j with counterfactuals...
        python src\knowledge_graph\neo_loader.py
        if not errorlevel 1 (
            echo     ✓ Counterfactuals loaded into Neo4j
        )
    )
)
echo.

REM ========================================================================
REM PHASE 4: LLM EXPLANATIONS (Optional - requires OpenAI API)
REM ========================================================================

echo [6/9] Checking OpenAI API configuration...
python -c "import os; from dotenv import load_dotenv; load_dotenv(); key=os.getenv('OPENAI_API_KEY', ''); print('OpenAI API Key:', 'SET' if key and len(key) > 10 else 'NOT SET')"

set /p RUN_LLM="Do you want to generate LLM explanations? (requires OpenAI API) (y/n): "
if /i "%RUN_LLM%"=="y" (
    echo.
    echo     Generating sample LLM explanations...
    echo     Application 1: LP001006
    python src\explainability\llm_explainer.py LP001006

    echo     Application 2: LP001002
    python src\explainability\llm_explainer.py LP001002

    echo     Application 3: LP001003
    python src\explainability\llm_explainer.py LP001003

    if errorlevel 1 (
        echo WARNING: LLM explanation generation failed
        echo Make sure OPENAI_API_KEY is set in .env
    ) else (
        echo     ✓ Sample explanations generated
        echo     ✓ Saved: data\evaluation\explanation_*.json
    )
) else (
    echo     ⊘ Skipping LLM explanation generation
)
echo.

REM ========================================================================
REM PHASE 5: SYSTEM EVALUATION
REM ========================================================================

set /p RUN_EVAL="Do you want to run system evaluation? (requires OpenAI API) (y/n): "
if /i "%RUN_EVAL%"=="y" (
    echo [7/9] Running system evaluation (faithfulness, grounding, retrieval)...
    set /p NUM_SAMPLES="How many samples to evaluate? (default: 10): "
    if "%NUM_SAMPLES%"=="" set NUM_SAMPLES=10

    python src\evaluation\evaluation_system.py --all --n-samples %NUM_SAMPLES%
    if errorlevel 1 (
        echo WARNING: System evaluation failed - check OpenAI API key
    ) else (
        echo     ✓ System evaluation complete
        echo     ✓ Results: data\evaluation\eval_system_all.json
    )
) else (
    echo [7/9] ⊘ Skipping system evaluation
)
echo.

REM ========================================================================
REM PHASE 6: HUMAN-CENTRIC EVALUATION
REM ========================================================================

set /p RUN_HUMAN="Do you want to run human-centric evaluation? (requires OpenAI API) (y/n): "
if /i "%RUN_HUMAN%"=="y" (
    echo [8/9] Running human-centric evaluation (alignment, traceability)...
    if "%NUM_SAMPLES%"=="" set NUM_SAMPLES=10

    python src\evaluation\evaluation_human.py --n-samples %NUM_SAMPLES%
    if errorlevel 1 (
        echo WARNING: Human evaluation failed
    ) else (
        echo     ✓ Human evaluation complete
        echo     ✓ Excel workbook: data\evaluation\hoge_evaluation_workbook.xlsx
        echo     ✓ Results: data\evaluation\eval_human_results.json
    )
) else (
    echo [8/9] ⊘ Skipping human-centric evaluation
)
echo.

REM ========================================================================
REM PHASE 7: VISUALIZATIONS (Optional)
REM ========================================================================

set /p RUN_VIZ="Do you want to generate KG visualizations? (y/n): "
if /i "%RUN_VIZ%"=="y" (
    echo [9/9] Generating knowledge graph visualizations...
    python src\knowledge_graph\generate_kg_visualizations.py
    if not errorlevel 1 (
        echo     ✓ Visualizations saved to figures\
    )
) else (
    echo [9/9] ⊘ Skipping visualizations
)
echo.

REM ========================================================================
REM SUMMARY
REM ========================================================================

echo.
echo ========================================================================
echo PIPELINE EXECUTION COMPLETE!
echo ========================================================================
echo.
echo Generated Files:
echo ----------------
echo Models:
echo   • models\loan_xgb_monotonic.joblib
echo.
echo Data:
echo   • data\processed\shap_long.csv
echo   • data\processed\shap_wide.csv
echo   • data\processed\scored_applications_*.csv
echo.
if /i "%RUN_LLM%"=="y" (
    echo Explanations:
    echo   • data\evaluation\explanation_*.json
    echo.
)
if /i "%RUN_EVAL%"=="y" (
    echo Evaluation Results:
    echo   • data\evaluation\eval_system_all.json
    echo   • data\evaluation\eval_faithfulness.json
    echo   • data\evaluation\eval_hallucination.json
    echo   • data\evaluation\eval_retrieval.json
    echo.
)
if /i "%RUN_HUMAN%"=="y" (
    echo Human Evaluation:
    echo   • data\evaluation\hoge_evaluation_workbook.xlsx
    echo   • data\evaluation\eval_human_results.json
    echo.
)
echo.
echo Next Steps:
echo -----------
echo 1. Review evaluation results in data\evaluation\
if /i "%RUN_HUMAN%"=="y" (
    echo 2. Open Excel workbook for detailed analysis
    start "" "data\evaluation\hoge_evaluation_workbook.xlsx" 2>nul
)
echo 3. Launch web interface: streamlit run app.py
echo 4. Explore KG in Neo4j Browser: http://localhost:7474
echo.
echo ========================================================================
pause
