"""
HOGE Framework - Interactive Web Interface
============================================
Streamlit app for generating and exploring loan application explanations.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import sys
import datetime
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import services (abstraction layer)
from backend.src.api.services import ExplanationService, ApplicationService
from backend.src.api.models import ExplanationRequest
from backend.src.api.adapters.config import get_neo4j_config as get_backend_neo4j_config

# Import backend modules for features not yet in services
from backend.src.explainability.concept_mapper import ConceptMapper
from backend.src.explainability.explanation_cache import ExplanationCache
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Debug: verify .env is loaded (only for local dev)
if not os.getenv("OPENAI_API_KEY"):
    # Try alternative loading methods
    load_dotenv(override=True)  # Try current directory
    if not os.getenv("OPENAI_API_KEY"):
        # Try explicit path
        import sys
        if hasattr(sys, '_MEIPASS'):  # Running as PyInstaller bundle
            pass
        else:
            # Development mode - try parent directory
            load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)
            if not os.getenv("OPENAI_API_KEY"):
                load_dotenv(dotenv_path=Path.cwd().parent / ".env", override=True)

# Initialize services
explanation_service = ExplanationService()
application_service = ApplicationService()

# Initialize explanation cache
explanation_cache = ExplanationCache()

# Page config
st.set_page_config(
    page_title="HOGE - Explainable Loan Decisions",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to get Neo4j config from Streamlit secrets or environment
def get_neo4j_config():
    """Get Neo4j configuration from Streamlit secrets (production) or environment (local)"""
    try:
        # Try to read from Streamlit secrets (production)
        if hasattr(st, 'secrets') and 'neo4j' in st.secrets:
            return {
                'uri': st.secrets["neo4j"]["uri"],
                'user': st.secrets["neo4j"]["user"],
                'password': st.secrets["neo4j"]["password"],
                'database': st.secrets["neo4j"].get("database", "neo4j")
            }
    except Exception:
        pass

    # Fall back to backend config adapter (supports USE_REMOTE_NEO4J flag)
    # This properly handles local vs remote Neo4j based on .env settings
    return get_backend_neo4j_config()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive-driver {
        color: #2ca02c;
        font-weight: bold;
    }
    .negative-driver {
        color: #d62728;
        font-weight: bold;
    }
    .policy-violation {
        color: #ff7f0e;
        font-weight: bold;
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

# Define audience options globally so they can be used across pages
audience_options = {
    "Technical": "technical",
    "Non-Technical": "non_technical",
    "Executive": "executive",
    "Business": "business"
}


# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50.png?text=HOGE", width=150)
    st.markdown("### HOGE Framework")
    st.markdown("Human-Centric Ontology-Grounded Explanations")

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["Home", "Explain Application", "Batch Analysis", "Evaluation Dashboard", "Settings"]
    )

    st.markdown("---")
    st.markdown("#### System Status")

    # Check Neo4j connection
    try:
        config = get_neo4j_config()
        driver = GraphDatabase.driver(config['uri'], auth=(config['user'], config['password']))
        driver.verify_connectivity()
        st.success("Neo4j Connected")
        driver.close()
    except Exception as e:
        st.error(f"Neo4j Disconnected: {str(e)}")

    st.markdown("---")

    # OpenAI Configuration in sidebar (available on all pages)
    st.markdown("#### 🤖 OpenAI API Key")

    # Pre-fill with key for local development (try secrets first, then .env)
    default_key = ""
    default_model = "gpt-4o"

    # Try Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            default_key = st.secrets["OPENAI_API_KEY"]
            default_model = st.secrets.get("OPENAI_MODEL", "gpt-4o")
    except:
        pass

    # Fall back to environment variables
    if not default_key:
        default_key = os.getenv("OPENAI_API_KEY", "")
        default_model = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Show key input
    openai_key = st.text_input("API Key", type="password", value=default_key,
                                help="Your OpenAI API key", label_visibility="collapsed")

    model_options = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
    default_index = model_options.index(default_model) if default_model in model_options else 0
    openai_model = st.selectbox("Model", model_options, index=default_index, label_visibility="collapsed")

    # Store in session state
    st.session_state['user_openai_key'] = openai_key
    st.session_state['user_openai_model'] = openai_model

    # Show status
    if openai_key:
        st.success("✓ API Key Loaded")
    else:
        st.info("ℹ️ Enter API key above")


# Main content
if page == "Home":
    st.markdown('<div class="main-header">HOGE Framework</div>', unsafe_allow_html=True)
    st.markdown("### Human-Centric Ontology-Grounded Explanation System")

    st.markdown("""
    This system provides transparent, grounded explanations for loan approval decisions by combining:

    - **🤖 ML Model**: XGBoost with monotonic constraints
    - **📊 SHAP Analysis**: Feature attribution for individual predictions
    - **🕸️ Knowledge Graph**: Semantic representation in Neo4j
    - **💬 LLM Narratives**: GPT-4o generates human-readable explanations
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Model Performance")
        st.metric("ROC-AUC", "0.839")
        st.metric("Accuracy", "80%")

    with col2:
        st.markdown("#### Explanation Quality")
        st.metric("Faithfulness", "0.760")
        st.metric("Hallucination Rate", "20.7%")

    with col3:
        st.markdown("#### Knowledge Graph")
        st.metric("Total Nodes", "23,000+")
        st.metric("Applications", "500")

    st.markdown("---")
    st.markdown("### 📊 System Provenance")
    st.caption("Global metadata for the entire HOGE pipeline (shared across all explanations)")

    # Load system-level provenance files
    def load_provenance_file(filepath):
        """Load provenance JSON file if it exists"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
        return {}

    model_prov = load_provenance_file("models/model_provenance.json").get('model_provenance', {})
    kg_prov = load_provenance_file("data/provenance/kg_provenance.json").get('kg_provenance', {})
    eval_prov = load_provenance_file("data/provenance/eval_provenance.json").get('evaluation_provenance', {})

    # Create tabs for system provenance
    sys_tabs = st.tabs(["📊 Model", "🕸️ Knowledge Graph", "✅ Evaluation", "🖥️ System"])

    with sys_tabs[0]:
        st.markdown("**Model Training Provenance**")
        if model_prov:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", model_prov.get('model_type', 'N/A'))
                st.metric("Training Samples", model_prov.get('training_samples', 'N/A'))
            with col2:
                perf = model_prov.get('performance_metrics', {})
                st.metric("Accuracy", f"{perf.get('accuracy', 0)*100:.1f}%")
                st.metric("ROC-AUC", f"{perf.get('roc_auc', 0):.3f}")
            with col3:
                st.metric("F1 Score", f"{perf.get('f1_score', 0):.3f}")
                st.metric("Features", model_prov.get('feature_count', 'N/A'))

            with st.expander("🔍 View Full Details"):
                st.json(model_prov)
        else:
            st.info("Run `python src/models/train_model.py` to generate model provenance")

    with sys_tabs[1]:
        st.markdown("**Knowledge Graph Provenance**")
        if kg_prov:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Nodes", kg_prov.get('total_nodes', 'N/A'))
            with col2:
                st.metric("Total Relationships", kg_prov.get('total_relationships', 'N/A'))
            with col3:
                st.metric("Neo4j Version", kg_prov.get('neo4j_version', 'N/A'))

            with st.expander("🔍 View Full Details"):
                st.json(kg_prov)
        else:
            st.info("Run `python src/knowledge_graph/neo_loader.py` to generate KG provenance")

    with sys_tabs[2]:
        st.markdown("**Evaluation Provenance**")
        if eval_prov:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Evaluations Run", len(eval_prov.get('evaluations_run', [])))
            with col2:
                st.metric("Sample Size", eval_prov.get('n_samples', 'N/A'))

            with st.expander("🔍 View Full Details"):
                st.json(eval_prov)
        else:
            st.info("Run `python src/evaluation/evaluation_system.py` to generate evaluation provenance")

    with sys_tabs[3]:
        st.markdown("**System Environment**")
        import platform
        import sys as system

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Python Version", f"{system.version_info.major}.{system.version_info.minor}.{system.version_info.micro}")
            st.metric("Platform", platform.system())
        with col2:
            st.metric("HOGE Version", "1.0.0")
            try:
                import subprocess
                git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
                st.metric("Git Commit", git_commit)
            except:
                st.metric("Git Commit", "Unknown")

    st.markdown("---")
    st.markdown("### Quick Start")
    st.markdown("""
    1. Select **Explain Application** from the sidebar
    2. Enter a Loan ID (e.g., LP001006)
    3. Click **Generate Explanation**
    4. Explore SHAP values, policy violations, and counterfactual scenarios
    """)


elif page == "Explain Application":
    st.markdown('<div class="main-header">Explain Loan Application</div>', unsafe_allow_html=True)

    # Fetch available application IDs from Neo4j
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_available_application_ids():
        """Fetch all available application IDs from Neo4j"""
        try:
            config = get_neo4j_config()
            NEO4J_URI = config['uri']
            NEO4J_USER = config['user']
            NEO4J_PASSWORD = config['password']
            NEO4J_DATABASE = config['database']

            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

            with driver.session(database=NEO4J_DATABASE) as session:
                result = session.run("""
                    MATCH (app:LoanApplication)
                    OPTIONAL MATCH (app)-[:HAS_SHAP_EXPLANATION]->(exp:ModelExplanation)
                    RETURN app.application_id AS application_id,
                           app.status AS decision,
                           exp.probability AS probability
                    ORDER BY app.application_id
                """)

                apps = []
                for record in result:
                    app_id = record["application_id"]
                    decision = record.get("decision", "Unknown")
                    prob = record.get("probability")
                    prob_str = f"{prob*100:.1f}%" if prob is not None else "N/A"

                    apps.append({
                        "id": app_id,
                        "label": f"{app_id} - {decision} ({prob_str})",
                        "decision": decision,
                        "probability": prob_str
                    })

            driver.close()
            return apps
        except Exception as e:
            st.error(f"Error fetching applications from Neo4j: {str(e)}")
            return []

    # Get available IDs
    available_apps = get_available_application_ids()

    if not available_apps:
        st.warning("No applications found in Neo4j. Please ensure the database is populated.")
        st.info("Run: `python src/knowledge_graph/neo_loader.py` to load data into Neo4j")

    # Two-column layout: Audience Selector | Application Selector
    col_left, col_right = st.columns(2)

    # LEFT COLUMN: Audience selector with help guide
    with col_left:
        st.markdown("### Select Target Audience")

        selected_audience_label = st.selectbox(
            "Who is this explanation for?",
            options=list(audience_options.keys()),
            index=1,  # Default to Non-Technical
            help="Choose the target audience to customize the explanation style"
        )
        audience = audience_options[selected_audience_label]

        # Store in session state for tab rendering
        st.session_state['selected_audience'] = audience

        with st.expander("Help"):
            st.markdown("""
            **Technical**
            For data scientists and ML engineers who want detailed SHAP values and technical metrics.

            **Non-Technical**
            For loan applicants and customer service reps who need simple, jargon-free explanations.

            **Executive**
            For C-suite and leadership who need concise strategic insights and business impact.

            **Business**
            For analysts and loan officers who need actionable recommendations and business metrics.
            """)

        # Concept toggle for executive/business audiences
        use_concepts = False
        if audience in ["executive", "business"]:
            st.markdown("##### Options")
            use_concepts = st.checkbox(
                "Use Business Concepts",
                value=True,
                help="Group features into business concepts"
            )

    # RIGHT COLUMN: Application ID selection
    with col_right:
        st.markdown("### Select Application")

        if available_apps:
            # Dropdown selection
            app_options = [app["label"] for app in available_apps]
            app_ids_only = [app["id"] for app in available_apps]

            # Find default index (LP001006 if exists, otherwise first)
            default_idx = 0
            if "LP001006" in app_ids_only:
                default_idx = app_ids_only.index("LP001006")

            selected_label = st.selectbox(
                "Select Loan Application ID:",
                options=app_options,
                index=default_idx,
                help="Choose from available applications in Neo4j"
            )

            # Extract just the ID from the label
            loan_id = selected_label.split(" - ")[0] if selected_label else ""

            # Optional custom ID entry
            with st.expander("Or enter custom Application ID"):
                custom_id = st.text_input(
                    "Custom Application ID:",
                    value="",
                    placeholder="LP001234",
                    help="Enter an application ID not in the dropdown"
                )
                if custom_id:
                    loan_id = custom_id

            generate_btn = st.button("Generate Explanation", type="primary", use_container_width=True)
        else:
            # Fallback to text input if Neo4j unavailable
            loan_id = st.text_input(
                "Enter Loan Application ID:",
                value="LP001006",
                help="Example: LP001006, LP001002, LP001003"
            )
            generate_btn = st.button("Generate Explanation", type="primary", use_container_width=True)

    st.markdown("---")

    # Cache Management Section
    with st.sidebar:
        st.markdown("### 💾 Explanation Cache")

        # Show file-based cache stats (for deployment)
        cache_stats = explanation_cache.stats()

        if cache_stats['total_cached'] > 0:
            st.metric("Pre-cached Explanations", cache_stats['total_cached'])
            st.caption(f"For {cache_stats['application_ids']} app IDs")

            with st.expander("📋 View Cached Apps"):
                cached_ids = explanation_cache.get_all_cached_ids()
                for app_id in cached_ids:
                    st.markdown(f"- {app_id}")
        else:
            st.info("No cached explanations yet")
            st.caption("Generate explanations to build cache")

    st.markdown("---")

    # Initialize session state for storing explanation
    if 'current_explanation' not in st.session_state:
        st.session_state.current_explanation = None
    if 'current_context' not in st.session_state:
        st.session_state.current_context = None
    if 'current_loan_id' not in st.session_state:
        st.session_state.current_loan_id = None

    # Initialize LLM response cache (persistent across reruns)
    if 'llm_cache' not in st.session_state:
        st.session_state.llm_cache = {}

    # Also load cache from disk if available
    CACHE_DIR = "data/cache"
    CACHE_FILE = os.path.join(CACHE_DIR, "llm_responses.json")

    def load_cache_from_disk():
        """Load cached LLM responses from disk"""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
        return {}

    def save_cache_to_disk(cache):
        """Save LLM response cache to disk"""
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving cache: {e}")

    # Load disk cache into session state if empty
    if not st.session_state.llm_cache:
        st.session_state.llm_cache = load_cache_from_disk()

    if generate_btn and loan_id:
        # Create cache key based on loan_id, audience, and use_concepts
        cache_key = f"{loan_id}_{audience}_{use_concepts}"

        # Check if we have a cached response
        if cache_key in st.session_state.llm_cache:
            st.info(f"✅ Using cached LLM response for {loan_id} (audience: {audience}, concepts: {use_concepts})")

            # Retrieve from cache
            cached_data = st.session_state.llm_cache[cache_key]
            explanation = cached_data['explanation']
            context = cached_data['context']

            # Store in current session state
            st.session_state.current_explanation = explanation
            st.session_state.current_context = context
            st.session_state.current_loan_id = loan_id
        else:
            # Check if explanation is cached FIRST (before checking API key)
            cached_explanation = explanation_cache.get_explanation(
                loan_id,
                audience=audience,
                use_concepts=use_concepts
            )

            if cached_explanation:
                st.info("📦 Using cached explanation (pre-generated locally)")
                explanation = cached_explanation
            else:
                # Not cached - need to generate with LLM
                # Get user's API key from session state
                user_api_key = st.session_state.get('user_openai_key')
                user_model = st.session_state.get('user_openai_model', 'gpt-4o')

                # Check if user provided API key
                if not user_api_key:
                    st.error("⚠️ No cached explanation found. Please provide your OpenAI API key in the sidebar to generate a new explanation.")
                    st.stop()

                st.info("🤖 Generating new explanation using your OpenAI API key...")

                try:
                    # Use ExplanationService (abstraction layer)
                    request = ExplanationRequest(
                        application_id=loan_id,
                        audience=audience,
                        use_concepts=use_concepts
                    )

                    # Generate explanation via service with user's API key
                    response = explanation_service.generate_explanation(
                        request,
                        api_key=user_api_key,
                        model=user_model
                    )

                    # Convert response DTO back to dict format for compatibility with existing UI code
                    explanation = response.to_dict()

                    # Auto-save to cache when running locally (for deployment later)
                    try:
                        explanation_cache.set_explanation(
                            loan_id,
                            audience,
                            use_concepts,
                            explanation
                        )
                        st.success("💾 Saved to local cache for future deployment")
                    except Exception as e:
                        st.warning(f"Could not save to cache: {e}")

                except Exception as e:
                    st.error(f"❌ Error generating explanation: {str(e)}")
                    import traceback
                    with st.expander("🐛 Error Details"):
                        st.code(traceback.format_exc())
                    st.stop()

            # Try to get raw context for visualizations (optional for cached explanations)
            try:
                from backend.src.explainability.llm_explainer import get_application_explanation_data
                context = get_application_explanation_data(loan_id)
            except Exception as e:
                # If Neo4j is unavailable, use minimal context from cached explanation
                st.warning(f"⚠️ Could not connect to Neo4j for additional context. Using cached data only.")
                context = {
                    'application_id': loan_id,
                    'model_prediction': explanation.get('decision', 'Unknown'),
                    'prediction_probability': explanation.get('probability', 0),
                    'features': {},
                    'shap_values': {}
                }

            # Store in session state
            st.session_state.current_explanation = explanation
            st.session_state.current_context = context
            st.session_state.current_loan_id = loan_id

            # Cache the response in session state too
            st.session_state.llm_cache[cache_key] = {
                'explanation': explanation,
                'context': context,
                'loan_id': loan_id,
                'audience': audience,
                'use_concepts': use_concepts,
                'cached_at': datetime.datetime.now().isoformat()
            }

            # Save cache to disk
            save_cache_to_disk(st.session_state.llm_cache)

    # Display explanation if available (either freshly generated or from session state)
    if st.session_state.current_explanation is not None:
        explanation = st.session_state.current_explanation
        context = st.session_state.current_context
        loan_id = st.session_state.current_loan_id

        # Display results
        st.success(f"Explanation generated successfully for **{selected_audience_label}**")

        # Show audience badge with concept mode indicator
        concept_badge = ""
        if explanation.get("used_concepts"):
            concept_badge = " | <strong>Concept Mode</strong>: Using business concepts instead of raw features"

        st.markdown(f"<div style='background-color: #e3f2fd; padding: 0.5rem; border-radius: 0.3rem; margin-bottom: 1rem;'>"
                   f"<strong>Audience:</strong> {selected_audience_label}{concept_badge}</div>",
                   unsafe_allow_html=True)

        # Create organized tabs for better readability
        st.markdown("---")

        # Determine tab list based on AUDIENCE
        # Different audiences see different levels of detail
        audience_key = st.session_state.get("selected_audience", "technical")

        if audience_key == "non_technical":
            # Non-technical: Simple, minimal tabs focused on understanding and action
            tab_list = [
                "📋 Simple Explanation",
                "🎯 What Matters Most",
                "💡 What You Can Do"
            ]
        elif audience_key == "executive":
            # Executive: High-level strategic view
            tab_list = [
                "📊 Executive Dashboard",
                "💼 Business Impact",
                "⚠️ Risk Assessment",
                "🔍 Deep Dive (Optional)"
            ]
        elif audience_key == "business":
            # Business: Balanced view with actionable insights
            tab_list = [
                "📋 Overview",
                "🎯 Key Factors",
                "💼 Business View",
                "🔮 What-If Scenarios",
                "🔬 Technical Details"
            ]
        else:  # technical
            # Technical: Full detailed view with all tabs
            tab_list = [
                "📋 Overview",
                "🎯 Key Factors",
                "🔮 What-If Scenarios",
                "🔬 Technical Analysis",
                "🌲 Mechanistic Deep Dive",
                "🕸️ Knowledge Graph",
                "📊 Evidence & Provenance"
            ]

        tabs = st.tabs(tab_list)

        # ============================================================
        # AUDIENCE-SPECIFIC TAB RENDERING
        # ============================================================

        if audience_key == "non_technical":
            # NON-TECHNICAL VIEW: Simple, minimal jargon
            # Tab 1: Simple Explanation
            with tabs[0]:
                st.markdown('<div class="sub-header">📋 Your Loan Decision</div>', unsafe_allow_html=True)

                decision = context['model_prediction']
                color = "green" if decision == "Approved" else "red"
                st.markdown(f"### Decision: <span style='color:{color};'>{decision}</span>", unsafe_allow_html=True)

                if decision == "Approved":
                    st.success("✅ Good news! Your loan application has been approved.")
                else:
                    st.error("❌ Unfortunately, your loan application was not approved at this time.")

                st.markdown("---")
                st.markdown("### Why This Decision Was Made")
                st.markdown(explanation['narrative'])

                st.markdown("---")
                st.markdown("### What This Means for You")
                st.info(explanation['recommendation'])

            # Tab 2: What Matters Most
            with tabs[1]:
                st.markdown('<div class="sub-header">🎯 What Affected Your Application</div>', unsafe_allow_html=True)

                # Show in simple language
                if explanation['positive_drivers']:
                    st.markdown("### ✅ Things That Helped")
                    for driver in explanation['positive_drivers']:
                        st.markdown(f"- {driver}")

                st.markdown("")

                if explanation['negative_drivers']:
                    st.markdown("### ⚠️ Things That Didn't Help")
                    for driver in explanation['negative_drivers']:
                        st.markdown(f"- {driver}")

            # Tab 3: What You Can Do
            with tabs[2]:
                st.markdown('<div class="sub-header">💡 Next Steps</div>', unsafe_allow_html=True)

                if decision == "Approved":
                    st.success("""
                    **Congratulations!** Your application was approved. Here's what happens next:

                    1. You'll receive official approval documents
                    2. Review the loan terms carefully
                    3. Contact us if you have any questions
                    """)
                else:
                    st.info(f"""
                    **Don't worry** - you have options:

                    {explanation.get('counterfactual_what_if', 'Contact us to discuss how you might improve your application.')}
                    """)

                    # Show simple improvement suggestions
                    if context.get('counterfactual_scenarios'):
                        st.markdown("### 🔧 What Could Help")
                        for i, cf in enumerate(context['counterfactual_scenarios'][:3], 1):
                            if cf.get('flipped_prediction'):
                                st.markdown(f"**{i}. {cf.get('change_description', 'Make a change')}**")
                                st.markdown(f"   - This could change the decision to **Approved**")

        elif audience_key == "executive":
            # EXECUTIVE VIEW: Strategic, high-level dashboard
            # Tab 1: Executive Dashboard
            with tabs[0]:
                st.markdown('<div class="sub-header">📊 Executive Summary</div>', unsafe_allow_html=True)

                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    decision = context['model_prediction']
                    st.metric("Decision", decision)

                with col2:
                    prob = context['approval_probability']
                    st.metric("Confidence", f"{prob*100:.0f}%")

                with col3:
                    num_violations = len(context['violated_rules'])
                    st.metric("Policy Issues", num_violations)

                with col4:
                    risk_level = "Low" if prob > 0.7 else "Medium" if prob > 0.4 else "High"
                    st.metric("Risk Level", risk_level)

                st.markdown("---")
                st.markdown("### Strategic Assessment")
                st.info(explanation['summary'])

            # Tab 2: Business Impact
            with tabs[1]:
                st.markdown('<div class="sub-header">💼 Business Impact Analysis</div>', unsafe_allow_html=True)

                # Show concept view if available
                if explanation.get("business_concepts"):
                    concepts = explanation["business_concepts"]

                    import plotly.graph_objects as go
                    concept_names = [c["concept_name"] for c in concepts[:6]]
                    concept_impacts = [c["aggregated_shap_value"] for c in concepts[:6]]

                    colors = ['#2ca02c' if x > 0 else '#d62728' for x in concept_impacts]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=concept_impacts,
                        y=concept_names,
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"{x:+.2f}" for x in concept_impacts],
                        textposition='auto'
                    ))

                    fig.update_layout(
                        title="Key Business Drivers",
                        xaxis_title="Impact",
                        height=400,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Business interpretation
                    for c in concepts[:3]:
                        st.markdown(f"**{c['concept_name']}**: {c['business_interpretation']}")
                else:
                    st.markdown(explanation['narrative'])

            # Tab 3: Risk Assessment
            with tabs[2]:
                st.markdown('<div class="sub-header">⚠️ Risk Factors</div>', unsafe_allow_html=True)

                if explanation['policy_violations']:
                    st.warning("**Policy Compliance Issues Detected:**")
                    for violation in explanation['policy_violations']:
                        st.markdown(f"- {violation}")
                else:
                    st.success("✅ No policy violations detected")

                st.markdown("---")
                st.markdown("### Risk Mitigation")
                st.markdown(explanation.get('counterfactual_what_if', 'Risk profile is acceptable'))

            # Tab 4: Deep Dive (optional technical details)
            with tabs[3]:
                st.markdown('<div class="sub-header">🔍 Detailed Analysis (Optional)</div>', unsafe_allow_html=True)
                st.caption("Technical details for deeper investigation if needed")

                with st.expander("📊 Feature Importance"):
                    shap_df = pd.DataFrame(context['shap_details'][:10])
                    st.dataframe(shap_df[['feature', 'value', 'shap', 'direction']], use_container_width=True)

                with st.expander("🔮 What-If Scenarios"):
                    st.markdown(explanation.get('counterfactual_what_if', 'No scenarios available'))

        else:
            # TECHNICAL/BUSINESS VIEW: Full detailed view (original behavior)
            # TAB 1: OVERVIEW
            with tabs[0]:
                st.markdown('<div class="sub-header">📋 Decision Summary</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)

                with col1:
                    decision = context['model_prediction']
                    color = "green" if decision == "Approved" else "red"
                    st.markdown(f"**Decision:** <span style='color:{color}; font-size:1.5rem;'>{decision}</span>",
                               unsafe_allow_html=True)

                with col2:
                    prob = context['approval_probability']
                    st.metric("Approval Probability", f"{prob*100:.1f}%")

                with col3:
                    num_violations = len(context['violated_rules'])
                    st.metric("Policy Violations", num_violations)

                # Summary
                st.info(f"**Summary:** {explanation['summary']}")

                # Narrative
                st.markdown("---")
                st.markdown('<div class="sub-header">📖 Explanation Narrative</div>', unsafe_allow_html=True)
                st.markdown(explanation['narrative'])

                # Recommendation
                st.markdown("---")
                st.markdown('<div class="sub-header">💡 Recommendation</div>', unsafe_allow_html=True)
                st.success(explanation['recommendation'])

            # TAB 2: KEY FACTORS
            with tabs[1]:
                st.markdown('<div class="sub-header">🎯 Key Factors Influencing Decision</div>', unsafe_allow_html=True)
                st.caption("These are the most important factors that influenced the loan decision.")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="sub-header">Positive Drivers</div>', unsafe_allow_html=True)
                    if explanation['positive_drivers']:
                        for driver in explanation['positive_drivers']:
                            st.markdown(f'<div class="positive-driver">+ {driver}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("*No positive drivers identified*")

                with col2:
                    st.markdown('<div class="sub-header">Negative Drivers</div>', unsafe_allow_html=True)
                    if explanation['negative_drivers']:
                        for driver in explanation['negative_drivers']:
                            st.markdown(f'<div class="negative-driver">- {driver}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("*No negative drivers identified*")

                # Policy violations
                if explanation['policy_violations']:
                    st.markdown("---")
                    st.markdown('<div class="sub-header">Policy Violations</div>', unsafe_allow_html=True)
                    st.caption("These policy rules were violated and may have affected the decision.")
                    for violation in explanation['policy_violations']:
                        st.markdown(f'<div class="policy-violation">! {violation}</div>', unsafe_allow_html=True)

            # TAB 3: WHAT-IF SCENARIOS
            with tabs[2]:
                st.markdown('<div class="sub-header">🔮 What-If Analysis</div>', unsafe_allow_html=True)
                st.caption("Explore how changes to key factors would affect the loan decision.")
                st.info(f"**{explanation.get('counterfactual_what_if', 'No counterfactual scenarios available.')}**")

                # Show detailed counterfactuals if available
                if context.get('counterfactual_scenarios'):
                    st.markdown("---")
                    st.markdown("**Detailed Counterfactual Scenarios:**")

                    for i, cf in enumerate(context['counterfactual_scenarios'][:5], 1):
                        impact_color = "green" if cf.get('probability_shift', 0) > 0 else "red"
                        flip_badge = "🔄 **Decision Flip!**" if cf.get('flipped_prediction') else ""

                        st.markdown(f"**Scenario {i}:** {cf.get('feature', 'N/A')}")
                        st.markdown(f"- Change from **{cf.get('current_value')}** to **{cf.get('changed_to')}** ({cf.get('change_description', 'N/A')})")
                        st.markdown(f"- Probability shift: <span style='color:{impact_color}'>{cf.get('probability_shift', 0)*100:+.1f}%</span> {flip_badge}", unsafe_allow_html=True)
                        st.markdown(f"- New probability: {cf.get('new_probability', 0)*100:.1f}%")
                        st.markdown("---")

            # TAB 4 onwards: Only for business/technical audiences
            if audience_key in ["business", "technical"]:
                # Business View tab (if concepts used)
                tab_idx = 3
                if explanation.get("used_concepts") and explanation.get("business_concepts"):
                    with tabs[tab_idx]:
                        st.markdown('<div class="sub-header">💼 Business Concept Map</div>', unsafe_allow_html=True)
                        st.caption("Technical features translated into business concepts for strategic decision-making.")

                        concepts = explanation["business_concepts"]

                        # Create concept visualization
                        import plotly.graph_objects as go

                        concept_names = [c["concept_name"] for c in concepts[:8]]
                        concept_impacts = [c["aggregated_shap_value"] for c in concepts[:8]]
                        concept_categories = [c["concept_category"] for c in concepts[:8]]

                        # Color by direction
                        colors = ['#2ca02c' if x > 0 else '#d62728' for x in concept_impacts]

                        fig_concepts = go.Figure()
                        fig_concepts.add_trace(go.Bar(
                            x=concept_impacts,
                            y=concept_names,
                            orientation='h',
                            marker=dict(color=colors),
                            text=[f"{x:+.3f}" for x in concept_impacts],
                            textposition='auto',
                            hovertemplate='<b>%{y}</b><br>Impact: %{x:+.3f}<br>Category: %{customdata}<extra></extra>',
                            customdata=concept_categories
                        ))

                        fig_concepts.update_layout(
                            title="Business Concept Impact Analysis",
                            xaxis_title="Aggregated Impact",
                            yaxis_title="Business Concept",
                            height=400,
                            showlegend=False,
                            yaxis={'categoryorder': 'total ascending'}
                        )

                        st.plotly_chart(fig_concepts, use_container_width=True)

                        # Concept details table with confidence indicators
                        concept_table = []
                        for c in concepts[:5]:
                            # Confidence emoji
                            confidence_emoji = {
                                "high": "🟢",
                                "medium": "🟡",
                                "low": "🔴"
                            }.get(c.get("confidence_level", "medium"), "⚪")

                            concept_table.append({
                                "Concept": c["concept_name"],
                                "Category": c["concept_category"].capitalize(),
                                "Impact": f"{c['aggregated_shap_value']:+.3f}",
                                "Direction": c["overall_direction"].capitalize(),
                                "Confidence": f"{confidence_emoji} {c.get('confidence_level', 'N/A').capitalize()}",
                                "Interpretation": c["business_interpretation"]
                            })

                        # Clean currency strings for Arrow compatibility
                        concept_table_df = pd.DataFrame(concept_table)
                        st.dataframe(concept_table_df, use_container_width=True)

                        # Feature to Concept mapping view
                        with st.expander("🔄 Feature ↔ Concept Mapping"):
                            st.markdown("**See how individual features map to business concepts:**")

                            for c in concepts[:5]:
                                st.markdown(f"**{c['concept_name']}** ({c['concept_category']})")
                                st.markdown(f"*Related features:* {', '.join(c['related_features'])}")
                                st.markdown(f"*Combined impact:* {c['aggregated_shap_value']:+.3f}")
                                st.markdown("---")

                    # Adjust tab index for subsequent tabs
                    tab_idx = 4
                else:
                    tab_idx = 3

                # TAB 5: TECHNICAL DEEP DIVE
                with tabs[tab_idx]:

                    # SHAP Waterfall Chart
                    st.markdown("#### 📊 SHAP Feature Contributions")

                    shap_df = pd.DataFrame(context['shap_details'][:10])

                    # Create interactive bar chart
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    # Separate positive and negative contributions
                    colors = ['#2ca02c' if x > 0 else '#d62728' for x in shap_df['shap']]

                    fig.add_trace(go.Bar(
                        x=shap_df['shap'],
                        y=shap_df['feature'],
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"{x:.4f}" for x in shap_df['shap']],
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<br>Value: %{customdata[0]}<extra></extra>',
                        customdata=shap_df[['value']].values
                    ))

                    fig.update_layout(
                        title="Top 10 Feature Contributions (SHAP Values)",
                        xaxis_title="SHAP Value (Impact on Prediction)",
                        yaxis_title="Feature",
                        height=500,
                        showlegend=False,
                        yaxis={'categoryorder': 'total ascending'}
                    )

                    st.plotly_chart(fig, width="stretch")

                    # Feature importance table
                    st.markdown("#### 📋 Detailed Feature Analysis")
                    # Clean any currency-formatted values
                    shap_display_df = shap_df[['feature', 'value', 'shap', 'direction', 'rank']].copy()
                    if shap_display_df['value'].dtype == 'object':
                        # Try to clean currency formatting
                        try:
                            shap_display_df['value'] = shap_display_df['value'].astype(str).str.replace('$', '').str.replace(',', '')
                            shap_display_df['value'] = pd.to_numeric(shap_display_df['value'], errors='ignore')
                        except:
                            pass  # Keep original if cleaning fails

                    st.dataframe(
                        shap_display_df.style.background_gradient(
                            subset=['shap'], cmap='RdYlGn', vmin=-1, vmax=1
                        ),
                        width="stretch"
                    )

                    # Feature importance pie chart
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### 🥧 Contribution Distribution")

                        # Pie chart for positive vs negative
                        pos_sum = sum(shap_df[shap_df['shap'] > 0]['shap'])
                        neg_sum = abs(sum(shap_df[shap_df['shap'] < 0]['shap']))

                        fig_pie = go.Figure(data=[go.Pie(
                            labels=['Positive Impact', 'Negative Impact'],
                            values=[pos_sum, neg_sum],
                            marker=dict(colors=['#2ca02c', '#d62728']),
                            hole=0.3
                        )])

                        fig_pie.update_layout(
                            title="Net SHAP Contribution",
                            height=300
                        )

                        st.plotly_chart(fig_pie, width="stretch")

                    with col2:
                        st.markdown("#### 🎯 Top Drivers")

                        # Top 5 by absolute value
                        top_5 = shap_df.nlargest(5, 'importance')

                        fig_top = go.Figure(data=[go.Bar(
                            x=top_5['importance'],
                            y=top_5['feature'],
                            orientation='h',
                            marker=dict(color='#1f77b4'),
                            text=[f"{x:.4f}" for x in top_5['importance']],
                            textposition='auto'
                        )])

                        fig_top.update_layout(
                            title="Top 5 by Importance",
                            xaxis_title="Absolute SHAP Value",
                            height=300,
                            showlegend=False
                        )

                        st.plotly_chart(fig_top, width="stretch")

                    # Features used in LLM explanation
                    st.markdown("#### 🤖 Features Referenced by LLM")

                    llm_features = explanation['features_used']
                    shap_features = shap_df['feature'].tolist()

                    # Show which features were used vs available
                    coverage_data = []
                    for feat in shap_features[:10]:
                        coverage_data.append({
                            'Feature': feat,
                            'In SHAP Top 10': '✓',
                            'Used by LLM': '✓' if feat in llm_features else '✗'
                        })

                    coverage_df = pd.DataFrame(coverage_data)
                    st.dataframe(coverage_df, width="stretch")

                    # Coverage metrics
                    coverage_rate = len(set(llm_features) & set(shap_features)) / len(shap_features) * 100
                    st.metric("LLM Coverage of Top 10 SHAP Features", f"{coverage_rate:.0f}%")

                    # Feature vs Concept Comparison (if concepts were used)
                    if explanation.get("used_concepts"):
                        st.markdown("#### 🔄 Feature-Level vs Concept-Level Comparison")

                        comp_col1, comp_col2 = st.columns(2)

                        with comp_col1:
                            st.markdown("**📊 Feature-Level View**")
                            st.markdown(f"*{len(context['shap_details'])} individual features*")

                            # Show top features
                            for feat in context['shap_details'][:5]:
                                direction_emoji = "✅" if feat.get('shap', 0) > 0 else "❌"
                                st.markdown(f"{direction_emoji} {feat['feature']}: {feat.get('shap', 0):+.3f}")

                        with comp_col2:
                            st.markdown("**🧩 Concept-Level View**")
                            concepts = explanation.get("business_concepts", [])
                            st.markdown(f"*{len(concepts)} business concepts*")

                            # Show top concepts
                            for concept in concepts[:5]:
                                direction_emoji = "✅" if concept['aggregated_shap_value'] > 0 else "❌"
                                st.markdown(f"{direction_emoji} {concept['concept_name']}: {concept['aggregated_shap_value']:+.3f}")

                        st.caption("💡 Concepts aggregate multiple features for simpler understanding")

            # TAB: MECHANISTIC DEEP DIVE (only for technical audience)
            if audience_key == "technical":
                with tabs[4]:  # Tab 5 for technical audience
                    st.markdown("**Deep mechanistic analysis of XGBoost decision-making**")
                    st.caption("Understand how the model makes decisions at the tree level, including decision paths, feature interactions, and ensemble patterns.")
                    st.markdown("---")

                    # Mechanistic Interpretability Analysis
                    st.markdown("#### 🧠 Mechanistic Interpretability Analysis")

                    if st.button("🔬 Run Deep Model Analysis", help="Analyze decision paths, feature interactions, and tree-level patterns"):
                        with st.spinner("Running mechanistic analysis..."):
                            try:
                                from backend.src.explainability.mechanistic_interpreter import MechanisticInterpreter
                                import pandas as pd
                                import joblib

                                # Load model and data
                                MODEL_PATH = "models/loan_xgb_monotonic.joblib"

                                # Try multiple data sources
                                import os
                                if os.path.exists("data/raw/df1_loan.csv"):
                                    df = pd.read_csv("data/raw/df1_loan.csv")
                                    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

                                    # Clean ALL currency and numeric columns thoroughly
                                    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Total_Income"]
                                    for col in numeric_cols:
                                        if col in df.columns:
                                            df[col] = (df[col].astype(str)
                                                      .str.replace("$", "", regex=False)
                                                      .str.replace(",", "", regex=False)
                                                      .str.strip())
                                            df[col] = pd.to_numeric(df[col], errors="coerce")

                                    # Add DTI
                                    df["DTI"] = df["LoanAmount"] / (df["ApplicantIncome"] + df["CoapplicantIncome"] + 1)
                                else:
                                    st.error("Raw data file not found: data/raw/df1_loan.csv")
                                    raise FileNotFoundError("data/raw/df1_loan.csv")

                                app_data = df[df['Loan_ID'] == loan_id]

                                if not app_data.empty:
                                    feature_cols = [col for col in df.columns if col not in ['Loan_ID', 'Loan_Status', 'Total_Income', 'DTI', 'Unnamed: 0']]
                                    # Add the engineered features
                                    feature_cols = feature_cols + ['Total_Income', 'DTI']
                                    X = app_data[feature_cols]

                                    # Initialize interpreter
                                    interpreter = MechanisticInterpreter(MODEL_PATH, feature_cols)

                                    # Extract decision paths
                                    paths = interpreter.extract_decision_path(loan_id, X, tree_limit=50)
                                    interactions = interpreter.detect_feature_interactions(min_co_occurrence=5)
                                    tree_insights = interpreter.analyze_tree_level_patterns()

                                    # Generate narrative
                                    narrative = interpreter.generate_narrative_explanation(
                                        loan_id, paths, interactions, tree_insights, audience=audience
                                    )

                                    st.markdown(narrative)

                                    # Show interactive visualizations
                                    st.markdown("**📊 Interactive Decision Path Visualization**")

                                    from backend.src.explainability.tree_visualizer import TreeVisualizer
                                    visualizer = TreeVisualizer()

                                    # Decision paths (interactive 6-panel dashboard)
                                    fig_paths = visualizer.visualize_decision_paths_interactive(paths, top_k=20)
                                    if fig_paths:
                                        st.plotly_chart(fig_paths, width="stretch")

                                    # Feature interaction network
                                    st.markdown("**🔗 Feature Interaction Network**")
                                    fig_interactions = visualizer.visualize_feature_interactions_network(interactions, top_k=15)
                                    if fig_interactions:
                                        st.plotly_chart(fig_interactions, width="stretch")

                                    # Tree ensemble summary
                                    st.markdown("**🌲 Tree Ensemble Summary**")
                                    fig_ensemble = visualizer.visualize_tree_ensemble_summary(tree_insights)
                                    if fig_ensemble:
                                        st.plotly_chart(fig_ensemble, width="stretch")

                                    st.success("Mechanistic analysis complete - all visualizations are interactive (hover, zoom, pan)")
                                else:
                                    st.error(f"Application {loan_id} not found in processed data")

                            except FileNotFoundError as e:
                                st.error(f"Required file not found: {e}")
                                st.info("Make sure model and data files exist at the specified paths")
                            except Exception as e:
                                st.error(f"Error during mechanistic analysis: {e}")
                                import traceback
                                st.code(traceback.format_exc())

            # TAB: KNOWLEDGE GRAPH (only for technical audience)
            if audience_key == "technical":
                with tabs[5]:  # Tab 6 for technical audience
                    st.markdown("**Knowledge Graph Network Visualization**")
                    st.caption("Ontology-grounded evidence showing the relationships between this loan, SHAP values, policy rules, and counterfactuals.")
                    st.markdown("---")

                    # Knowledge Graph Network Visualization using Graphviz
                    try:
                        from backend.src.knowledge_graph.kg_visualizer import KGVisualizer

                        kg_viz = KGVisualizer()

                        # Extract evidence from context
                        evidence = kg_viz.extract_evidence_from_context(context)

                        # Try Neo4j query first, fall back to context-based
                        try:
                            graph_viz = kg_viz.create_graphviz_from_neo4j(loan_id)
                            if graph_viz:
                                st.graphviz_chart(graph_viz)
                                st.success("✅ Network loaded from Neo4j knowledge graph")
                            else:
                                # Fallback to context-based network
                                graph_viz = kg_viz.create_graphviz_network(evidence)
                                st.graphviz_chart(graph_viz)
                                st.info("ℹ️ Network generated from explanation context (Neo4j not available)")
                        except Exception as e:
                            # Fallback to context-based network
                            graph_viz = kg_viz.create_graphviz_network(evidence)
                            st.graphviz_chart(graph_viz)
                            st.info("ℹ️ Network generated from explanation context")

                        st.markdown("""
                        **Legend:**
                        - 🔵 **Blue (Ellipse)**: Loan application node
                        - 🟢 **Green**: Positive SHAP features (↑ approval)
                        - 🔴 **Red**: Negative SHAP features or policy violations (↓ approval)
                        - 🟠 **Orange**: Counterfactual scenarios
                        - **Dashed lines**: Policy rule violations
                        - **Dotted lines**: Counterfactual relationships
                        """)

                        # Close connection
                        kg_viz.close()

                    except ImportError as e:
                        st.warning(f"⚠️ KG Visualizer not available: {e}")
                    except Exception as e:
                        st.error(f"❌ Error creating KG visualization: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            # TAB: EVIDENCE & PROVENANCE (only for technical audience)
            if audience_key == "technical":
                with tabs[6]:  # Last tab for technical audience
                    st.markdown("**Full Traceability & Reproducibility**")
                    st.markdown("- Complete provenance metadata for all pipeline components")
                    st.markdown("- Download options for explanation, context, and provenance")
                    st.markdown("---")

                    # Provenance
                    st.markdown("#### 🔍 Provenance & Metadata")

                    # Check if enhanced provenance is available
                    prov_compact = explanation.get('provenance_compact', {})
                    prov_full = explanation.get('provenance_full', {})

                    # Load saved provenance files (model, KG, eval)
                    def load_provenance_file(filepath):
                        """Load provenance JSON file if it exists"""
                        try:
                            if os.path.exists(filepath):
                                with open(filepath, 'r') as f:
                                    return json.load(f)
                        except Exception as e:
                            print(f"Error loading {filepath}: {e}")
                        return {}

                    # Load external provenance sources
                    model_prov_file = load_provenance_file("models/model_provenance.json")
                    kg_prov_file = load_provenance_file("data/provenance/kg_provenance.json")
                    eval_prov_file = load_provenance_file("data/provenance/eval_provenance.json")

                    # Merge external provenance into prov_full
                    if model_prov_file and 'model_provenance' in model_prov_file:
                            prov_full['model_provenance'] = model_prov_file['model_provenance']

                    if kg_prov_file and 'kg_provenance' in kg_prov_file:
                            prov_full['kg_provenance'] = kg_prov_file['kg_provenance']

                    if eval_prov_file and 'evaluation_provenance' in eval_prov_file:
                            prov_full['evaluation_provenance'] = eval_prov_file['evaluation_provenance']

                    if prov_compact:
                            # Show compact instance-level provenance only
                            st.markdown("**Instance-Level Metadata:**")
                            st.caption("LLM generation details for this specific explanation")

                            metric_cols = st.columns(4)
                            with metric_cols[0]:
                                st.metric("LLM Model", prov_compact.get('llm_model', 'N/A'))
                            with metric_cols[1]:
                                st.metric("Tokens", prov_compact.get('llm_tokens', 'N/A'))
                            with metric_cols[2]:
                                st.metric("Evidence Items", prov_compact.get('evidence_items', 'N/A'))
                            with metric_cols[3]:
                                st.metric("Audience", prov_compact.get('target_audience', 'N/A'))

                    else:
                            # Fallback to basic provenance if enhanced not available
                            st.markdown("**Instance-Level Metadata:**")
                            st.caption("Explanation generation details")

                            prov_col1, prov_col2 = st.columns(2)

                            with prov_col1:
                                st.write(f"- Model: {explanation.get('provenance', {}).get('model_name', 'N/A')}")
                                st.write(f"- XAI Method: {explanation.get('provenance', {}).get('xai_method', 'N/A')}")
                                st.write(f"- HOGE Version: {explanation.get('provenance', {}).get('hoge_version', 'N/A')}")

                            with prov_col2:
                                st.write(f"- Target Audience: {audience}")
                                st.write(f"- Used Concepts: {explanation.get('used_concepts', False)}")
                                st.write(f"- Evidence Items: {len(explanation.get('evidence_bundle', []))}")

                    # Download options
                    st.markdown("---")
                    st.markdown("### 📥 Download Options")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                            st.download_button(
                                "📥 Explanation (JSON)",
                                data=json.dumps(explanation, indent=2, default=str),
                                file_name=f"explanation_{loan_id}.json",
                                mime="application/json",
                                help="Download complete explanation with all provenance"
                            )

                    with col2:
                            st.download_button(
                                "📥 Context Data (JSON)",
                                data=json.dumps(context, indent=2, default=str),
                                file_name=f"context_{loan_id}.json",
                                mime="application/json",
                                help="Download raw context from knowledge graph"
                            )

                    with col3:
                            # Download full provenance if available
                            if prov_full:
                                st.download_button(
                                    "📊 Full Provenance (JSON)",
                                    data=json.dumps(prov_full, indent=2, default=str),
                                    file_name=f"provenance_{loan_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    help="Download complete provenance metadata for reproducibility"
                                )
                            else:
                                st.download_button(
                                    "📊 Basic Provenance (JSON)",
                                    data=json.dumps(explanation.get('provenance', {}), indent=2, default=str),
                                    file_name=f"provenance_{loan_id}.json",
                                    mime="application/json",
                                    help="Download basic provenance metadata"
                                )


elif page == "Batch Analysis":
    st.markdown('<div class="main-header">📊 Batch Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    Generate explanations for multiple applications and compare results.
    """)

    # Load available applications
    try:
        df = pd.read_csv("data/raw/df1_loan.csv")
        app_ids = df['Loan_ID'].tolist()[:20]  # First 20 for demo

        selected_apps = st.multiselect(
            "Select Applications to Analyze:",
            options=app_ids,
            default=app_ids[:5]
        )

        # Ask for audience for batch analysis
        batch_audience = st.selectbox(
            "Select audience for batch analysis:",
            options=list(audience_options.keys()),
            index=0
        )
        batch_audience_key = audience_options[batch_audience]

        if st.button("🚀 Analyze Selected Applications", type="primary"):
            if not selected_apps:
                st.warning("Please select at least one application")
            else:
                # Get user's API key from session state
                user_api_key = st.session_state.get('user_openai_key')
                user_model = st.session_state.get('user_openai_model', 'gpt-4o')

                # Check if API key is provided
                if not user_api_key:
                    st.error("⚠️ Please provide your OpenAI API key in the sidebar to generate explanations.")
                    st.stop()

                results = []
                progress_bar = st.progress(0)

                for i, app_id in enumerate(selected_apps):
                    try:
                        # Check cache first
                        cached_explanation = explanation_cache.get_explanation(
                            app_id,
                            audience=batch_audience_key,
                            use_concepts=False
                        )

                        if cached_explanation:
                            explanation = cached_explanation
                        else:
                            # Use ExplanationService for batch processing
                            request = ExplanationRequest(
                                application_id=app_id,
                                audience=batch_audience_key,
                                use_concepts=False
                            )
                            response = explanation_service.generate_explanation(
                                request,
                                api_key=user_api_key,
                                model=user_model
                            )
                            explanation = response.to_dict()

                            # Auto-save to cache
                            try:
                                explanation_cache.set_explanation(
                                    app_id,
                                    batch_audience_key,
                                    False,
                                    explanation
                                )
                            except Exception:
                                pass  # Silently fail in batch mode

                        # Get context for compatibility
                        from backend.src.explainability.llm_explainer import get_application_explanation_data
                        context = get_application_explanation_data(app_id)

                        results.append({
                            'Application ID': app_id,
                            'Decision': context['model_prediction'],
                            'Probability': f"{context['approval_probability']*100:.1f}%",
                            'Policy Violations': len(context['violated_rules']),
                            'Summary': explanation['summary']
                        })
                    except Exception as e:
                        st.warning(f"Error processing {app_id}: {str(e)}")

                    progress_bar.progress((i + 1) / len(selected_apps))

                # Display results
                if results:
                    st.markdown("### Analysis Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, width="stretch")

                    # Summary stats
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        approved = sum(1 for r in results if r['Decision'] == 'Approved')
                        st.metric("Approved", f"{approved}/{len(results)}")

                    with col2:
                        total_violations = sum(r['Policy Violations'] for r in results)
                        st.metric("Total Violations", total_violations)

                    with col3:
                        avg_prob = sum(float(r['Probability'].rstrip('%')) for r in results) / len(results)
                        st.metric("Avg Probability", f"{avg_prob:.1f}%")

    except FileNotFoundError:
        st.error("df1_loan.csv not found. Please ensure the data file is in the root directory.")


elif page == "Evaluation Dashboard":
    st.markdown('<div class="main-header">📈 System Evaluation Dashboard</div>', unsafe_allow_html=True)

    st.markdown("""
    This dashboard visualizes the quantitative and qualitative evaluation metrics for the HOGE framework.
    Metrics include faithfulness, hallucination rates, retrieval accuracy, and human-centric evaluations.
    """)

    # Create tabs for different evaluation categories
    eval_tabs = st.tabs([
        "📊 Overview",
        "🔬 Faithfulness",
        "🎯 Hallucination & Grounding",
        "🔍 Retrieval Accuracy",
        "👥 Human Evaluation"
    ])

    # Tab 1: Overview
    with eval_tabs[0]:
        st.markdown("### System Evaluation Summary")

        # Try to load the combined evaluation file
        try:
            with open("data/evaluation/eval_system_all.json", "r") as f:
                eval_all = json.load(f)

            st.success(f"Evaluation data loaded (timestamp: {eval_all.get('evaluation_timestamp', 'N/A')})")

            # Create metrics summary
            col1, col2, col3 = st.columns(3)

            # Calculate summary metrics
            if "hallucination" in eval_all.get("results", {}):
                hall_data = eval_all["results"]["hallucination"]
                avg_semantic_fidelity = sum(r.get("semantic_fidelity", 0) for r in hall_data if isinstance(r.get("semantic_fidelity"), (int, float))) / max(len(hall_data), 1)
                avg_hallucination = sum(r.get("hallucination_rate", 0) for r in hall_data if isinstance(r.get("hallucination_rate"), (int, float))) / max(len(hall_data), 1)

                with col1:
                    st.metric("Semantic Fidelity", f"{avg_semantic_fidelity:.2%}",
                             delta=f"{(avg_semantic_fidelity - 0.8):.1%}" if avg_semantic_fidelity > 0.8 else None)

                with col2:
                    st.metric("Hallucination Rate", f"{avg_hallucination:.2%}",
                             delta=f"{(0.1 - avg_hallucination):.1%}" if avg_hallucination < 0.1 else None,
                             delta_color="inverse")

            if "faithfulness" in eval_all.get("results", {}):
                faith_data = eval_all["results"]["faithfulness"]
                avg_faithfulness = sum(r.get("faithfulness_score", 0) for r in faith_data if isinstance(r.get("faithfulness_score"), (int, float))) / max(len(faith_data), 1)

                with col3:
                    st.metric("Faithfulness Score", f"{avg_faithfulness:.2f}/1.0")

            # Show dataset info
            st.markdown("---")
            st.markdown("### Evaluation Dataset")

            datasets = []
            if "hallucination" in eval_all.get("results", {}):
                datasets.append(("Hallucination Test", len(eval_all["results"]["hallucination"])))
            if "faithfulness" in eval_all.get("results", {}):
                datasets.append(("Faithfulness Test", len(eval_all["results"]["faithfulness"])))
            if "retrieval" in eval_all.get("results", {}):
                datasets.append(("Retrieval Test", len(eval_all["results"]["retrieval"])))

            df_datasets = pd.DataFrame(datasets, columns=["Evaluation Type", "Sample Size"])
            st.dataframe(df_datasets, use_container_width=True)

        except FileNotFoundError:
            st.warning("No evaluation data found. Run evaluations first:")
            st.code("python src/evaluation/evaluation_system.py --all --n-samples 10", language="bash")

    # Tab 2: Faithfulness
    with eval_tabs[1]:
        st.markdown("### 🔬 Explanation Faithfulness (Perturbation Tests)")
        st.markdown("""
        Tests whether explanations remain faithful when input features are perturbed.
        A high faithfulness score indicates explanations correctly reflect model behavior.
        """)

        try:
            with open("data/evaluation/eval_faithfulness.json", "r") as f:
                faith_eval = json.load(f)

            results = faith_eval.get("results", [])

            if results:
                # Create DataFrame
                df_faith = pd.DataFrame([
                    {
                        "Application ID": r["application_id"],
                        "Original Feature": r["original_top_feature"],
                        "Original Prob": f"{r['original_prob']:.3f}",
                        "New Prob": f"{r['new_prob']:.3f}",
                        "Probability Shift": f"{r['prob_shift']:.3f}",
                        "Faithfulness Score": r.get("faithfulness_score", "N/A"),
                        "Perturbation Reflected": r.get("perturbation_reflected", "N/A")
                    }
                    for r in results
                ])

                # Show summary stats
                scores = [r["faithfulness_score"] for r in results if isinstance(r.get("faithfulness_score"), (int, float))]
                if scores:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Score", f"{sum(scores)/len(scores):.3f}")
                    with col2:
                        st.metric("Pass Rate (≥0.5)", f"{sum(1 for s in scores if s >= 0.5)}/{len(scores)}")
                    with col3:
                        st.metric("Perfect Scores (1.0)", f"{sum(1 for s in scores if s >= 0.99)}/{len(scores)}")

                st.dataframe(df_faith, use_container_width=True)

                # Show detailed results
                with st.expander("📋 View Detailed Results"):
                    for r in results:
                        st.markdown(f"**{r['application_id']}**: {r.get('reasoning', 'N/A')}")
                        st.markdown(f"- Perturbation: {r['perturbation']}")
                        st.markdown("---")
            else:
                st.info("No faithfulness test results available.")

        except FileNotFoundError:
            st.warning("Run faithfulness evaluation first:")
            st.code("python src/evaluation/evaluation_system.py --faithfulness --n-samples 10", language="bash")

    # Tab 3: Hallucination & Grounding
    with eval_tabs[2]:
        st.markdown("### 🎯 Hallucination Rate & Grounding Precision")
        st.markdown("""
        Measures how well LLM explanations are grounded in the knowledge graph evidence.
        - **Semantic Fidelity**: Proportion of claims supported by evidence
        - **Evidence Coverage**: Proportion of evidence used in explanation
        - **Hallucination Rate**: Proportion of unsupported claims
        """)

        try:
            with open("data/evaluation/eval_hallucination.json", "r") as f:
                hall_eval = json.load(f)

            results = hall_eval.get("results", [])

            if results:
                # Create DataFrame
                df_hall = pd.DataFrame([
                    {
                        "Application ID": r["application_id"],
                        "Prediction": r["prediction"],
                        "Probability": f"{r['probability']:.3f}",
                        "Total Claims": r.get("total_claims", 0),
                        "Grounded": r.get("grounded_claims", 0),
                        "Hallucinated": r.get("hallucinated_claims", 0),
                        "Semantic Fidelity": f"{r.get('semantic_fidelity', 0):.2%}",
                        "Evidence Coverage": f"{r.get('evidence_coverage', 0):.2%}",
                        "Hallucination Rate": f"{r.get('hallucination_rate', 0):.2%}",
                    }
                    for r in results
                ])

                # Summary metrics
                sfs = [r["semantic_fidelity"] for r in results if isinstance(r.get("semantic_fidelity"), (int, float))]
                ecs = [r["evidence_coverage"] for r in results if isinstance(r.get("evidence_coverage"), (int, float))]
                hrs = [r["hallucination_rate"] for r in results if isinstance(r.get("hallucination_rate"), (int, float))]

                col1, col2, col3, col4 = st.columns(4)
                if sfs:
                    with col1:
                        st.metric("Avg Semantic Fidelity", f"{sum(sfs)/len(sfs):.2%}")
                if ecs:
                    with col2:
                        st.metric("Avg Evidence Coverage", f"{sum(ecs)/len(ecs):.2%}")
                if hrs:
                    with col3:
                        st.metric("Avg Hallucination Rate", f"{sum(hrs)/len(hrs):.2%}")
                    with col4:
                        st.metric("Zero Hallucinations", f"{sum(1 for h in hrs if h == 0)}/{len(hrs)}")

                st.dataframe(df_hall, use_container_width=True)

                # Show hallucinated details
                with st.expander("🚨 View Hallucinated Claims"):
                    for r in results:
                        if r.get("hallucinated_claims", 0) > 0:
                            st.markdown(f"**{r['application_id']}** ({r.get('hallucinated_claims', 0)} hallucinations):")
                            for detail in r.get("hallucinated_details", []):
                                st.markdown(f"- {detail}")
                            st.markdown("---")
            else:
                st.info("No hallucination test results available.")

        except FileNotFoundError:
            st.warning("Run hallucination evaluation first:")
            st.code("python src/evaluation/evaluation_system.py --hallucination --n-samples 10", language="bash")

    # Tab 4: Retrieval Accuracy
    with eval_tabs[3]:
        st.markdown("### 🔍 Graph Retrieval Accuracy (Precision@K)")
        st.markdown("""
        Evaluates how accurately the knowledge graph retrieves SHAP contributions.
        Compares KG retrieval against ground truth from shap_long.csv.
        """)

        try:
            with open("data/evaluation/eval_retrieval.json", "r") as f:
                retr_eval = json.load(f)

            results = retr_eval.get("results", [])
            valid_results = [r for r in results if r.get("kg_found")]

            if valid_results:
                # Create DataFrame
                df_retr = pd.DataFrame([
                    {
                        "Application ID": r["application_id"],
                        "Prediction Match": "✅" if r["prediction_match"] else "❌",
                        "Precision@3": f"{r['precision_at_3']:.3f}",
                        "Precision@5": f"{r['precision_at_5']:.3f}",
                        "Precision@10": f"{r['precision_at_10']:.3f}",
                        "Feature Recall": f"{r['feature_recall']:.3f}",
                        "GT Features": r["total_gt_features"],
                        "KG Features": r["total_kg_features"],
                    }
                    for r in valid_results
                ])

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_p3 = sum(r["precision_at_3"] for r in valid_results) / len(valid_results)
                    st.metric("Avg Precision@3", f"{avg_p3:.3f}")
                with col2:
                    avg_p5 = sum(r["precision_at_5"] for r in valid_results) / len(valid_results)
                    st.metric("Avg Precision@5", f"{avg_p5:.3f}")
                with col3:
                    avg_p10 = sum(r["precision_at_10"] for r in valid_results) / len(valid_results)
                    st.metric("Avg Precision@10", f"{avg_p10:.3f}")
                with col4:
                    pred_match_rate = sum(1 for r in valid_results if r["prediction_match"]) / len(valid_results)
                    st.metric("Prediction Match", f"{pred_match_rate:.1%}")

                st.dataframe(df_retr, use_container_width=True)

                # Visualization
                import plotly.express as px

                fig = px.bar(
                    x=["Precision@3", "Precision@5", "Precision@10"],
                    y=[avg_p3, avg_p5, avg_p10],
                    labels={"x": "Metric", "y": "Score"},
                    title="Average Retrieval Precision by K"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No retrieval test results available.")

        except FileNotFoundError:
            st.warning("Run retrieval evaluation first:")
            st.code("python src/evaluation/evaluation_system.py --retrieval --n-samples 10", language="bash")

    # Tab 5: Human Evaluation
    with eval_tabs[4]:
        st.markdown("### 👥 Human-Centric Evaluation")
        st.markdown("""
        Qualitative evaluation of explanation quality from a human perspective.
        Includes alignment, traceability, and audience appropriateness metrics.
        """)

        try:
            with open("data/evaluation/eval_human_results.json", "r") as f:
                human_eval = json.load(f)

            # Show key metrics in a clean format instead of raw JSON
            if "summary" in human_eval:
                summary = human_eval["summary"]

                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Samples Evaluated", summary.get("total_samples", "N/A"))
                with col2:
                    st.metric("Avg Alignment", f"{summary.get('avg_alignment_score', 0):.2f}")
                with col3:
                    st.metric("Avg Traceability", f"{summary.get('avg_traceability_score', 0):.2f}")

            # Show top results in a table (limit to 5 for better UX)
            if "results" in human_eval:
                st.markdown("#### Sample Evaluation Results (Top 5)")
                results = human_eval["results"][:5]  # Show only top 5

                results_table = []
                for r in results:
                    results_table.append({
                        "Application": r.get("application_id", "N/A"),
                        "Alignment": f"{r.get('alignment_score', 0):.2f}",
                        "Traceability": f"{r.get('traceability_score', 0):.2f}",
                        "Appropriateness": f"{r.get('audience_appropriateness', 0):.2f}"
                    })

                st.dataframe(results_table, use_container_width=True, hide_index=True)

                # Option to view full results
                with st.expander("🔍 View Full JSON Data"):
                    st.json(human_eval)

            st.markdown("---")
            st.info("📊 For detailed evaluation, check: `data/evaluation/hoge_evaluation_workbook.xlsx`")

        except FileNotFoundError:
            st.warning("Run human evaluation first:")
            st.code("python src/evaluation/evaluation_human.py --n-samples 10", language="bash")

            st.markdown("---")
            st.info("📊 Alternatively, check the Excel workbook: `data/evaluation/hoge_evaluation_workbook.xlsx`")


elif page == "Settings":
    st.markdown('<div class="main-header">System Settings</div>', unsafe_allow_html=True)

    st.markdown("### Configuration")

    # Neo4j settings
    st.markdown("#### 🕸️ Neo4j Connection")
    col1, col2 = st.columns(2)

    with col1:
        neo4j_uri = st.text_input("Neo4j URI", value=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"))
        neo4j_user = st.text_input("Neo4j User", value=os.getenv("NEO4J_USER", "neo4j"))

    with col2:
        neo4j_password = st.text_input("Neo4j Password", type="password", value="")

    if st.button("Test Neo4j Connection"):
        try:
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            driver.verify_connectivity()
            st.success("Connection successful!")
            driver.close()
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")

    st.markdown("---")

    # OpenAI settings
    st.markdown("#### 🤖 OpenAI Configuration")

    # Pre-fill with key for local development (try secrets first, then .env)
    default_key = ""
    default_model = "gpt-4o"

    # Try Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            default_key = st.secrets["OPENAI_API_KEY"]
            default_model = st.secrets.get("OPENAI_MODEL", "gpt-4o")
    except:
        pass

    # Fall back to environment variables
    if not default_key:
        default_key = os.getenv("OPENAI_API_KEY", "")
        default_model = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Debug info
    if default_key:
        st.markdown("*Using API key from .env file (local development)*")
        st.caption(f"✓ Key loaded (starts with: {default_key[:10]}...)")
    else:
        st.markdown("*Provide your own OpenAI API key to use this app*")
        # Show debug info to help troubleshoot
        st.caption(f"⚠️ .env not loaded | CWD: {os.getcwd()}")
        env_file = Path(__file__).resolve().parent.parent / ".env"
        st.caption(f"Looking for: {env_file} | Exists: {env_file.exists()}")

    openai_key = st.text_input("OpenAI API Key", type="password", value=default_key,
                                help="Get your API key from https://platform.openai.com/api-keys")

    model_options = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
    default_index = model_options.index(default_model) if default_model in model_options else 0
    openai_model = st.selectbox("Model", model_options, index=default_index)

    # Store in session state for use in explanation generation
    st.session_state['user_openai_key'] = openai_key
    st.session_state['user_openai_model'] = openai_model

    # Show status
    if openai_key:
        st.success("✓ OpenAI API Key Loaded")
    else:
        st.info("ℹ️ Enter your OpenAI API key above to enable explanation generation")

    st.markdown("---")

    # About
    st.markdown("### About HOGE Framework")
    st.markdown("""
    **Version:** 1.0.0
    **Paper:** *HOGE: Human-Centric Ontology-Grounded Explanation Framework*
    **Components:**
    - Component A: XGBoost + SHAP
    - Component B: Neo4j Knowledge Graph
    - Component C: GPT-4o Narrative Generator

    **Evaluation Metrics:**
    - Faithfulness Score: 0.760
    - Semantic Fidelity: 0.793
    - Evidence Coverage: 0.907
    - Hallucination Rate: 20.7%
    """)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "HOGE Framework © 2024 | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
