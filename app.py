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
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.explainability.llm_explainer import get_application_explanation_data, call_llm_for_explanation
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="HOGE - Explainable Loan Decisions",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

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


# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50.png?text=HOGE", width=150)
    st.markdown("### 🏦 HOGE Framework")
    st.markdown("Human-Centric Ontology-Grounded Explanations")

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Explain Application", "📊 Batch Analysis", "⚙️ Settings"]
    )

    st.markdown("---")
    st.markdown("#### System Status")

    # Check Neo4j connection
    try:
        NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        st.success("✅ Neo4j Connected")
        driver.close()
    except:
        st.error("❌ Neo4j Disconnected")

    # Check OpenAI key
    if os.getenv("OPENAI_API_KEY"):
        st.success("✅ OpenAI Configured")
    else:
        st.warning("⚠️ OpenAI Key Missing")


# Main content
if page == "🏠 Home":
    st.markdown('<div class="main-header">🏦 HOGE Framework</div>', unsafe_allow_html=True)
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
        st.markdown("#### 📈 Model Performance")
        st.metric("ROC-AUC", "0.839")
        st.metric("Accuracy", "80%")

    with col2:
        st.markdown("#### 🎯 Explanation Quality")
        st.metric("Faithfulness", "0.760")
        st.metric("Hallucination Rate", "20.7%")

    with col3:
        st.markdown("#### 🕸️ Knowledge Graph")
        st.metric("Total Nodes", "23,000+")
        st.metric("Applications", "500")

    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. Select **🔍 Explain Application** from the sidebar
    2. Enter a Loan ID (e.g., LP001006)
    3. Click **Generate Explanation**
    4. Explore SHAP values, policy violations, and counterfactual scenarios
    """)


elif page == "🔍 Explain Application":
    st.markdown('<div class="main-header">🔍 Explain Loan Application</div>', unsafe_allow_html=True)

    # Input section
    col1, col2 = st.columns([3, 1])

    with col1:
        loan_id = st.text_input(
            "Enter Loan Application ID:",
            value="LP001006",
            help="Example: LP001006, LP001002, LP001003"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("🚀 Generate Explanation", type="primary", use_container_width=True)

    if generate_btn and loan_id:
        with st.spinner(f"Generating explanation for {loan_id}..."):
            try:
                # Get context from KG
                context = get_application_explanation_data(loan_id)

                # Generate LLM explanation
                explanation = call_llm_for_explanation(context)

                # Display results
                st.success(f"✅ Explanation generated successfully!")

                # Decision summary
                st.markdown("---")
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

                # Drivers
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="sub-header">✅ Positive Drivers</div>', unsafe_allow_html=True)
                    if explanation['positive_drivers']:
                        for driver in explanation['positive_drivers']:
                            st.markdown(f'<div class="positive-driver">+ {driver}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("*No positive drivers identified*")

                with col2:
                    st.markdown('<div class="sub-header">❌ Negative Drivers</div>', unsafe_allow_html=True)
                    if explanation['negative_drivers']:
                        for driver in explanation['negative_drivers']:
                            st.markdown(f'<div class="negative-driver">- {driver}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("*No negative drivers identified*")

                # Policy violations
                if explanation['policy_violations']:
                    st.markdown("---")
                    st.markdown('<div class="sub-header">⚠️ Policy Violations</div>', unsafe_allow_html=True)
                    for violation in explanation['policy_violations']:
                        st.markdown(f'<div class="policy-violation">! {violation}</div>', unsafe_allow_html=True)

                # Counterfactual
                st.markdown("---")
                st.markdown('<div class="sub-header">🔮 What-If Analysis</div>', unsafe_allow_html=True)
                st.info(f"**{explanation.get('counterfactual_what_if', 'No counterfactual scenarios available.')}**")

                # Recommendation
                st.markdown("---")
                st.markdown('<div class="sub-header">💡 Recommendation</div>', unsafe_allow_html=True)
                st.success(explanation['recommendation'])

                # Technical details (expandable)
                with st.expander("🔬 Technical Details"):
                    st.markdown("#### SHAP Feature Contributions (Top 10)")

                    shap_df = pd.DataFrame(context['shap_details'][:10])
                    st.dataframe(shap_df, use_container_width=True)

                    st.markdown("#### Features Used in Explanation")
                    st.write(explanation['features_used'])

                    st.markdown("#### Provenance")
                    st.json(explanation.get('provenance', {}))

                # Download options
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        "📥 Download Full Explanation (JSON)",
                        data=json.dumps(explanation, indent=2, default=str),
                        file_name=f"explanation_{loan_id}.json",
                        mime="application/json"
                    )

                with col2:
                    st.download_button(
                        "📥 Download Context Data (JSON)",
                        data=json.dumps(context, indent=2, default=str),
                        file_name=f"context_{loan_id}.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"❌ Error generating explanation: {str(e)}")
                st.exception(e)


elif page == "📊 Batch Analysis":
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

        if st.button("🚀 Analyze Selected Applications", type="primary"):
            if not selected_apps:
                st.warning("Please select at least one application")
            else:
                results = []
                progress_bar = st.progress(0)

                for i, app_id in enumerate(selected_apps):
                    try:
                        context = get_application_explanation_data(app_id)
                        explanation = call_llm_for_explanation(context)

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
                    st.dataframe(results_df, use_container_width=True)

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
        st.error("❌ df1_loan.csv not found. Please ensure the data file is in the root directory.")


elif page == "⚙️ Settings":
    st.markdown('<div class="main-header">⚙️ System Settings</div>', unsafe_allow_html=True)

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
            st.success("✅ Connection successful!")
            driver.close()
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")

    st.markdown("---")

    # OpenAI settings
    st.markdown("#### 🤖 OpenAI Configuration")
    openai_key = st.text_input("OpenAI API Key", type="password", value="")
    openai_model = st.selectbox("Model", ["gpt-4o", "gpt-4", "gpt-3.5-turbo"], index=0)

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
