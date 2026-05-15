"""
Attention Mechanistic Analysis Dashboard
=========================================
Interactive Streamlit app for exploring attention patterns in BERT.

Run: streamlit run frontend/attention_dashboard.py
"""

import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import novel research modules
from backend.src.explainability.attention_circuits import (
    analyze_attention_mechanisms,
    generate_attention_narrative,
    detect_induction_heads,
    discover_attention_circuits,
    classify_attention_head_type
)

# Page config
st.set_page_config(
    page_title="Attention Analysis Dashboard",
    page_icon="🔬",
    layout="wide"
)

# Title
st.markdown("# 🔬 Attention Mechanistic Analysis Dashboard")
st.markdown("Explore how BERT attention mechanisms process different texts")

# Sidebar
st.sidebar.markdown("## ⚙️ Settings")

# Model selection
st.sidebar.markdown("## 🤖 Model Selection")
model_options = {
    "BERT-base": "bert-base-uncased",
    "RoBERTa-base": "roberta-base",
    "DistilBERT": "distilbert-base-uncased",
    "ALBERT-base": "albert-base-v2"
}
selected_model_name = st.sidebar.selectbox(
    "Choose transformer model",
    list(model_options.keys()),
    help="Different models have different attention patterns"
)
selected_model_path = model_options[selected_model_name]

# Cache model loading
@st.cache_resource
def load_model(model_path):
    """Load transformer model and tokenizer"""
    with st.spinner(f"Loading {model_path}..."):
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, attn_implementation="eager")
        model.eval()
    return tokenizer, model, model_path

tokenizer, model, loaded_model = load_model(selected_model_path)
st.sidebar.success(f"✓ {selected_model_name} loaded!")

# Detect model architecture
num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads
st.sidebar.info(f"📊 Architecture: {num_layers} layers, {num_heads} heads per layer")

# Layer and head selection
st.sidebar.markdown("### Visualization Options")
selected_layer = st.sidebar.slider("Layer", 0, num_layers-1, num_layers//2, help=f"Model has {num_layers} layers")
selected_head = st.sidebar.slider("Head", 0, num_heads-1, num_heads//4, help=f"Each layer has {num_heads} attention heads")

# Analysis mode
analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Single Text", "Compare Two Texts", "Batch Analysis"]
)

# Research features (novel contributions)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Research Features (PhD Contributions)")
enable_llm_narrative = st.sidebar.checkbox(
    "LLM Narrative Generation (RC2)",
    value=True,
    help="Use GPT-4 to generate human-readable explanations"
)
enable_circuit_discovery = st.sidebar.checkbox(
    "Circuit Discovery",
    value=True,
    help="Find attention paths from input -> prediction"
)
enable_induction_heads = st.sidebar.checkbox(
    "Induction Head Detection",
    value=True,
    help="Detect heads that copy patterns (Olsson et al., 2022)"
)
explanation_audience = st.sidebar.radio(
    "Explanation Style",
    ["non-technical", "technical"],
    help="Technical: entropy, layer numbers. Non-technical: simple language"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Attention Mechanistic Analysis**

This dashboard helps you understand:
- Which words BERT focuses on
- How attention flows through layers
- Which heads are most important
- Differences between positive/negative texts
""")

# Helper functions
def extract_attention(text):
    """Extract attention weights for text"""
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    return {
        'tokens': tokens,
        'attention': outputs.attentions,  # (12 layers, batch, 12 heads, seq, seq)
        'inputs': inputs
    }

def calculate_head_importance(attentions):
    """Calculate entropy-based importance for all heads"""
    importance = []

    for layer_idx in range(len(attentions)):
        for head_idx in range(attentions[0].shape[1]):
            attn = attentions[layer_idx][0, head_idx].numpy()
            probs = attn + 1e-10
            entropy = -np.sum(probs * np.log(probs), axis=-1).mean()

            importance.append({
                'Layer': layer_idx,
                'Head': head_idx,
                'Entropy': entropy,
                'Focus': 1 / (entropy + 1)  # Higher = more focused
            })

    return pd.DataFrame(importance)

def create_attention_heatmap(attention_matrix, tokens, layer, head, title=None):
    """Create interactive Plotly heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        text=np.round(attention_matrix, 3),
        texttemplate='%{text}',
        textfont={"size": 8},
        hovertemplate='From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=title or f"Attention Pattern: Layer {layer}, Head {head}",
        xaxis_title="Attended Token (Key)",
        yaxis_title="Attending Token (Query)",
        height=600,
        font=dict(size=10)
    )

    fig.update_xaxes(tickangle=45)

    return fig

# Main content area
if analysis_mode == "Single Text":
    st.markdown("## 📝 Single Text Analysis")

    # Text input
    default_text = "I love this movie! It's amazing and wonderful."
    text = st.text_area(
        "Enter text to analyze:",
        value=default_text,
        height=100,
        help="Enter any sentence to see how BERT processes it"
    )

    if st.button("🔍 Analyze Attention", type="primary"):
        with st.spinner("Extracting attention patterns..."):
            result = extract_attention(text)
            tokens = result['tokens']
            attentions = result['attention']

        # Run comprehensive analysis with research features
        with st.spinner("Running mechanistic analysis (circuit discovery, induction heads, LLM narratives)..."):
            analysis = analyze_attention_mechanisms(
                tokens=tokens,
                attentions=attentions,
                prediction=None,  # Could add sentiment classifier here
                audience=explanation_audience,
                enable_llm_narrative=enable_llm_narrative,
                enable_circuit_discovery=enable_circuit_discovery
            )

        st.success(f"✓ Analyzed {len(tokens)} tokens across {num_layers * num_heads} attention heads")

        # Show LLM-generated narrative (RC2)
        if analysis['narrative']:
            st.markdown("### 🤖 AI-Generated Explanation")
            st.info(analysis['narrative'])
            st.caption(f"Audience: {explanation_audience} | Generated using {os.getenv('OPENAI_MODEL', 'gpt-4o')}")

        # Show tokens
        st.markdown("### Tokens")
        st.code(" | ".join(tokens))

        # Create tabs with new research features
        tabs = ["📊 Attention Heatmap", "📈 Head Importance", "🎯 Focus Words"]
        if enable_circuit_discovery:
            tabs.append("🔗 Attention Circuits")
        if enable_induction_heads:
            tabs.append("🔄 Induction Heads")

        tab_objects = st.tabs(tabs)
        tab1 = tab_objects[0]
        tab2 = tab_objects[1]
        tab3 = tab_objects[2]

        with tab1:
            st.markdown(f"### Attention Pattern: Layer {selected_layer}, Head {selected_head}")

            attention_matrix = attentions[selected_layer][0, selected_head].numpy()
            fig = create_attention_heatmap(attention_matrix, tokens, selected_layer, selected_head)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **How to read this heatmap:**
            - **Rows** (Y-axis): Tokens asking "what should I attend to?"
            - **Columns** (X-axis): Tokens being attended to
            - **Darker blue**: Stronger attention
            - **Hover** over cells to see exact attention values
            """)

        with tab2:
            st.markdown("### Head Importance Ranking")

            importance_df = calculate_head_importance(attentions)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Top 10 Most Focused Heads")
                top_focused = importance_df.nsmallest(10, 'Entropy')
                st.dataframe(
                    top_focused[['Layer', 'Head', 'Entropy']].reset_index(drop=True),
                    use_container_width=True
                )

            with col2:
                st.markdown("#### Top 10 Least Focused Heads")
                top_dispersed = importance_df.nlargest(10, 'Entropy')
                st.dataframe(
                    top_dispersed[['Layer', 'Head', 'Entropy']].reset_index(drop=True),
                    use_container_width=True
                )

            # Heatmap of all heads
            st.markdown("#### Attention Focus Across All Heads")

            pivot = importance_df.pivot(index='Layer', columns='Head', values='Entropy')

            fig_heads = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn_r',
                text=np.round(pivot.values, 2),
                texttemplate='%{text}',
                textfont={"size": 8},
                hovertemplate='Layer: %{y}<br>Head: %{x}<br>Entropy: %{z:.3f}<extra></extra>'
            ))

            fig_heads.update_layout(
                title="Entropy by Layer and Head (Lower = More Focused)",
                xaxis_title="Head",
                yaxis_title="Layer",
                height=500
            )

            st.plotly_chart(fig_heads, use_container_width=True)

        with tab3:
            st.markdown("### Focus Words Analysis")

            # Use analysis results
            focus_words_data = analysis['focus_words']

            st.markdown("#### Most Attended Tokens")
            focus_df = pd.DataFrame(focus_words_data)
            st.dataframe(focus_df, use_container_width=True)

            st.markdown("#### Head Type Classification (RC4)")
            st.info("""
            **Novel Contribution**: Automatic head taxonomy
            - **Sentiment heads**: Focus on emotional words
            - **Syntax heads**: Focus on grammatical structure
            - **Induction heads**: Copy patterns
            - **Broadcast heads**: Attend uniformly
            """)

            # Show head types from analysis
            head_types = {}
            for head_info in analysis['top_heads'][:10]:
                head_type = head_info.get('type', 'unknown')
                head_types[head_type] = head_types.get(head_type, 0) + 1

            st.bar_chart(pd.DataFrame({'Count': head_types}))

            # Find sentiment words for backward compatibility
            sentiment_words = ['love', 'amazing', 'wonderful', 'great', 'fantastic',
                             'hate', 'terrible', 'awful', 'horrible', 'bad', 'pathetic']
            sentiment_indices = [i for i, token in enumerate(tokens)
                               if token.lower() in sentiment_words]

            if sentiment_indices:
                st.success(f"Found {len(sentiment_indices)} sentiment words: {[tokens[i] for i in sentiment_indices]}")

                # Calculate attention to sentiment words
                sentiment_attention = []
                for layer_idx in range(len(attentions)):
                    for head_idx in range(12):
                        attn = attentions[layer_idx][0, head_idx].numpy()
                        attn_to_sentiment = attn[:, sentiment_indices].mean()

                        sentiment_attention.append({
                            'Layer': layer_idx,
                            'Head': head_idx,
                            'Sentiment Focus': attn_to_sentiment
                        })

                sentiment_df = pd.DataFrame(sentiment_attention)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Top 10 Sentiment-Focused Heads")
                    top_sentiment = sentiment_df.nlargest(10, 'Sentiment Focus')
                    st.dataframe(
                        top_sentiment[['Layer', 'Head', 'Sentiment Focus']].reset_index(drop=True),
                        use_container_width=True
                    )

                with col2:
                    # Visualize attention to sentiment words for selected head
                    st.markdown(f"#### Layer {selected_layer}, Head {selected_head} → Sentiment Words")

                    attn = attentions[selected_layer][0, selected_head].numpy()
                    sentiment_attn_values = attn[:, sentiment_indices].mean(axis=1)

                    fig_sentiment = go.Figure(data=[
                        go.Bar(
                            x=tokens,
                            y=sentiment_attn_values,
                            marker_color='lightblue'
                        )
                    ])

                    fig_sentiment.update_layout(
                        title="Average Attention to Sentiment Words",
                        xaxis_title="Token",
                        yaxis_title="Attention Weight",
                        height=300
                    )

                    st.plotly_chart(fig_sentiment, use_container_width=True)

            else:
                st.warning("No sentiment words found in this text. Try adding words like 'love', 'hate', 'amazing', etc.")

        # New research tabs
        if enable_circuit_discovery and len(tab_objects) > 3:
            with tab_objects[3]:
                st.markdown("### 🔗 Attention Circuits (Novel Contribution)")
                st.info("""
                **Research Contribution**: Circuit discovery inspired by Anthropic's work (Wang et al., 2023)

                Traces attention paths from input tokens → heads → prediction
                """)

                circuits_data = analysis['circuits']
                if circuits_data:
                    for i, circuit in enumerate(circuits_data):
                        with st.expander(f"Circuit {i+1}: {circuit['input_token']} (total attention: {circuit['total_attention']:.3f})"):
                            st.markdown(f"**Input token**: `{circuit['input_token']}`")
                            st.markdown(f"**Path depth**: {circuit['depth']} layers")

                            # Show path
                            path_str = " → ".join([
                                f"L{p['layer']}H{p['head']} ({p['attention']:.2f})"
                                for p in circuit['path']
                            ])
                            st.code(path_str)
                else:
                    st.warning("No strong circuits detected. Try text with clear sentiment words.")

        if enable_induction_heads and len(tab_objects) > 4:
            with tab_objects[4]:
                st.markdown("### 🔄 Induction Heads (Olsson et al., 2022)")
                st.info("""
                **Research Contribution**: Detection of induction heads in transformers

                Induction heads perform pattern matching: [A][B] ... [A] → predict [B]

                These are key to in-context learning abilities.
                """)

                induction_data = analysis['induction_heads']
                if induction_data:
                    induction_df = pd.DataFrame(induction_data)
                    st.dataframe(induction_df, use_container_width=True)

                    st.markdown("#### Induction Score Distribution")
                    fig_induction = go.Figure(data=[
                        go.Bar(
                            x=[f"L{h['layer']}H{h['head']}" for h in induction_data],
                            y=[h['induction_score'] for h in induction_data],
                            marker_color='lightgreen'
                        )
                    ])
                    fig_induction.update_layout(
                        title="Induction Head Scores",
                        xaxis_title="Head",
                        yaxis_title="Induction Score",
                        height=400
                    )
                    st.plotly_chart(fig_induction, use_container_width=True)
                else:
                    st.warning("No induction heads detected. Try text with repeated patterns.")

elif analysis_mode == "Compare Two Texts":
    st.markdown("## 🔄 Compare Two Texts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Text 1 (e.g., Positive)")
        text1 = st.text_area(
            "Enter first text:",
            value="I love this movie! It's amazing.",
            height=100,
            key="text1"
        )

    with col2:
        st.markdown("### Text 2 (e.g., Negative)")
        text2 = st.text_area(
            "Enter second text:",
            value="I hate this movie! It's terrible.",
            height=100,
            key="text2"
        )

    if st.button("🔍 Compare Attention", type="primary"):
        with st.spinner("Analyzing both texts..."):
            result1 = extract_attention(text1)
            result2 = extract_attention(text2)

        st.success("✓ Comparison complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### Text 1: Layer {selected_layer}, Head {selected_head}")
            attn1 = result1['attention'][selected_layer][0, selected_head].numpy()
            fig1 = create_attention_heatmap(attn1, result1['tokens'], selected_layer, selected_head, "Text 1")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown(f"### Text 2: Layer {selected_layer}, Head {selected_head}")
            attn2 = result2['attention'][selected_layer][0, selected_head].numpy()
            fig2 = create_attention_heatmap(attn2, result2['tokens'], selected_layer, selected_head, "Text 2")
            st.plotly_chart(fig2, use_container_width=True)

        # Difference analysis
        st.markdown("### 📊 Attention Difference Analysis")

        importance1 = calculate_head_importance(result1['attention'])
        importance2 = calculate_head_importance(result2['attention'])

        # Find heads with biggest difference
        importance1['Text'] = 'Text 1'
        importance2['Text'] = 'Text 2'

        merged = importance1.merge(
            importance2,
            on=['Layer', 'Head'],
            suffixes=('_1', '_2')
        )
        merged['Difference'] = abs(merged['Entropy_1'] - merged['Entropy_2'])

        st.markdown("#### Heads with Biggest Attention Differences")
        top_diff = merged.nlargest(10, 'Difference')[['Layer', 'Head', 'Entropy_1', 'Entropy_2', 'Difference']]
        st.dataframe(top_diff.reset_index(drop=True), use_container_width=True)

elif analysis_mode == "Batch Analysis":
    st.markdown("## 📦 Batch Analysis")

    st.markdown("Analyze multiple sentences at once to find patterns")

    # Predefined examples
    examples = [
        "I love this movie! It's amazing and wonderful.",
        "I hate this movie! It's terrible and boring.",
        "The movie is okay, nothing special.",
        "Great acting but terrible plot.",
        "This is the best film I've ever seen!"
    ]

    use_examples = st.checkbox("Use example sentences", value=True)

    if use_examples:
        texts = examples
        st.info(f"Using {len(examples)} example sentences")
        for i, text in enumerate(texts):
            st.text(f"{i+1}. {text}")
    else:
        text_input = st.text_area(
            "Enter sentences (one per line):",
            height=200,
            placeholder="I love this\nI hate this\n..."
        )
        texts = [t.strip() for t in text_input.split('\n') if t.strip()]

    if st.button("🔍 Analyze Batch", type="primary") and texts:
        with st.spinner(f"Analyzing {len(texts)} texts..."):
            results = [extract_attention(text) for text in texts]

        st.success(f"✓ Analyzed {len(texts)} texts")

        # Aggregate head importance
        all_importance = []
        for i, result in enumerate(results):
            imp = calculate_head_importance(result['attention'])
            imp['Text_ID'] = i
            imp['Text'] = texts[i][:30] + "..." if len(texts[i]) > 30 else texts[i]
            all_importance.append(imp)

        combined = pd.concat(all_importance)

        # Average importance across all texts
        avg_importance = combined.groupby(['Layer', 'Head'])['Entropy'].mean().reset_index()
        avg_importance = avg_importance.sort_values('Entropy')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Most Consistent Focused Heads")
            st.dataframe(
                avg_importance.head(10)[['Layer', 'Head', 'Entropy']].reset_index(drop=True),
                use_container_width=True
            )

        with col2:
            st.markdown("#### Most Variable Heads (Across Texts)")
            variance = combined.groupby(['Layer', 'Head'])['Entropy'].std().reset_index()
            variance = variance.sort_values('Entropy', ascending=False)
            variance.columns = ['Layer', 'Head', 'Variance']
            st.dataframe(
                variance.head(10).reset_index(drop=True),
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("**Attention Mechanistic Analysis Dashboard** | Built for attention interpretability research")
