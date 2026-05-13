# Attention Mechanistic Interpretability - Implementation Plan

**Branch**: `feature/attention-mechanistic`
**Goal**: Extend HOGE framework to explain multimodal attention mechanisms in transformer models

---

## 🎯 Project Overview

Build mechanistic interpretability for attention-based multimodal sentiment analysis using CMU-MOSEI dataset. Explain:
- Which attention heads matter for each modality
- How cross-modal attention flows (text→audio, text→video, audio→video)
- Feature interactions within attention layers
- Circuit-level explanations for predictions

---

## 📊 Dataset: CMU-MOSEI

**Why CMU-MOSEI?**
- ✅ Already have infrastructure (`multimodal_model.py`, `cmu_mosei_loader.py`)
- ✅ Multimodal (text + audio + video) = rich attention patterns
- ✅ Sentiment analysis task (regression: -3 to +3)
- ✅ 23K+ utterances from YouTube videos

**Data Structure:**
```
- Text: BERT embeddings (768-dim)
- Audio: COVAREP features (74-dim)
- Video: Facet features (35-dim)
- Labels: Sentiment scores + 6 emotions
```

**Start Small:**
- Initial experiments: 100-500 samples
- Full dataset: 23K samples (train/val/test split)

---

## 🏗️ Architecture Plan

### 1. **Enhanced Multimodal Model with Attention Hooks**

```python
class AttentionProbeMultimodalModel(SimpleMultimodalModel):
    """
    Extended model that captures attention weights at every layer
    """
    def forward(self, text, audio, video, return_attention=False):
        # Capture BERT self-attention
        text_outputs = self.bert(text, output_attentions=True)
        text_attention = text_outputs.attentions  # (num_layers, batch, heads, seq, seq)

        # Capture cross-modal attention (if using attention fusion)
        cross_modal_attention = self.fusion_attention(...)

        if return_attention:
            return output, {
                'text_self_attention': text_attention,
                'cross_modal_attention': cross_modal_attention,
                'audio_importance': ...,
                'video_importance': ...
            }
        return output
```

### 2. **Attention Mechanistic Interpreter**

New file: `backend/src/explainability/attention_interpreter.py`

**Features:**
- Extract attention weights for any input
- Identify important attention heads
- Compute cross-modal attention flows
- Detect attention circuits (paths through heads)
- Generate natural language explanations

**Key Methods:**
```python
class AttentionMechanisticInterpreter:
    def extract_attention_patterns(utterance_id, model)
    def identify_important_heads(attention_weights, threshold=0.1)
    def compute_cross_modal_flows(text_attn, audio_attn, video_attn)
    def discover_attention_circuits(attention_graph)
    def generate_attention_narrative(patterns, audience='technical')
```

### 3. **Attention Visualizations**

New file: `backend/src/explainability/attention_visualizer.py`

**Visualizations:**
1. **Attention heatmaps** (which tokens attend to which)
2. **Head importance plots** (which heads matter most)
3. **Cross-modal flow diagrams** (Sankey diagram: text→audio→video)
4. **Attention circuits** (graph showing head→head connections)
5. **Layer-wise attention evolution** (how attention changes across layers)

### 4. **Knowledge Graph Integration**

Extend Neo4j schema:
```cypher
// New node types
(:AttentionHead {layer: int, head_id: int, importance: float})
(:AttentionPattern {pattern_type: str, strength: float})
(:ModalityInteraction {source_modality: str, target_modality: str, attention_flow: float})

// New relationships
(:Utterance)-[:HAS_ATTENTION_PATTERN]->(:AttentionPattern)
(:AttentionHead)-[:ATTENDS_TO]->(:Token)
(:AttentionHead)-[:INFLUENCES]->(:AttentionHead)  // Circuit connections
(:AttentionPattern)-[:CONTRIBUTES_TO]->(:Prediction)
```

---

## 📝 Implementation Steps

### **Phase 1: Dataset & Model Setup** (Week 1)

1. ✅ Generate/download small CMU-MOSEI sample (100 utterances)
   ```bash
   python .development/scripts/generate_synthetic_mosei.py --n-samples 100
   # OR
   python .development/scripts/download_cmu_mosei.py --subset mini
   ```

2. Train baseline multimodal model
   ```bash
   python backend/src/models/train_multimodal.py --epochs 10 --batch-size 16
   ```

3. Verify model works and achieves reasonable sentiment prediction

### **Phase 2: Attention Extraction** (Week 1-2)

4. Implement `AttentionProbeMultimodalModel`
   - Add attention hooks to BERT layers
   - Add cross-modal attention capture
   - Test: Extract attention for 1 sample, verify shapes

5. Create `AttentionMechanisticInterpreter`
   - Implement head importance scoring
   - Implement cross-modal flow computation
   - Test: Analyze 1 utterance, output JSON

### **Phase 3: Analysis & Visualization** (Week 2)

6. Build attention visualizations
   - Attention heatmaps (Plotly)
   - Head importance bar charts
   - Cross-modal flow Sankey diagram
   - Test: Generate viz for 1 sample

7. Implement circuit discovery
   - Build attention graph (heads as nodes)
   - Find important paths using graph algorithms
   - Rank circuits by impact on prediction

### **Phase 4: Knowledge Graph Integration** (Week 3)

8. Extend Neo4j schema for attention
   - Create attention node types
   - Load attention patterns into KG

9. Link attention patterns to predictions
   - Connect attention heads → features → prediction
   - Enable queries like: "Which heads influenced this decision?"

### **Phase 5: Narrative Generation** (Week 3)

10. Extend LLM explainer for attention
    - New prompt template for attention explanations
    - Example: "The model focused primarily on the word 'angry' (head 3.5)
                and aligned it with rising pitch in audio (head 7.2)..."

### **Phase 6: Streamlit UI** (Week 4)

11. Add "Attention Analysis" tab to frontend
    - Upload utterance or select from dataset
    - Show attention visualizations
    - Display natural language explanation
    - Compare technical vs non-technical audience views

---

## 🔬 Research Questions to Answer

1. **Head Specialization**: Do specific heads specialize in specific modalities?
2. **Cross-Modal Fusion**: How does information flow between text/audio/video?
3. **Sentiment Encoding**: Which heads encode positive vs negative sentiment?
4. **Error Analysis**: When model fails, which attention patterns are wrong?
5. **Audience Adaptation**: Can we explain attention to non-experts?

---

## 📦 Deliverables

1. **Code**:
   - `attention_interpreter.py` (core analysis)
   - `attention_visualizer.py` (visualizations)
   - `train_multimodal.py` (model training)
   - Neo4j schema extensions

2. **Data**:
   - Trained multimodal model (`.resources/models/multimodal/`)
   - CMU-MOSEI sample dataset (`.resources/data/cmu_mosei/`)
   - Attention analysis results (`.resources/data/evaluation/`)

3. **Documentation**:
   - This plan (ATTENTION_MECHANISTIC_PLAN.md)
   - API documentation for new modules
   - Jupyter notebook with examples

4. **UI**:
   - Streamlit "Attention Analysis" page
   - Interactive attention visualizations
   - Audience-adaptive explanations

---

## 🚀 Next Steps (Start Now)

```bash
# 1. Check if you have CMU-MOSEI data
ls .resources/data/cmu_mosei/

# 2. If not, generate synthetic sample for testing
python .development/scripts/generate_synthetic_mosei.py --n-samples 100

# 3. Test multimodal model infrastructure
python .development/scripts/test_multimodal_setup.py

# 4. Start implementing AttentionProbeMultimodalModel
# Create: backend/src/models/attention_probe_model.py
```

---

## 📚 Key Papers to Reference

1. **Attention Interpretability**: "Analyzing Multi-Head Self-Attention" (Voita et al., 2019)
2. **Multimodal Attention**: "MAG-BERT" (Rahman et al., 2020)
3. **Circuits**: "In-context Learning and Induction Heads" (Olsson et al., 2022)
4. **MOSEI Dataset**: "Multimodal Language Analysis in the Wild" (Zadeh et al., 2018)

---

## 🎓 PhD Contribution

**Novelty:**
- First work to apply mechanistic interpretability to **multimodal** transformers
- Cross-modal attention circuits not studied before
- Ontology-grounded explanations for attention mechanisms
- Human-centric explanations for attention (technical vs non-technical)

**Comparison to Prior Work:**
- Tree-based models (XGBoost): Path-level analysis ✅ (already done)
- Attention models: Head-level + circuit analysis 🔜 (this branch)
- Vision models: Feature attribution ⏳ (future work)
