# Attention Mechanistic Interpretability - Implementation Plan

**Branch**: `feature/attention-mechanistic`
**Goal**: Extend HOGE framework to explain multimodal attention mechanisms in transformer models

---

## 🎯 Project Overview

**Main Goal**: Extend HOGE's mechanistic interpretability from tree-based models (XGBoost) to attention-based transformers (multimodal).

**Concrete Outputs:**
1. **Attention Head Analysis**: Which heads specialize in each modality (text/audio/video)?
2. **Cross-Modal Circuits**: How does information flow between modalities in attention layers?
3. **Sentiment Mechanisms**: What attention patterns distinguish positive vs negative sentiment?
4. **Human Explanations**: Can non-experts understand attention-based explanations?

**PhD Contribution**: First mechanistic interpretability framework for multimodal transformers with human-centric explanations

---

## 📊 Dataset Decision: CMU-MOSEI vs Alternatives

### **Option 1: CMU-MOSEI (RECOMMENDED)** ⭐
**Pros:**
- ✅ Infrastructure ready (`multimodal_model.py`, `cmu_mosei_loader.py`)
- ✅ True multimodal (text + audio + video) → rich cross-modal attention
- ✅ Real-world YouTube data → practical insights
- ✅ 23K utterances → enough for analysis

**Cons:**
- ⚠️ Large download (~20GB raw)
- ⚠️ Preprocessing complexity
- ⚠️ Requires more compute

**Pragmatic Approach:**
- **Phase 1** (Weeks 1-2): Use synthetic data (100 samples) ✅ Already generated
- **Phase 2** (Week 3): Download mini-subset (500 samples)
- **Phase 3** (Week 4): Full dataset if needed for paper results

### **Option 2: Pure NLP (SST-2 Sentiment)**
**Pros:**
- ✅ Smaller, faster (67K samples, text-only)
- ✅ Well-studied baseline
- ✅ Can reuse BERT attention tools

**Cons:**
- ❌ No cross-modal attention (loses uniqueness)
- ❌ Less novel for PhD contribution

### **Decision: Start with CMU-MOSEI synthetic → validate approach → scale up**

**Data Structure (CMU-MOSEI):**
```
Input:
- Text: Raw transcript → BERT tokens → (batch, seq_len)
- Audio: COVAREP features → (batch, seq_len, 74)
- Video: Facet features → (batch, seq_len, 35)

Output:
- Sentiment score: [-3.0, +3.0] (regression)
- Emotions: {happiness, sadness, anger, fear, disgust, surprise} × [0, 3]

Attention:
- BERT self-attention: (12 layers, 12 heads, seq×seq)
- Cross-modal attention: (text→audio, text→video, fusion)
```

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

## 📝 Implementation Roadmap (Refined)

### **Week 1: Foundation - Minimal Viable Analysis** 🎯

**Goal**: Get 1 utterance analyzed end-to-end

**Tasks:**
1. ✅ **Data Ready**: Synthetic MOSEI (100 samples) - DONE
2. **Model Setup**:
   - [ ] Train simple multimodal model on synthetic data (just needs to run, doesn't need high accuracy)
   - [ ] Save trained model to `.resources/models/multimodal/`
   - **Success metric**: Model predicts sentiment for 1 sample

3. **Attention Extraction**:
   - [ ] Modify `SimpleMultimodalModel` to return BERT attention weights
   - [ ] Extract attention for 1 sample
   - [ ] Save to JSON: `{layer_0_head_0: [...], layer_0_head_1: [...]}`
   - **Success metric**: Can see attention weights in JSON

4. **Basic Visualization**:
   - [ ] Create simple heatmap: One attention head for one sample
   - [ ] Plot using matplotlib or Plotly
   - **Success metric**: Can see which words attend to which

**Deliverable**: Jupyter notebook showing attention for 1 utterance with 1 heatmap

---

### **Week 2: Analysis - Head Importance & Patterns** 🔬

**Goal**: Understand which heads matter and why

**Tasks:**
5. **Head Importance Scoring**:
   - [ ] Implement attention flow metric: How much information flows through each head?
   - [ ] Ablation test: Zero out each head, measure impact on prediction
   - [ ] Rank heads by importance
   - **Success metric**: "Head 3.5 has 0.42 importance, Head 8.2 has 0.15"

6. **Pattern Discovery**:
   - [ ] Text-modality heads: Which heads focus on sentiment words?
   - [ ] Cross-modal heads: Which heads bridge text→audio or text→video?
   - [ ] Categorize heads: {sentiment-focused, syntax-focused, cross-modal}
   - **Success metric**: Can say "Head X specializes in positive sentiment words"

7. **Visualization Suite**:
   - [ ] Head importance bar chart
   - [ ] Attention heatmap (multiple heads side-by-side)
   - [ ] Cross-modal flow diagram (simple version)
   - **Success metric**: 3 publication-ready visualizations

**Deliverable**: Analysis report showing "Top 5 most important heads and their roles"

---

### **Week 3: Integration - HOGE Framework** 🕸️

**Goal**: Make it work like existing XGBoost mechanistic analysis

**Tasks:**
8. **Attention Interpreter Class**:
   - [ ] Create `AttentionMechanisticInterpreter` (similar to `MechanisticInterpreter` for XGBoost)
   - [ ] Methods: `extract_patterns()`, `rank_heads()`, `compute_flows()`
   - [ ] Output format: JSON matching XGBoost analysis structure
   - **Success metric**: Can call `interpreter.analyze(utterance_id)` and get results

9. **Neo4j Integration** (OPTIONAL - can skip for MVP):
   - [ ] Design schema: `(:AttentionHead)`, `(:AttentionPattern)`
   - [ ] Loader script: Save attention analysis to KG
   - [ ] Query: "Which heads influenced this prediction?"
   - **Success metric**: Can query Neo4j for attention insights

10. **LLM Narrative** (SIMPLE version):
    - [ ] Prompt template: "Given these attention patterns: {top_heads}, explain sentiment prediction"
    - [ ] Generate 1-paragraph explanation
    - [ ] Test with 3 samples
    - **Success metric**: Human-readable explanation of attention

**Deliverable**: `attention_interpreter.py` module that works like existing mechanistic analysis

---

### **Week 4: Interface & Evaluation** 🎨

**Goal**: Make it usable and measure quality

**Tasks:**
11. **Streamlit UI**:
    - [ ] Add "Attention Analysis" page to frontend
    - [ ] Input: Select utterance from dropdown
    - [ ] Output: Show visualizations + narrative
    - [ ] Audience toggle: Technical vs Non-technical
    - **Success metric**: Can demo to advisor in Streamlit

12. **Evaluation**:
    - [ ] Faithfulness: Do important heads actually impact prediction?
    - [ ] Interpretability: Can humans identify sentiment from attention alone?
    - [ ] User study (if time): Show to 3-5 people, ask "Does this make sense?"
    - **Success metric**: Attention explanations ≥70% faithful

13. **Documentation**:
    - [ ] README for attention analysis
    - [ ] Code comments
    - [ ] Example notebook
    - **Success metric**: Someone else can run your code

**Deliverable**: Working Streamlit demo + evaluation results

---

## 🎯 Minimum Viable Product (MVP) Scope

**MUST HAVE** (Core PhD contribution):
- ✅ Attention extraction from multimodal model
- ✅ Head importance ranking
- ✅ Cross-modal attention flow analysis
- ✅ Human-readable explanations
- ✅ Visualizations (heatmaps + importance plots)

**NICE TO HAVE** (Enhance paper):
- ⭐ Circuit discovery (attention head graphs)
- ⭐ Neo4j knowledge graph integration
- ⭐ Audience-adaptive explanations (technical vs non-technical)

**CAN SKIP** (Future work):
- ⏸️ Full CMU-MOSEI dataset (synthetic is fine for proof-of-concept)
- ⏸️ Advanced circuit analysis (e.g., induction heads)
- ⏸️ Formal user study (informal feedback is enough)

---

## ⚠️ Risk Mitigation

**Risk 1**: "Model doesn't learn on synthetic data"
- **Mitigation**: Synthetic data has clear sentiment signals → should work
- **Backup**: Use pre-trained BERT + simple fusion (skip training)

**Risk 2**: "Attention patterns too noisy to interpret"
- **Mitigation**: Aggregate across multiple samples, use top-k heads only
- **Backup**: Focus on 1-2 clear examples instead of statistical analysis

**Risk 3**: "Cross-modal attention doesn't exist in simple model"
- **Mitigation**: Add explicit cross-attention layer in model architecture
- **Backup**: Focus on BERT self-attention only (still novel for multimodal sentiment)

**Risk 4**: "Takes too long to implement"
- **Mitigation**: Cut scope to MVP (skip Neo4j, skip circuits, skip full dataset)
- **Backup**: Treat as "pilot study" for future work, get 1 working example

---

## 🔬 Research Questions (Prioritized for PhD)

### **RQ1: Head Specialization** (CORE - Week 2)
"Do multimodal transformers develop specialized attention heads for each modality?"
- **Hypothesis**: Some heads focus on text sentiment, others on audio prosody, others on facial expressions
- **Method**: Ablation study - remove each head, measure modality-specific performance drop
- **Metric**: Head importance score per modality
- **Expected Result**: Find 3-5 "text specialist" heads, 2-3 "audio specialist" heads

### **RQ2: Cross-Modal Attention Circuits** (CORE - Week 3)
"How do attention mechanisms integrate information across text, audio, and video?"
- **Hypothesis**: Multi-hop attention paths (e.g., text→audio→fusion) exist
- **Method**: Trace attention flow through layers, build head dependency graph
- **Metric**: Cross-modal attention strength, circuit impact on prediction
- **Expected Result**: Identify 2-3 key "fusion circuits" that combine modalities

### **RQ3: Sentiment Mechanisms** (NICE TO HAVE - Week 2)
"What attention patterns distinguish positive from negative sentiment predictions?"
- **Hypothesis**: Positive predictions attend to different word types than negative
- **Method**: Compare attention distributions for positive vs negative samples
- **Metric**: Attention divergence between sentiment classes
- **Expected Result**: "Happy" predictions attend to adjectives, "sad" to emotion verbs

### **RQ4: Explanation Quality** (EVALUATION - Week 4)
"Can non-experts understand attention-based explanations?"
- **Hypothesis**: Attention heatmaps + natural language narrative are interpretable
- **Method**: Show 5 people attention visualizations, ask comprehension questions
- **Metric**: % correct answers, subjective usefulness rating
- **Expected Result**: ≥70% comprehension, ≥4/5 usefulness score

### **RQ5: Comparison to Tree Models** (DISCUSSION - Week 4)
"How do attention-based explanations compare to decision path explanations (XGBoost)?"
- **Hypothesis**: Attention is more fine-grained but harder to understand
- **Method**: Qualitative comparison of explanation formats
- **Metric**: Narrative length, number of components mentioned
- **Expected Result**: Attention explanations mention more interactions but require more expertise

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

## 🚀 Getting Started (Next 2 Hours)

### **Step 1: Verify Setup** ✅
```bash
# Check data exists (should show train.pkl, val.pkl, test.pkl)
ls data/cmu_mosei/

# Check model infrastructure exists
ls backend/src/models/multimodal_model.py
```

### **Step 2: Train Baseline Model** (30 min)
```bash
# Train simple multimodal model on synthetic data
python backend/src/models/train_multimodal.py \
    --data-dir data/cmu_mosei \
    --epochs 5 \
    --batch-size 8 \
    --lr 0.001

# Expected output: Model saved to .resources/models/multimodal/sentiment_model.pt
```

### **Step 3: Extract Attention (First Sample)** (30 min)
```python
# Create notebook: notebooks/01_attention_extraction.ipynb

from backend.src.models.multimodal_model import SimpleMultimodalModel
import torch

# Load model
model = SimpleMultimodalModel()
model.load_state_dict(torch.load('.resources/models/multimodal/sentiment_model.pt'))

# Load 1 sample
# ... (get text, audio, video)

# Extract attention
model.bert.config.output_attentions = True
outputs = model.bert(text_tokens)
attention = outputs.attentions  # (12 layers, batch=1, 12 heads, seq, seq)

# Visualize layer 0, head 0
import seaborn as sns
sns.heatmap(attention[0][0, 0, :, :].detach().numpy())
```

### **Step 4: Create Heatmap** (30 min)
```python
# Visualize attention for one head
import plotly.express as px

attn_matrix = attention[6][0, 3, :, :].detach().numpy()  # Layer 6, Head 3
tokens = ["I", "love", "this", "!"]  # Example tokens

fig = px.imshow(attn_matrix,
                labels=dict(x="Attended Token", y="Attending Token"),
                x=tokens, y=tokens,
                title="Attention Pattern: Layer 6, Head 3")
fig.show()
```

### **Step 5: Measure Head Importance** (30 min)
```python
# Simple ablation: Zero out each head, measure prediction change
baseline_pred = model(text, audio, video)

for layer in range(12):
    for head in range(12):
        # Zero out this head
        model.bert.encoder.layer[layer].attention.self.query.weight[:, head*64:(head+1)*64] = 0

        # Measure prediction change
        ablated_pred = model(text, audio, video)
        importance = abs(baseline_pred - ablated_pred).item()

        print(f"Layer {layer}, Head {head}: Importance = {importance:.4f}")

        # Restore head
        # ... (reload model)
```

### **Checklist for End of Day 1:**
- [ ] Model trains without errors
- [ ] Can extract attention weights as tensor
- [ ] Have 1 heatmap visualization
- [ ] Calculated importance score for ≥1 head
- [ ] Created Jupyter notebook with above code

**Tomorrow**: Automate this for all 144 heads, rank them, categorize by function

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
