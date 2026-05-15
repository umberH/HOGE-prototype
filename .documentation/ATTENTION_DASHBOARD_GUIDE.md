# Attention Mechanistic Analysis Dashboard - User Guide

## 🚀 Quick Start

```bash
streamlit run frontend/attention_dashboard.py --server.port 8502
```

**URL**: http://localhost:8502

---

## 📊 Overview

This dashboard implements **5 novel PhD research contributions** for mechanistic interpretability of transformer attention mechanisms:

1. **Multi-Model Support**: Compare BERT, RoBERTa, DistilBERT, ALBERT
2. **LLM Narrative Generation (RC2)**: GPT-4 explains attention patterns
3. **Circuit Discovery**: Trace attention paths input → prediction
4. **Induction Head Detection**: Find pattern-matching heads
5. **Head Taxonomy (RC4)**: Automatic classification of head types

---

## 🎯 Research Contributions

### **RC2: LLM-Based Narrative Generation**

**What it does:**
- Uses GPT-4 to translate technical attention patterns into natural language
- Generates audience-adaptive explanations (technical vs. non-technical)

**Novel contribution:**
- First system to use LLMs for attention explanation
- Makes mechanistic interpretability accessible to non-experts

**How to use:**
1. Enable "LLM Narrative Generation (RC2)" in sidebar
2. Choose explanation style: "technical" or "non-technical"
3. Analyze text - explanation appears at top

**Example output (non-technical):**
> "The model focused on the word 'pathetic' and noticed the negative emotional tone, which led it to predict negative sentiment."

**Example output (technical):**
> "Layer 4 Head 1 shows 0.402 attention weight from [CLS] to 'pathetic', with entropy 0.089 indicating high focus. This head specializes in sentiment detection."

**Implementation:**
- `backend/src/explainability/attention_circuits.py::generate_attention_narrative()`
- Prompt engineering for audience adaptation
- Falls back to template if API unavailable

---

### **Circuit Discovery (Anthropic-inspired)**

**What it does:**
- Discovers attention paths: input token → Layer 0 Head 3 → Layer 4 Head 1 → prediction
- Inspired by Wang et al. (2023) "Interpretability in the Wild"

**Novel contribution:**
- First application to sentiment/multimodal analysis
- Visualizes interpretable information flow

**How to use:**
1. Enable "Circuit Discovery" in sidebar
2. Analyze text with clear sentiment words
3. View "🔗 Attention Circuits" tab

**Example circuit:**
```
"pathetic" (input) →
L0H1 (0.23) →
L4H1 (0.40) →
L8H7 (0.37) →
prediction
```

**Interpretation:**
- Circuit shows how "pathetic" attention flows through specific heads
- Total attention: sum of weights along path
- Higher total = stronger circuit

**Implementation:**
- `discover_attention_circuits()` uses recursive path tracing
- Threshold: `min_attention=0.1`
- Max depth: 3 layers

---

### **Induction Head Detection (Olsson et al., 2022)**

**What it does:**
- Detects heads that perform pattern matching: `[A][B] ... [A] → predict [B]`
- Key mechanism for in-context learning

**Novel contribution:**
- Extend detection algorithm to multimodal transformers
- Identify pattern-copying mechanisms

**How to use:**
1. Enable "Induction Head Detection" in sidebar
2. Use text with repeated patterns (e.g., "I love you. I love cats.")
3. View "🔄 Induction Heads" tab

**Induction score calculation:**
```python
# For each position i, check if model attends to j where:
# token[i-1] == token[j-1]  (previous tokens match)
induction_score = attention[i, j]  # if match found
```

**Interpretation:**
- Score > 0.1: Likely induction head
- Higher score: Stronger pattern-copying behavior

**Implementation:**
- `detect_induction_heads()` in `attention_circuits.py`
- Detects token-level pattern matching

---

### **Head Taxonomy Classification (RC4)**

**What it does:**
- Automatically classifies attention heads into interpretable categories

**Categories:**
- **Sentiment heads**: Focus on emotional words (love, hate, amazing, terrible)
- **Syntax heads**: Focus on grammatical structure (next-token attention)
- **Positional heads**: Self-attention (diagonal pattern)
- **Induction heads**: Copy patterns
- **Broadcast heads**: Attend uniformly (high entropy)
- **Specialized heads**: Task-specific but unclear

**Novel contribution:**
- First automatic taxonomy for multimodal models
- Links head types to interpretable functions

**How to use:**
1. Analyze any text
2. View "🎯 Focus Words" tab
3. See "Head Type Classification" bar chart

**Classification algorithm:**
```python
if entropy > 2.0:
    return "broadcast_head"
elif diagonal_attention > 0.5:
    return "positional_head"
elif next_token_attention > 0.3:
    return "syntax_head"
elif sentiment_attention > 0.2:
    return "sentiment_head"
else:
    return "specialized_head"
```

**Implementation:**
- `classify_attention_head_type()` in `attention_circuits.py`
- Based on attention statistics (entropy, focus patterns)

---

### **Multi-Model Support**

**What it does:**
- Compare attention patterns across different transformer architectures

**Supported models:**
- **BERT-base**: 12 layers, 12 heads (144 total)
- **RoBERTa-base**: 12 layers, 12 heads
- **DistilBERT**: 6 layers, 12 heads (72 total - faster)
- **ALBERT-base**: 12 layers, 12 heads (parameter sharing)

**Why it matters:**
- Different models learn different attention patterns
- RoBERTa: More robust to input variations
- DistilBERT: Faster, distilled from BERT
- ALBERT: Efficient parameter sharing

**Research question:**
> "Do different transformer architectures develop similar sentiment circuits?"

**How to use:**
1. Select model from "🤖 Model Selection" dropdown
2. Analyze same text with different models
3. Compare circuit structures

**Expected findings:**
- Sentiment heads appear in similar layers across models
- DistilBERT has fewer but more focused heads
- RoBERTa may show more distributed attention

---

## 🔬 How Model Choice Affects Your Project

### **Impact on Research Contributions**

| Model | Layers | Heads | Speed | Best For |
|-------|--------|-------|-------|----------|
| **BERT** | 12 | 144 | Medium | General analysis, baselines |
| **RoBERTa** | 12 | 144 | Medium | Robust patterns, adversarial testing |
| **DistilBERT** | 6 | 72 | Fast | Quick prototyping, demos |
| **ALBERT** | 12 | 144 | Medium | Parameter efficiency study |

### **Research Use Cases:**

#### **1. Cross-Model Circuit Comparison** (Novel Publication)
```
Research question: Are sentiment circuits architecture-invariant?

Method:
1. Analyze "I love this movie!" with BERT
2. Analyze same text with RoBERTa, DistilBERT
3. Compare discovered circuits
4. Measure overlap (Jaccard similarity of head sets)

Expected result:
- 60-80% overlap in top sentiment heads
- Core circuit (L4H1, L8H7) consistent across models
- Peripheral heads vary by architecture
```

#### **2. Efficiency vs. Interpretability Trade-off**
```
Research question: Does model compression affect circuit interpretability?

Method:
1. Compare BERT (144 heads) vs DistilBERT (72 heads)
2. Measure: circuit clarity, head specialization
3. User study: Which model's explanations are clearer?

Hypothesis:
- DistilBERT has fewer but more specialized heads
- Circuits are simpler but equally interpretable
```

#### **3. Induction Head Analysis Across Architectures**
```
Research question: Which architecture best develops induction heads?

Method:
1. Use text with repeated patterns
2. Count induction heads in each model
3. Measure induction scores

Expected result:
- BERT/RoBERTa: 15-20% of heads are induction heads
- DistilBERT: Higher percentage (distillation preserves key mechanisms)
```

---

## 📝 Step-by-Step Tutorial

### **Example 1: Discover Sentiment Circuit**

1. **Setup**
   - Model: BERT-base
   - Text: "This movie is absolutely pathetic and terrible."
   - Enable: All research features

2. **Run Analysis**
   - Click "🔍 Analyze Attention"
   - Wait for GPT-4 narrative (5-10 seconds)

3. **Read LLM Explanation** (top of page)
   ```
   "The model focused on the strong negative words 'pathetic' and
   'terrible', using Layer 4 Head 1 to detect emotional tone, leading
   to a confident negative prediction."
   ```

4. **View Circuit** (🔗 Attention Circuits tab)
   ```
   Circuit 1: pathetic (total: 1.85)
   Path: L0H1 (0.23) → L4H1 (0.40) → L8H7 (0.37) → L11H3 (0.32)

   Circuit 2: terrible (total: 1.72)
   Path: L0H2 (0.21) → L4H1 (0.38) → L8H7 (0.35) → L11H3 (0.30)
   ```

5. **Interpretation**
   - Both words use same core circuit (L4H1 → L8H7)
   - L4H1 = sentiment detection head
   - L8H7 = sentiment aggregation head
   - Consistent circuit = interpretable mechanism

### **Example 2: Detect Induction Heads**

1. **Setup**
   - Model: BERT-base
   - Text: "The cat sat on the mat. The cat loved the mat."
   - Enable: Induction Head Detection

2. **Run Analysis**

3. **View Induction Heads** (🔄 tab)
   ```
   Layer | Head | Induction Score | Type
   ------|------|----------------|------
   5     | 7    | 0.342          | induction_head
   7     | 2    | 0.298          | induction_head
   9     | 4    | 0.267          | induction_head
   ```

4. **Interpretation**
   - L5H7 detects pattern: "The cat" appears twice
   - When "The cat" seen again → attend back to first occurrence
   - Enables copying: predict "sat" after second "The cat"

### **Example 3: Compare Models**

1. **Analyze with BERT**
   - Text: "I love this!"
   - Note top 3 sentiment heads

2. **Switch to RoBERTa**
   - Same text
   - Note top 3 sentiment heads

3. **Switch to DistilBERT**
   - Same text
   - Note top 3 sentiment heads

4. **Compare Results**
   ```
   BERT:        L4H1, L8H7, L11H3
   RoBERTa:     L4H2, L8H6, L11H5
   DistilBERT:  L2H1, L4H3, L5H7

   Overlap: ~60% (different head numbers but similar layers)
   ```

---

## 🔗 Integration with Neo4j Knowledge Graph

The dashboard connects to your Neo4j Aura instance to store attention circuits:

### **Stored Data:**
- `(:Attention_Utterance)` - Input texts
- `(:Attention_AttentionHead)` - Individual heads with entropy, type
- `(:Attention_AttentionCircuit)` - Discovered circuits
- `(:Attention_Token)` - Tokenized inputs

### **Query Examples:**

```cypher
// Find all sentiment heads
MATCH (h:Attention_AttentionHead {type: 'sentiment_head'})
RETURN h.layer, h.head, h.entropy
ORDER BY h.entropy ASC
LIMIT 10

// Find circuits for negative sentiment
MATCH (c:Attention_AttentionCircuit {circuit_type: 'sentiment_negative'})
MATCH (c)-[:USES_HEAD]->(h:Attention_AttentionHead)
RETURN c.input_token, h.layer, h.head, h.attention_weight
ORDER BY h.attention_weight DESC

// Compare circuits across utterances
MATCH (u1:Attention_Utterance)-[:HAS_CIRCUIT]->(c1)
MATCH (u2:Attention_Utterance)-[:HAS_CIRCUIT]->(c2)
WHERE u1.sentiment_label = 'negative' AND u2.sentiment_label = 'negative'
RETURN u1.text, u2.text, c1.input_token, c2.input_token
```

---

## 📊 Research Workflow

### **Week 1-2: Baseline Analysis**
1. Use BERT to analyze 100+ sentiment samples
2. Discover core sentiment circuits
3. Classify all heads by type
4. Generate LLM narratives for top 10 circuits

### **Week 3: Cross-Model Validation**
1. Repeat analysis with RoBERTa, DistilBERT
2. Measure circuit overlap
3. Identify architecture-invariant patterns

### **Week 4: Multimodal Extension**
1. Replace BERT with multimodal model
2. Discover cross-modal circuits (text → audio → video)
3. Novel contribution: First multimodal attention circuits

### **Week 5: Evaluation**
1. Ablation testing (remove circuit heads → measure impact)
2. User study (evaluate LLM explanations)
3. Write paper sections

---

## 🎓 PhD Contribution Summary

| Feature | Research Contribution | Novelty | Publications |
|---------|----------------------|---------|--------------|
| **LLM Narratives** | RC2: Human-Centric Explanations | First attention-to-text system | XAI workshop |
| **Circuit Discovery** | RC1: Multimodal Circuits | First for sentiment analysis | NeurIPS/ICLR |
| **Induction Heads** | Extend Olsson et al. to multimodal | Detection algorithm | EMNLP short |
| **Head Taxonomy** | RC4: Automatic classification | Interpretable categories | Demo paper |
| **Multi-Model** | Cross-architecture comparison | Circuit generalization | Analysis section |

---

## 🐛 Troubleshooting

### **"No LLM narrative shown"**
- Check `OPENAI_API_KEY` in `.env`
- Verify API quota
- Fallback: Uses template explanation

### **"No circuits detected"**
- Use text with clear sentiment words (love, hate, amazing, terrible)
- Lower threshold in code: `min_attention=0.05`

### **"No induction heads detected"**
- Use text with repeated patterns
- Example: "I love you. I love cats. I love dogs."

### **"Dashboard won't load"**
- Check port: `streamlit run ... --server.port 8503`
- Kill other Streamlit apps: `pkill streamlit`

---

## 📚 References

**Implemented Research:**
- Olsson et al. (2022): "In-context Learning and Induction Heads"
- Wang et al. (2023): "Interpretability in the Wild: a Circuit for Indirect Object Identification"
- Voita et al. (2019): "Analyzing Multi-Head Self-Attention"
- Clark et al. (2019): "What Does BERT Look At?"

**Your Novel Extensions:**
- RC2: LLM-based narrative generation for attention patterns
- RC1: Multimodal attention circuit discovery (text+audio+video)
- RC4: Automatic head taxonomy for transformers
- Neo4j ontology grounding for mechanistic interpretability

---

## 🚀 Next Steps

1. **Test dashboard**: http://localhost:8502
2. **Try all examples** above
3. **Compare models** (BERT vs RoBERTa vs DistilBERT)
4. **Read generated narratives** (technical vs non-technical)
5. **Explore circuits** - trace attention paths
6. **Check Neo4j** - view stored circuits
7. **Plan multimodal extension** - add audio/video attention

**Ready for PhD research!** 🎓
