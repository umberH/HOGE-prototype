"""
Attention Circuit Discovery & Analysis
=======================================
Novel research contributions for PhD:
1. LLM-based narrative generation (RC2)
2. Circuit discovery algorithms (inspired by Anthropic's work)
3. Induction head detection
4. Neo4j ontology integration

References:
- Olsson et al. (2022): Induction Heads in GPT-2
- Wang et al. (2023): Indirect Object Identification Circuit
"""

import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

# Initialize OpenAI client for narrative generation
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# RC2: LLM-BASED NARRATIVE GENERATION
# ============================================================================

def generate_attention_narrative(
    tokens: List[str],
    attention_patterns: Dict,
    prediction: Optional[str] = None,
    audience: str = "non-technical"
) -> str:
    """
    Novel Contribution: Use LLM to translate attention patterns into human narratives.

    This addresses RC2 in ATTENTION_RESEARCH_CONTRIBUTIONS.md:
    - Technical audience: Detailed head numbers, entropy values
    - Non-technical audience: Simple explanations "The model noticed..."

    Args:
        tokens: List of input tokens
        attention_patterns: Dict with 'top_heads', 'entropy', 'focus_words'
        prediction: Model's prediction (e.g., "positive sentiment")
        audience: "technical" or "non-technical"

    Returns:
        Human-readable explanation of attention patterns
    """

    # Extract key information
    top_heads = attention_patterns.get('top_heads', [])
    focus_words = attention_patterns.get('focus_words', [])

    # Build prompt based on audience
    if audience == "technical":
        prompt = f"""
You are explaining transformer attention patterns to an AI researcher.

Input text: "{' '.join(tokens)}"
Top attention heads (by focus):
{chr(10).join([f"- Layer {h['layer']}, Head {h['head']}: Entropy={h['entropy']:.3f}, Focus={h['focus']:.3f}" for h in top_heads[:5]])}

Most attended words: {', '.join([f"'{w['word']}' ({w['attention']:.2f})" for w in focus_words[:3]])}
Prediction: {prediction or 'N/A'}

Provide a technical explanation (2-3 sentences) of:
1. Which heads are most important and why
2. What linguistic patterns they capture
3. How this relates to the prediction

Use technical terminology (entropy, attention weights, layer composition).
"""
    else:  # non-technical
        prompt = f"""
You are explaining how an AI model processes text to a non-technical user.

Input text: "{' '.join(tokens)}"
The AI model paid most attention to these words: {', '.join([w['word'] for w in focus_words[:3]])}
The model's prediction: {prediction or 'analyzing the text'}

Explain in 1-2 simple sentences:
1. What the model noticed in the text
2. Why that led to the prediction

Use everyday language. Start with "The model noticed..." or "The AI focused on..."
Avoid technical jargon like "layer", "head", "entropy".
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": "You are an expert at explaining AI model behavior clearly and accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )

        narrative = response.choices[0].message.content.strip()
        return narrative

    except Exception as e:
        # Fallback if API fails
        if audience == "technical":
            head_names = ', '.join([f"L{h['layer']}H{h['head']}" for h in top_heads[:3]])
            word_names = ', '.join([w['word'] for w in focus_words[:3]])
            return f"Top attention heads: {head_names}. Focus on: {word_names}."
        else:
            word_names = ', '.join([w['word'] for w in focus_words[:3]])
            return f"The model focused on: {word_names}."


# ============================================================================
# MECHANISTIC CIRCUITS: INDUCTION HEAD DETECTION
# ============================================================================

def detect_induction_heads(attentions: List, tokens: List[str]) -> List[Dict]:
    """
    Detect induction heads (Olsson et al., 2022).

    Induction heads perform: [A][B] ... [A] -> predict [B]
    They copy patterns from earlier in the sequence.

    Novel contribution: Extend to multimodal transformers.

    Args:
        attentions: List of attention tensors (num_layers, batch, num_heads, seq, seq)
        tokens: List of input tokens

    Returns:
        List of detected induction heads with scores
    """

    induction_heads = []
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    seq_len = len(tokens)

    # Look for heads that attend to previous tokens with same prefix
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            attn = attentions[layer_idx][0, head_idx].cpu().numpy()

            # Induction score: Does position i attend to position j where token[j-1] == token[i-1]?
            induction_score = 0.0
            count = 0

            for i in range(2, seq_len):
                for j in range(1, i-1):
                    # Check if previous tokens match
                    if tokens[i-1] == tokens[j-1]:
                        # Measure attention from i to j
                        induction_score += attn[i, j]
                        count += 1

            if count > 0:
                avg_induction_score = induction_score / count

                if avg_induction_score > 0.1:  # Threshold
                    induction_heads.append({
                        'layer': layer_idx,
                        'head': head_idx,
                        'induction_score': float(avg_induction_score),
                        'type': 'induction_head'
                    })

    # Sort by induction score
    induction_heads.sort(key=lambda x: x['induction_score'], reverse=True)
    return induction_heads


# ============================================================================
# CIRCUIT DISCOVERY: ATTENTION FLOW PATHS
# ============================================================================

def discover_attention_circuits(
    attentions: List,
    tokens: List[str],
    target_token_idx: int,
    min_attention: float = 0.1,
    max_depth: int = 3
) -> List[Dict]:
    """
    Discover attention circuits: paths from input token -> heads -> prediction.

    Inspired by Wang et al. (2023) - Indirect Object Identification circuit.

    Novel contribution: Apply to sentiment/multimodal analysis.

    Args:
        attentions: Attention tensors
        tokens: Input tokens
        target_token_idx: Token to trace circuits from (e.g., sentiment word)
        min_attention: Minimum attention weight to include in circuit
        max_depth: Maximum path depth through layers

    Returns:
        List of discovered circuits with attention flow
    """

    circuits = []
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    # Find paths from target token through layers
    def trace_path(start_token, current_layer, path, total_attention):
        if current_layer >= min(num_layers, max_depth):
            # Reached end of path
            if total_attention > min_attention * max_depth:
                circuits.append({
                    'input_token': tokens[start_token],
                    'path': path.copy(),
                    'total_attention': total_attention,
                    'depth': len(path)
                })
            return

        # Check all heads in current layer
        attn = attentions[current_layer][0].cpu().numpy()

        for head_idx in range(num_heads):
            # Attention from all tokens to target token
            attn_to_target = attn[head_idx, :, start_token].max()

            if attn_to_target > min_attention:
                path.append({
                    'layer': current_layer,
                    'head': head_idx,
                    'attention': float(attn_to_target)
                })

                # Continue to next layer
                trace_path(start_token, current_layer + 1, path, total_attention + attn_to_target)

                path.pop()  # Backtrack

    # Start tracing from target token
    trace_path(target_token_idx, 0, [], 0.0)

    # Sort by total attention
    circuits.sort(key=lambda x: x['total_attention'], reverse=True)
    return circuits[:10]  # Return top 10 circuits


# ============================================================================
# CROSS-MODAL ATTENTION ANALYSIS
# ============================================================================

def analyze_cross_modal_attention(
    text_tokens: List[str],
    text_attention: np.ndarray,
    audio_features: Optional[np.ndarray] = None,
    video_features: Optional[np.ndarray] = None
) -> Dict:
    """
    Analyze attention between modalities (text <-> audio <-> video).

    Novel contribution: First cross-modal attention circuit analysis.

    Args:
        text_tokens: Text tokens
        text_attention: Text attention patterns
        audio_features: Audio feature embeddings (if available)
        video_features: Video feature embeddings (if available)

    Returns:
        Cross-modal attention analysis
    """

    analysis = {
        'modality': 'text-only',
        'cross_modal_heads': [],
        'fusion_score': 0.0
    }

    if audio_features is not None or video_features is not None:
        analysis['modality'] = 'multimodal'

        # TODO: Implement cross-modal attention extraction
        # This requires a multimodal model with cross-attention layers
        # For now, return placeholder
        analysis['note'] = "Cross-modal attention requires multimodal transformer model"

    return analysis


# ============================================================================
# NEO4J ONTOLOGY INTEGRATION
# ============================================================================

def store_attention_in_knowledge_graph(
    circuit_data: Dict,
    loader=None
) -> Dict:
    """
    Connect attention patterns to semantic ontology in Neo4j.

    Novel contribution: Ontology-grounded mechanistic interpretability.

    Links:
    - AttentionHead -> Token -> SentimentConcept
    - AttentionCircuit -> PredictionExplanation

    Args:
        circuit_data: Discovered circuit information
        loader: AttentionCircuitLoader instance (from neo_attention_loader.py)

    Returns:
        Storage result with graph IDs
    """

    if loader is None:
        # Try to import loader
        try:
            from backend.src.knowledge_graph.neo_attention_loader import AttentionCircuitLoader
            loader = AttentionCircuitLoader()
        except ImportError:
            return {'status': 'error', 'message': 'Neo4j loader not available'}

    # Store circuit
    result = loader.load_attention_circuit(
        utterance_id=circuit_data.get('utterance_id'),
        circuit_data=circuit_data
    )

    return result


# ============================================================================
# HEAD TAXONOMY (RC4)
# ============================================================================

def classify_attention_head_type(
    layer: int,
    head: int,
    attention: np.ndarray,
    tokens: List[str]
) -> str:
    """
    Classify attention head into interpretable categories.

    Categories:
    - Syntax heads (attend to syntactic patterns)
    - Sentiment heads (attend to emotional words)
    - Positional heads (attend based on position)
    - Induction heads (copy patterns)
    - Broadcast heads (attend uniformly)

    Novel contribution: Taxonomy for multimodal transformers.

    Args:
        layer, head: Head identifier
        attention: Attention matrix (seq_len, seq_len)
        tokens: Input tokens

    Returns:
        Head type label
    """

    seq_len = attention.shape[0]

    # Calculate attention statistics
    entropy = -np.sum(attention * np.log(attention + 1e-10), axis=-1).mean()
    diagonal_attention = np.mean([attention[i, i] for i in range(seq_len)])
    next_token_attention = np.mean([attention[i, min(i+1, seq_len-1)] for i in range(seq_len-1)])

    # Classify
    if entropy > 2.0:
        return "broadcast_head"  # Attends uniformly
    elif diagonal_attention > 0.5:
        return "positional_head"  # Self-attention
    elif next_token_attention > 0.3:
        return "syntax_head"  # Sequential processing
    else:
        # Check for sentiment words
        sentiment_words = ["love", "hate", "amazing", "terrible", "pathetic", "wonderful"]
        sentiment_attention = 0.0
        for i, token in enumerate(tokens):
            if any(sw in token.lower() for sw in sentiment_words):
                sentiment_attention += attention[:, i].mean()

        if sentiment_attention > 0.2:
            return "sentiment_head"
        else:
            return "specialized_head"  # Task-specific but unclear


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_attention_mechanisms(
    tokens: List[str],
    attentions: List,
    prediction: Optional[str] = None,
    audience: str = "non-technical",
    enable_llm_narrative: bool = True,
    enable_circuit_discovery: bool = True
) -> Dict:
    """
    Complete attention analysis pipeline combining all novel contributions.

    Returns comprehensive analysis including:
    - LLM-generated narrative (RC2)
    - Induction head detection
    - Attention circuits
    - Head taxonomy
    - Neo4j storage ready data

    Args:
        tokens: Input tokens
        attentions: Attention tensors from model
        prediction: Model prediction
        audience: "technical" or "non-technical"
        enable_llm_narrative: Use GPT-4 for explanations
        enable_circuit_discovery: Run circuit discovery

    Returns:
        Comprehensive attention analysis
    """

    import torch

    # Convert to numpy if needed
    if isinstance(attentions[0], torch.Tensor):
        attentions = [a.cpu() for a in attentions]

    # Basic analysis
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    # Calculate head importance
    head_importance = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            attn = attentions[layer_idx][0, head_idx].numpy()
            entropy = -np.sum(attn * np.log(attn + 1e-10), axis=-1).mean()

            # Classify head type
            head_type = classify_attention_head_type(layer_idx, head_idx, attn, tokens)

            head_importance.append({
                'layer': layer_idx,
                'head': head_idx,
                'entropy': float(entropy),
                'focus': float(1 / (entropy + 1)),
                'type': head_type
            })

    # Sort by focus
    head_importance.sort(key=lambda x: x['focus'], reverse=True)
    top_heads = head_importance[:10]

    # Find most attended words
    focus_words = []
    for i, token in enumerate(tokens):
        if token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
            total_attention = sum([
                attentions[l][0, :, :, i].mean().item()
                for l in range(num_layers)
            ])
            focus_words.append({
                'word': token,
                'attention': float(total_attention),
                'position': i
            })

    focus_words.sort(key=lambda x: x['attention'], reverse=True)

    # LLM narrative generation (RC2)
    narrative = None
    if enable_llm_narrative and os.getenv("OPENAI_API_KEY"):
        narrative = generate_attention_narrative(
            tokens=tokens,
            attention_patterns={'top_heads': top_heads, 'focus_words': focus_words},
            prediction=prediction,
            audience=audience
        )

    # Induction head detection
    induction_heads = detect_induction_heads(attentions, tokens)

    # Circuit discovery (if enabled and focus word exists)
    circuits = []
    if enable_circuit_discovery and focus_words:
        target_idx = focus_words[0]['position']
        circuits = discover_attention_circuits(attentions, tokens, target_idx)

    return {
        'tokens': tokens,
        'top_heads': top_heads,
        'focus_words': focus_words[:5],
        'narrative': narrative,
        'audience': audience,
        'induction_heads': induction_heads[:5],
        'circuits': circuits[:5],
        'prediction': prediction,
        'model_architecture': {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'total_heads': num_layers * num_heads
        }
    }
