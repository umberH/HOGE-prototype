"""
Quick Attention Extraction Test
================================
Demonstrates attention extraction and visualization without Jupyter.

Run: python .development/scripts/test_attention_extraction.py
"""

import sys
from pathlib import Path
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*60)
print("ATTENTION EXTRACTION DEMO")
print("="*60)

# Step 1: Load BERT
print("\n[1/5] Loading pre-trained BERT...")
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', attn_implementation="eager")
    model.eval()
    print("[OK] BERT loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading BERT: {e}")
    print("\nThis might be due to:")
    print("  - No internet connection (BERT needs to download ~400MB)")
    print("  - Insufficient memory")
    print("\nTry running: pip install transformers torch")
    sys.exit(1)

# Step 2: Encode Text
print("\n[2/5] Encoding sample text...")
text = "I love this movie! It's amazing and wonderful."
inputs = tokenizer(text, return_tensors='pt', padding=True)
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

print(f"   Text: {text}")
print(f"   Tokens: {tokens}")
print(f"   Sequence length: {len(tokens)}")

# Step 3: Extract Attention
print("\n[3/5] Extracting attention from all layers...")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attentions = outputs.attentions
num_layers = len(attentions)
num_heads = attentions[0].shape[1]

print(f"[OK] Extracted attention from {num_layers} layers × {num_heads} heads = {num_layers * num_heads} total heads")
print(f"   Attention shape per layer: {attentions[0].shape}")

# Step 4: Analyze Head Importance
print("\n[4/5] Ranking attention heads by importance...")

def calculate_attention_entropy(attention_matrix):
    """Lower entropy = more focused = more important"""
    probs = attention_matrix + 1e-10
    entropy = -np.sum(probs * np.log(probs), axis=-1).mean()
    return entropy

head_importance = []
for layer_idx in range(num_layers):
    for head_idx in range(num_heads):
        attn = attentions[layer_idx][0, head_idx].numpy()
        entropy = calculate_attention_entropy(attn)
        head_importance.append({
            'layer': layer_idx,
            'head': head_idx,
            'entropy': entropy
        })

# Sort by focus (lower entropy = more focused)
head_importance_sorted = sorted(head_importance, key=lambda x: x['entropy'])

print("\n[OK] Top 10 Most Focused Attention Heads:")
for i, head in enumerate(head_importance_sorted[:10]):
    print(f"   {i+1}. Layer {head['layer']:2d}, Head {head['head']:2d}: Entropy = {head['entropy']:.3f}")

# Step 5: Find Sentiment-Focused Heads
print("\n[5/5] Analyzing attention to sentiment words...")

sentiment_words = ['love', 'amazing', 'wonderful']
sentiment_indices = [i for i, token in enumerate(tokens) if token in sentiment_words]

print(f"   Sentiment words at indices: {sentiment_indices}")
print(f"   Tokens: {[tokens[i] for i in sentiment_indices]}")

sentiment_attention = []
for layer_idx in range(num_layers):
    for head_idx in range(num_heads):
        attn = attentions[layer_idx][0, head_idx].numpy()
        attn_to_sentiment = attn[:, sentiment_indices].mean()

        sentiment_attention.append({
            'layer': layer_idx,
            'head': head_idx,
            'sentiment_focus': attn_to_sentiment
        })

sentiment_sorted = sorted(sentiment_attention, key=lambda x: x['sentiment_focus'], reverse=True)

print("\n[OK] Top 10 Heads Focusing on Sentiment Words:")
for i, head in enumerate(sentiment_sorted[:10]):
    print(f"   {i+1}. Layer {head['layer']:2d}, Head {head['head']:2d}: Focus = {head['sentiment_focus']:.3f}")

# Step 6: Visualize One Head
print("\n[6/6] Creating attention heatmap...")
print("   (Saving to: figures/attention_demo.png)")

# Create figure directory
fig_dir = Path(".resources/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# Plot the most sentiment-focused head
best_head = sentiment_sorted[0]
layer_idx = best_head['layer']
head_idx = best_head['head']

attention_matrix = attentions[layer_idx][0, head_idx].numpy()

plt.figure(figsize=(10, 8))
sns.heatmap(
    attention_matrix,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap='Blues',
    cbar=True,
    square=True,
    linewidths=0.5
)

plt.title(f"Attention Pattern: Layer {layer_idx}, Head {head_idx}\n(Most focused on sentiment words)")
plt.xlabel("Key (Attended Token)")
plt.ylabel("Query (Attending Token)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

output_path = fig_dir / "attention_demo.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[OK] Saved heatmap to: {output_path}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"[OK] Analyzed {num_layers * num_heads} attention heads")
print(f"[OK] Most focused head: Layer {head_importance_sorted[0]['layer']}, Head {head_importance_sorted[0]['head']}")
print(f"[OK] Most sentiment-focused: Layer {sentiment_sorted[0]['layer']}, Head {sentiment_sorted[0]['head']}")
print(f"[OK] Visualization saved to: {output_path}")
print("\nNext steps:")
print("  1. Open the heatmap image to see attention patterns")
print("  2. Try different input texts to see how attention changes")
print("  3. Build AttentionMechanisticInterpreter class based on this analysis")
print("="*60)
