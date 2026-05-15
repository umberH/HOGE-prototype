"""
Load Multimodal Attention Analysis into Neo4j
==============================================
Combines attention circuit extraction with Neo4j loading.

Usage:
    python .development/scripts/load_attention_to_neo4j.py
"""

import sys
from pathlib import Path
import pickle
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.src.knowledge_graph.neo_attention_loader import AttentionCircuitLoader

print("="*80)
print("LOAD ATTENTION ANALYSIS INTO NEO4J")
print("="*80)

# Step 1: Load synthetic data
print("\n[1/4] Loading synthetic MOSEI data...")
with open('data/cmu_mosei/train.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"[OK] Loaded {len(data)} samples")

# Select sample
sample_idx = 0
sample = data[sample_idx]
print(f"\nAnalyzing sample {sample_idx}: '{sample['text']}'")

# Step 2: Extract attention
print("\n[2/4] Extracting attention patterns...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', attn_implementation="eager")
model.eval()

# Encode text
inputs = tokenizer(sample['text'], return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Extract BERT attention
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attentions = outputs.attentions

print(f"[OK] Extracted attention from {len(attentions)} layers, {attentions[0].shape[1]} heads")

# Step 3: Analyze attention patterns
print("\n[3/4] Analyzing attention heads...")

def calculate_entropy(attention_matrix):
    probs = attention_matrix + 1e-10
    entropy = -np.sum(probs * np.log(probs), axis=-1).mean()
    return entropy

head_analysis = []
for layer_idx in range(len(attentions)):
    for head_idx in range(12):
        attn = attentions[layer_idx][0, head_idx].numpy()
        entropy = calculate_entropy(attn)

        avg_attn_per_token = attn.mean(axis=0)
        max_token_idx = avg_attn_per_token.argmax()

        head_analysis.append({
            'layer': layer_idx,
            'head': head_idx,
            'entropy': float(entropy),
            'focus_token': tokens[max_token_idx],
            'focus_strength': float(avg_attn_per_token[max_token_idx])
        })

# Find circuits
sentiment_words = ['love', 'amazing', 'wonderful', 'pathetic', 'terrible', 'awful', 'bad', 'horrible']
sentiment_indices = [i for i, tok in enumerate(tokens) if tok in sentiment_words]

circuits = []
if sentiment_indices:
    sentiment_token = tokens[sentiment_indices[0]]
    sentiment_idx = sentiment_indices[0]

    circuit_heads = []
    for layer_idx in range(len(attentions)):
        for head_idx in range(12):
            attn = attentions[layer_idx][0, head_idx].numpy()
            attn_to_sentiment = attn[:, sentiment_idx].mean()

            if attn_to_sentiment > 0.1:  # Threshold
                circuit_heads.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'attention_weight': float(attn_to_sentiment)
                })

    # Sort by attention weight
    circuit_heads_sorted = sorted(circuit_heads, key=lambda x: x['attention_weight'], reverse=True)

    # Take top 5 for circuit
    circuit = {
        'circuit_type': f"sentiment_{sample['sentiment_label']}",
        'input_token': sentiment_token,
        'heads': circuit_heads_sorted[:5],  # Top 5 heads
        'prediction': sample['sentiment_label'],
        'confidence': abs(float(sample['sentiment_score']))
    }
    circuits.append(circuit)

    print(f"[OK] Discovered {len(circuits)} circuits with {len(circuit_heads)} total heads")
else:
    print("[WARN] No sentiment words found, creating empty circuit list")

# Simulated cross-modal similarity (would be real with trained model)
cross_modal_sim = {
    'text_audio': -0.016,
    'text_video': 0.062,
    'audio_video': -0.027
}

# Step 4: Load into Neo4j
print("\n[4/4] Loading into Neo4j...")

try:
    # Use separate database if local, or default with prefix if Aura
    import os
    use_remote = os.getenv("USE_REMOTE_NEO4J", "local").lower() == "remote"

    if use_remote:
        loader = AttentionCircuitLoader()  # Uses default 'neo4j' database
        print("[INFO] Using Neo4j Aura - default database")
    else:
        loader = AttentionCircuitLoader(database="attention")  # Separate database
        print("[INFO] Using separate 'attention' database")

    # Create constraints first
    loader.create_constraints()

    # Prepare data
    utterance_data = {
        'utterance_id': sample['utterance_id'],
        'text': sample['text'],
        'sentiment_score': float(sample['sentiment_score']),
        'sentiment_label': sample['sentiment_label']
    }

    # Load complete analysis
    result = loader.load_complete_analysis(
        utterance_data=utterance_data,
        tokens=tokens,
        head_analysis=head_analysis,
        circuits=circuits,
        cross_modal_sim=cross_modal_sim
    )

    print("\n" + "="*80)
    print("SUCCESS! ATTENTION ANALYSIS LOADED INTO NEO4J")
    print("="*80)
    print(f"\nUtterance: \"{utterance_data['text']}\"")
    print(f"Utterance ID: {result['utterance_id']}")
    print(f"Tokens: {result['num_tokens']}")
    print(f"Attention Heads: {result['num_heads']}")
    print(f"Circuits: {result['num_circuits']}")

    if result['circuit_ids']:
        print(f"\nCircuit IDs:")
        for cid in result['circuit_ids']:
            print(f"  - {cid}")

    # Query back and generate explanation
    print("\n" + "="*80)
    print("GENERATING EXPLANATION FROM KNOWLEDGE GRAPH")
    print("="*80)
    explanation = loader.generate_circuit_explanation(result['utterance_id'])
    print(explanation)

    # Example Cypher queries
    print("\n" + "="*80)
    print("EXAMPLE NEO4J QUERIES")
    print("="*80)
    print("\n1. View all utterances:")
    print("   MATCH (u:Utterance) RETURN u LIMIT 10")

    print("\n2. Find most important attention heads:")
    print("   MATCH (h:AttentionHead)")
    print("   RETURN h.layer, h.head, h.entropy")
    print("   ORDER BY h.entropy ASC LIMIT 10")

    print("\n3. View sentiment circuit:")
    print(f"   MATCH (c:AttentionCircuit {{circuit_id: '{result['circuit_ids'][0]}'}})")
    print("   MATCH (c)-[:USES_HEAD]->(h:AttentionHead)")
    print("   RETURN c, h")

    print("\n4. Find all circuits for an utterance:")
    print(f"   MATCH (u:Utterance {{utterance_id: '{result['utterance_id']}'}})")
    print("   MATCH (u)-[:HAS_CIRCUIT]->(c:AttentionCircuit)")
    print("   RETURN u.text, c.circuit_type, c.num_heads")

    loader.close()

    print("\n" + "="*80)
    print("You can now explore the knowledge graph in Neo4j Browser!")
    if not use_remote:
        print("Open: http://localhost:7474")
        print("Database: attention")
    else:
        print(f"Open your Neo4j Aura console")
    print("="*80)

except Exception as e:
    print(f"\n[ERROR] {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure Neo4j is running")
    print("  2. Run setup script first: python .development/scripts/setup_attention_neo4j.py")
    print("  3. Check .env has correct Neo4j credentials")
    import traceback
    traceback.print_exc()
