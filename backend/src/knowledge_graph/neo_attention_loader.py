"""
Neo4j Attention Circuit Loader
===============================
Load multimodal attention circuits into HOGE knowledge graph.

Extends the existing Neo4j schema with attention-specific nodes and relationships.

Usage:
    from backend.src.knowledge_graph.neo_attention_loader import AttentionCircuitLoader

    loader = AttentionCircuitLoader()
    loader.load_attention_analysis(utterance_id, attention_data)
"""

import os
from pathlib import Path
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class AttentionCircuitLoader:
    """
    Loads attention circuit analysis into Neo4j knowledge graph.

    Schema Extensions:
        Nodes:
        - (:Utterance) - Multimodal input (text + audio + video)
        - (:AttentionHead) - Individual attention head
        - (:AttentionCircuit) - Discovered circuit (path through heads)
        - (:Modality) - Text, Audio, or Video modality
        - (:Token) - Text token from BERT

        Relationships:
        - (:Utterance)-[:HAS_CIRCUIT]->(:AttentionCircuit)
        - (:AttentionCircuit)-[:USES_HEAD]->(:AttentionHead)
        - (:AttentionHead)-[:ATTENDS_TO]->(:Token)
        - (:AttentionHead)-[:FOCUSES_ON]->(:Modality)
        - (:AttentionCircuit)-[:PREDICTS]->(:SentimentLabel)
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j URI (default from env: NEO4J_REMOTE_URI or NEO4J_LOCAL_URI)
            user: Neo4j user (default from env)
            password: Neo4j password (default from env)
            database: Database name (default from env)
        """
        # Try to use centralized config
        try:
            from backend.src.api.adapters.config import get_neo4j_config
            config = get_neo4j_config()
            self.uri = uri or config['uri']
            self.user = user or config['user']
            self.password = password or config['password']
            self.database = database or config['database']
        except ImportError:
            # Fallback to environment variables
            use_remote = os.getenv("USE_REMOTE_NEO4J", "local").lower() == "remote"

            if use_remote:
                self.uri = uri or os.getenv("NEO4J_REMOTE_URI")
                self.user = user or os.getenv("NEO4J_REMOTE_USER", "neo4j")
                self.password = password or os.getenv("NEO4J_REMOTE_PASSWORD")
                self.database = database or os.getenv("NEO4J_REMOTE_DATABASE", "neo4j")
            else:
                self.uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
                self.user = user or os.getenv("NEO4J_USER", "neo4j")
                self.password = password or os.getenv("NEO4J_PASSWORD")
                self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        if not self.password:
            raise ValueError("Neo4j password must be provided via environment or parameter")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

        print(f"Connected to Neo4j: {self.uri} (database: {self.database})")

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_constraints(self):
        """Create uniqueness constraints for attention nodes"""
        constraints = [
            "CREATE CONSTRAINT utterance_id IF NOT EXISTS FOR (u:Utterance) REQUIRE u.utterance_id IS UNIQUE",
            "CREATE CONSTRAINT attention_head IF NOT EXISTS FOR (h:AttentionHead) REQUIRE (h.layer, h.head) IS UNIQUE",
            "CREATE CONSTRAINT attention_circuit IF NOT EXISTS FOR (c:AttentionCircuit) REQUIRE c.circuit_id IS UNIQUE",
            "CREATE CONSTRAINT token_id IF NOT EXISTS FOR (t:Token) REQUIRE (t.utterance_id, t.position) IS UNIQUE",
        ]

        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"[OK] Created constraint: {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print(f"[SKIP] Constraint already exists")
                    else:
                        print(f"[WARN] Could not create constraint: {e}")

    def load_utterance(self, session, utterance_data: Dict[str, Any]) -> str:
        """
        Create or update Utterance node.

        Args:
            session: Neo4j session
            utterance_data: Dict with keys: utterance_id, text, sentiment_score, sentiment_label

        Returns:
            utterance_id
        """
        query = """
        MERGE (u:Utterance {utterance_id: $utterance_id})
        SET u.text = $text,
            u.sentiment_score = $sentiment_score,
            u.sentiment_label = $sentiment_label,
            u.updated_at = datetime()
        RETURN u.utterance_id AS utterance_id
        """

        result = session.run(query, {
            'utterance_id': utterance_data['utterance_id'],
            'text': utterance_data['text'],
            'sentiment_score': utterance_data['sentiment_score'],
            'sentiment_label': utterance_data['sentiment_label']
        })

        return result.single()['utterance_id']

    def load_tokens(self, session, utterance_id: str, tokens: List[str]):
        """
        Create Token nodes for each token in the utterance.

        Args:
            session: Neo4j session
            utterance_id: ID of the utterance
            tokens: List of token strings
        """
        query = """
        MATCH (u:Utterance {utterance_id: $utterance_id})
        UNWIND $tokens AS token_data
        MERGE (t:Token {utterance_id: $utterance_id, position: token_data.position})
        SET t.text = token_data.text,
            t.token_type = CASE
                WHEN token_data.text IN ['[CLS]', '[SEP]', '[PAD]'] THEN 'special'
                ELSE 'word'
            END
        MERGE (u)-[:HAS_TOKEN]->(t)
        """

        tokens_data = [
            {'position': i, 'text': token}
            for i, token in enumerate(tokens)
        ]

        session.run(query, {
            'utterance_id': utterance_id,
            'tokens': tokens_data
        })

    def load_attention_heads(self, session, head_analysis: List[Dict[str, Any]]):
        """
        Create AttentionHead nodes.

        Args:
            session: Neo4j session
            head_analysis: List of dicts with keys: layer, head, entropy, focus_token, focus_strength
        """
        query = """
        UNWIND $heads AS head_data
        MERGE (h:AttentionHead {layer: head_data.layer, head: head_data.head})
        SET h.entropy = head_data.entropy,
            h.focus_token = head_data.focus_token,
            h.focus_strength = head_data.focus_strength,
            h.importance_rank = head_data.rank
        """

        # Add rank
        sorted_heads = sorted(head_analysis, key=lambda x: x['entropy'])
        for i, head in enumerate(sorted_heads):
            head['rank'] = i + 1

        session.run(query, {'heads': head_analysis})

    def load_attention_circuit(
        self,
        session,
        utterance_id: str,
        circuit_data: Dict[str, Any]
    ) -> str:
        """
        Create AttentionCircuit node and relationships.

        Args:
            session: Neo4j session
            utterance_id: ID of the utterance
            circuit_data: Dict with keys:
                - circuit_type: e.g., "sentiment_negative"
                - input_token: Token that triggers circuit
                - heads: List of (layer, head, attention_weight)
                - prediction: Final prediction
                - confidence: Prediction confidence

        Returns:
            circuit_id
        """
        circuit_id = f"{utterance_id}_{circuit_data['circuit_type']}"

        # Create circuit node
        create_circuit_query = """
        MATCH (u:Utterance {utterance_id: $utterance_id})
        MERGE (c:AttentionCircuit {circuit_id: $circuit_id})
        SET c.circuit_type = $circuit_type,
            c.input_token = $input_token,
            c.num_heads = $num_heads,
            c.prediction = $prediction,
            c.confidence = $confidence,
            c.created_at = datetime()
        MERGE (u)-[:HAS_CIRCUIT]->(c)
        """

        session.run(create_circuit_query, {
            'utterance_id': utterance_id,
            'circuit_id': circuit_id,
            'circuit_type': circuit_data['circuit_type'],
            'input_token': circuit_data['input_token'],
            'num_heads': len(circuit_data['heads']),
            'prediction': circuit_data['prediction'],
            'confidence': circuit_data['confidence']
        })

        # Link circuit to heads
        link_heads_query = """
        MATCH (c:AttentionCircuit {circuit_id: $circuit_id})
        UNWIND $heads AS head_data
        MATCH (h:AttentionHead {layer: head_data.layer, head: head_data.head})
        MERGE (c)-[r:USES_HEAD]->(h)
        SET r.attention_weight = head_data.attention_weight,
            r.rank_in_circuit = head_data.rank
        """

        # Add circuit rank
        for i, head in enumerate(circuit_data['heads']):
            head['rank'] = i + 1

        session.run(link_heads_query, {
            'circuit_id': circuit_id,
            'heads': circuit_data['heads']
        })

        return circuit_id

    def load_cross_modal_similarity(
        self,
        session,
        utterance_id: str,
        similarities: Dict[str, float]
    ):
        """
        Store cross-modal similarity scores.

        Args:
            session: Neo4j session
            utterance_id: ID of the utterance
            similarities: Dict with keys: text_audio, text_video, audio_video
        """
        query = """
        MATCH (u:Utterance {utterance_id: $utterance_id})
        SET u.cross_modal_text_audio = $text_audio,
            u.cross_modal_text_video = $text_video,
            u.cross_modal_audio_video = $audio_video
        """

        session.run(query, {
            'utterance_id': utterance_id,
            'text_audio': similarities.get('text_audio', 0.0),
            'text_video': similarities.get('text_video', 0.0),
            'audio_video': similarities.get('audio_video', 0.0)
        })

    def load_complete_analysis(
        self,
        utterance_data: Dict[str, Any],
        tokens: List[str],
        head_analysis: List[Dict[str, Any]],
        circuits: List[Dict[str, Any]],
        cross_modal_sim: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Load complete attention analysis into Neo4j.

        Args:
            utterance_data: Dict with utterance metadata
            tokens: List of tokens
            head_analysis: List of head importance data
            circuits: List of discovered circuits
            cross_modal_sim: Cross-modal similarity scores

        Returns:
            Dict with loaded node counts
        """
        with self.driver.session(database=self.database) as session:
            # Load in transaction
            with session.begin_transaction() as tx:
                # 1. Create utterance
                utterance_id = self.load_utterance(tx, utterance_data)
                print(f"[OK] Created/updated utterance: {utterance_id}")

                # 2. Create tokens
                self.load_tokens(tx, utterance_id, tokens)
                print(f"[OK] Created {len(tokens)} tokens")

                # 3. Create attention heads
                self.load_attention_heads(tx, head_analysis)
                print(f"[OK] Created/updated {len(head_analysis)} attention heads")

                # 4. Create circuits
                circuit_ids = []
                for circuit in circuits:
                    circuit_id = self.load_attention_circuit(tx, utterance_id, circuit)
                    circuit_ids.append(circuit_id)
                print(f"[OK] Created {len(circuit_ids)} attention circuits")

                # 5. Store cross-modal similarities
                self.load_cross_modal_similarity(tx, utterance_id, cross_modal_sim)
                print(f"[OK] Stored cross-modal similarities")

                tx.commit()

        return {
            'utterance_id': utterance_id,
            'num_tokens': len(tokens),
            'num_heads': len(head_analysis),
            'num_circuits': len(circuits),
            'circuit_ids': circuit_ids
        }

    def query_circuits_for_utterance(self, utterance_id: str) -> Dict[str, Any]:
        """
        Query all circuits for a given utterance.

        Args:
            utterance_id: ID of the utterance

        Returns:
            Dict with circuits and related data
        """
        query = """
        MATCH (u:Utterance {utterance_id: $utterance_id})
        OPTIONAL MATCH (u)-[:HAS_CIRCUIT]->(c:AttentionCircuit)
        OPTIONAL MATCH (c)-[r:USES_HEAD]->(h:AttentionHead)
        RETURN u.text AS text,
               u.sentiment_label AS sentiment,
               u.sentiment_score AS score,
               collect(DISTINCT {
                   circuit_id: c.circuit_id,
                   circuit_type: c.circuit_type,
                   input_token: c.input_token,
                   num_heads: c.num_heads,
                   prediction: c.prediction,
                   confidence: c.confidence
               }) AS circuits,
               collect(DISTINCT {
                   layer: h.layer,
                   head: h.head,
                   entropy: h.entropy,
                   attention_weight: r.attention_weight
               }) AS heads
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, {'utterance_id': utterance_id})
            return result.single().data()

    def generate_circuit_explanation(self, utterance_id: str) -> str:
        """
        Generate human-readable explanation from circuits.

        Args:
            utterance_id: ID of the utterance

        Returns:
            Natural language explanation
        """
        data = self.query_circuits_for_utterance(utterance_id)

        if not data or not data.get('circuits'):
            return f"No attention circuits found for utterance '{utterance_id}'"

        text = data['text']
        sentiment = data['sentiment']
        score = data['score']
        circuits = [c for c in data['circuits'] if c['circuit_id']]  # Filter nulls

        explanation = f"**Attention Circuit Analysis**\n\n"
        explanation += f"Text: \"{text}\"\n"
        explanation += f"Prediction: {sentiment} (confidence: {abs(score):.2f})\n\n"

        for i, circuit in enumerate(circuits, 1):
            explanation += f"**Circuit {i}: {circuit['circuit_type'].replace('_', ' ').title()}**\n"
            explanation += f"- Triggered by token: '{circuit['input_token']}'\n"
            explanation += f"- Uses {circuit['num_heads']} attention heads\n"
            explanation += f"- Contributes to: {circuit['prediction']}\n\n"

        return explanation


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("NEO4J ATTENTION CIRCUIT LOADER - EXAMPLE")
    print("="*80)

    # Initialize loader
    try:
        loader = AttentionCircuitLoader()

        # Create constraints
        print("\nCreating constraints...")
        loader.create_constraints()

        # Example data (from multimodal_attention_circuits.py output)
        example_data = {
            'utterance_data': {
                'utterance_id': 'mosei_train_000',
                'text': 'This is so pathetic.',
                'sentiment_score': -2.67,
                'sentiment_label': 'negative'
            },
            'tokens': ['[CLS]', 'this', 'is', 'so', 'pathetic', '.', '[SEP]'],
            'head_analysis': [
                {'layer': 2, 'head': 0, 'entropy': 0.043, 'focus_token': '[CLS]', 'focus_strength': 0.280},
                {'layer': 4, 'head': 1, 'entropy': 0.123, 'focus_token': 'pathetic', 'focus_strength': 0.402},
                # ... (would include all 144 heads in real usage)
            ],
            'circuits': [
                {
                    'circuit_type': 'sentiment_negative',
                    'input_token': 'pathetic',
                    'heads': [
                        {'layer': 4, 'head': 1, 'attention_weight': 0.402},
                        {'layer': 0, 'head': 1, 'attention_weight': 0.339},
                        {'layer': 9, 'head': 7, 'attention_weight': 0.311},
                    ],
                    'prediction': 'negative',
                    'confidence': 2.67
                }
            ],
            'cross_modal_sim': {
                'text_audio': -0.016,
                'text_video': 0.062,
                'audio_video': -0.027
            }
        }

        print("\nLoading example attention analysis...")
        result = loader.load_complete_analysis(**example_data)

        print("\n" + "="*80)
        print("LOADING COMPLETE")
        print("="*80)
        print(f"Utterance ID: {result['utterance_id']}")
        print(f"Tokens: {result['num_tokens']}")
        print(f"Heads: {result['num_heads']}")
        print(f"Circuits: {result['num_circuits']}")
        print(f"Circuit IDs: {result['circuit_ids']}")

        # Query back
        print("\n" + "="*80)
        print("QUERYING CIRCUITS")
        print("="*80)
        explanation = loader.generate_circuit_explanation('mosei_train_000')
        print(explanation)

        loader.close()

    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nMake sure Neo4j is running and credentials are set in .env file:")
        print("  NEO4J_URI=neo4j://localhost:7687")
        print("  NEO4J_USER=neo4j")
        print("  NEO4J_PASSWORD=your_password")
