"""
Simple Multimodal Model for CMU-MOSEI
======================================
A minimal multimodal sentiment analysis model using BERT + simple fusion.

This is a simplified version for quick proof-of-concept. For production,
consider using more sophisticated models like MAG-BERT or MulT.

Architecture:
    Text branch: BERT (bert-base-uncased)
    Audio branch: Simple MLP
    Video branch: Simple MLP
    Fusion: Weighted average or concatenation
    Output: Regression head for sentiment [-3, +3]

Usage:
    from src.models.multimodal_model import SimpleMultimodalModel

    model = SimpleMultimodalModel()
    sentiment = model(text_tokens, audio_features, video_features)
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, Optional, Tuple
import numpy as np


class SimpleMultimodalModel(nn.Module):
    """
    Simple multimodal model for sentiment analysis.

    Uses BERT for text and MLPs for audio/video, with simple fusion.
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        audio_dim: int = 74,  # COVAREP features
        video_dim: int = 35,  # Facet features
        hidden_dim: int = 256,
        fusion_type: str = 'concat',  # 'concat', 'weighted', 'attention'
        dropout: float = 0.3
    ):
        """
        Initialize multimodal model.

        Args:
            bert_model_name: Pre-trained BERT model name
            audio_dim: Audio feature dimension (COVAREP = 74)
            video_dim: Video feature dimension (Facet = 35)
            hidden_dim: Hidden dimension for audio/video MLPs
            fusion_type: How to fuse modalities ('concat', 'weighted', 'attention')
            dropout: Dropout rate
        """
        super().__init__()

        self.fusion_type = fusion_type

        # Text encoder: BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.text_dim = self.bert.config.hidden_size  # 768 for base

        # Audio encoder: Simple MLP
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Video encoder: Simple MLP
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion layer
        if fusion_type == 'concat':
            fusion_dim = self.text_dim + hidden_dim + hidden_dim
        elif fusion_type == 'weighted':
            fusion_dim = self.text_dim  # All mapped to same dim
            self.audio_proj = nn.Linear(hidden_dim, self.text_dim)
            self.video_proj = nn.Linear(hidden_dim, self.text_dim)
            self.modality_weights = nn.Parameter(torch.ones(3))
        elif fusion_type == 'attention':
            fusion_dim = self.text_dim
            self.audio_proj = nn.Linear(hidden_dim, self.text_dim)
            self.video_proj = nn.Linear(hidden_dim, self.text_dim)
            self.attention = nn.MultiheadAttention(self.text_dim, num_heads=4)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # Regression head for sentiment [-3, 3]
        self.sentiment_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        output_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) tokenized text
            attention_mask: (batch, seq_len) attention mask
            audio_features: (batch, seq_len, audio_dim) or (batch, audio_dim)
            video_features: (batch, seq_len, video_dim) or (batch, video_dim)
            output_attentions: Whether to return attention weights

        Returns:
            Dictionary with:
                - sentiment: (batch,) predicted sentiment scores
                - text_features: (batch, text_dim) text representations
                - audio_features: (batch, hidden_dim) audio representations
                - video_features: (batch, hidden_dim) video representations
                - attention_weights: (optional) attention weights
        """
        # Text encoding with BERT
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True
        )

        # Use [CLS] token representation
        text_emb = bert_output.last_hidden_state[:, 0, :]  # (batch, 768)

        # Audio encoding
        if len(audio_features.shape) == 3:  # (batch, seq_len, dim)
            audio_features = audio_features.mean(dim=1)  # Average pooling
        audio_emb = self.audio_encoder(audio_features)  # (batch, hidden_dim)

        # Video encoding
        if len(video_features.shape) == 3:  # (batch, seq_len, dim)
            video_features = video_features.mean(dim=1)  # Average pooling
        video_emb = self.video_encoder(video_features)  # (batch, hidden_dim)

        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([text_emb, audio_emb, video_emb], dim=1)
        elif self.fusion_type == 'weighted':
            audio_proj = self.audio_proj(audio_emb)
            video_proj = self.video_proj(video_emb)
            weights = torch.softmax(self.modality_weights, dim=0)
            fused = (weights[0] * text_emb +
                    weights[1] * audio_proj +
                    weights[2] * video_proj)
        elif self.fusion_type == 'attention':
            audio_proj = self.audio_proj(audio_emb).unsqueeze(0)  # (1, batch, dim)
            video_proj = self.video_proj(video_emb).unsqueeze(0)
            text_query = text_emb.unsqueeze(0)

            # Stack modalities
            modalities = torch.cat([text_query, audio_proj, video_proj], dim=0)  # (3, batch, dim)

            # Self-attention across modalities
            fused, attn_weights = self.attention(
                modalities, modalities, modalities
            )
            fused = fused.mean(dim=0)  # (batch, dim)

        # Sentiment prediction
        sentiment = self.sentiment_head(fused).squeeze(-1)  # (batch,)

        # Prepare output
        output = {
            'sentiment': sentiment,
            'text_features': text_emb,
            'audio_features': audio_emb,
            'video_features': video_emb
        }

        if output_attentions:
            output['bert_attentions'] = bert_output.attentions
            if self.fusion_type == 'attention':
                output['fusion_attention'] = attn_weights

        return output

    def get_modality_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute modality importance using ablation.

        Args:
            Same as forward()

        Returns:
            Dictionary with importance percentages:
                {'text': 0.6, 'audio': 0.25, 'video': 0.15}
        """
        self.eval()
        with torch.no_grad():
            # Full prediction
            full_output = self.forward(input_ids, attention_mask,
                                      audio_features, video_features)
            full_pred = full_output['sentiment']

            # Ablate text (use zeros)
            zero_ids = torch.zeros_like(input_ids)
            zero_mask = torch.zeros_like(attention_mask)
            no_text = self.forward(zero_ids, zero_mask,
                                  audio_features, video_features)['sentiment']

            # Ablate audio
            zero_audio = torch.zeros_like(audio_features)
            no_audio = self.forward(input_ids, attention_mask,
                                   zero_audio, video_features)['sentiment']

            # Ablate video
            zero_video = torch.zeros_like(video_features)
            no_video = self.forward(input_ids, attention_mask,
                                   audio_features, zero_video)['sentiment']

            # Compute importance as prediction change
            text_importance = torch.abs(full_pred - no_text).mean().item()
            audio_importance = torch.abs(full_pred - no_audio).mean().item()
            video_importance = torch.abs(full_pred - no_video).mean().item()

            # Normalize to percentages
            total = text_importance + audio_importance + video_importance
            if total == 0:
                return {'text': 0.33, 'audio': 0.33, 'video': 0.34}

            return {
                'text': round(text_importance / total * 100, 2),
                'audio': round(audio_importance / total * 100, 2),
                'video': round(video_importance / total * 100, 2)
            }


class MultimodalInferenceWrapper:
    """
    Easy-to-use wrapper for inference.

    Handles tokenization, feature preparation, and prediction.
    """

    def __init__(
        self,
        model: Optional[SimpleMultimodalModel] = None,
        tokenizer: Optional[BertTokenizer] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize wrapper.

        Args:
            model: Pre-trained multimodal model (or None to create new)
            tokenizer: BERT tokenizer (or None to load)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device

        # Load model
        if model is None:
            print(f"Creating new model...")
            model = SimpleMultimodalModel()

        self.model = model.to(device)
        self.model.eval()

        # Load tokenizer
        if tokenizer is None:
            print(f"Loading tokenizer...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.tokenizer = tokenizer

    def predict(
        self,
        text: str,
        audio_features: np.ndarray,
        video_features: np.ndarray,
        return_importance: bool = False
    ) -> Dict:
        """
        Predict sentiment for a single sample.

        Args:
            text: Text utterance
            audio_features: (seq_len, 74) or (74,) audio features
            video_features: (seq_len, 35) or (35,) video features
            return_importance: Whether to compute modality importance

        Returns:
            Dictionary with:
                - sentiment_score: Predicted sentiment
                - sentiment_label: 'positive', 'negative', or 'neutral'
                - modality_importance: (optional) importance percentages
        """
        # Tokenize text
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Prepare audio/video
        audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(self.device)
        video_tensor = torch.FloatTensor(video_features).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(input_ids, attention_mask, audio_tensor, video_tensor)
            sentiment_score = output['sentiment'].item()

        # Convert score to label
        if sentiment_score > 0.5:
            label = 'positive'
        elif sentiment_score < -0.5:
            label = 'negative'
        else:
            label = 'neutral'

        result = {
            'sentiment_score': round(sentiment_score, 3),
            'sentiment_label': label
        }

        # Compute importance if requested
        if return_importance:
            importance = self.model.get_modality_importance(
                input_ids, attention_mask, audio_tensor, video_tensor
            )
            result['modality_importance'] = importance

        return result

    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {path}")

    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from: {path}")


# Quick test function
def test_model():
    """Test the model with random data"""
    print("Testing Simple Multimodal Model...")

    # Create model
    model = SimpleMultimodalModel(fusion_type='concat')
    wrapper = MultimodalInferenceWrapper(model=model)

    # Create dummy data
    text = "I am so happy today!"
    audio_features = np.random.randn(74)  # Random audio
    video_features = np.random.randn(35)  # Random video

    # Predict
    result = wrapper.predict(text, audio_features, video_features,
                            return_importance=True)

    print(f"\nText: {text}")
    print(f"Sentiment: {result['sentiment_label']} ({result['sentiment_score']:.3f})")
    print(f"Modality Importance:")
    for modality, importance in result['modality_importance'].items():
        print(f"  {modality}: {importance}%")

    print("\nModel test passed!")


if __name__ == "__main__":
    test_model()
