"""
Generate Synthetic CMU-MOSEI-like Data for Testing
===================================================
Creates synthetic multimodal data matching CMU-MOSEI format
without requiring the actual dataset download.

This allows testing the full pipeline before getting real data.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json

def generate_synthetic_mosei(num_samples=100, output_dir="data/cmu_mosei"):
    """
    Generate synthetic multimodal sentiment data

    Args:
        num_samples: Number of samples to generate
        output_dir: Where to save the data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} synthetic MOSEI samples...")

    # Sentiment keywords for realistic text generation
    positive_words = ["love", "happy", "great", "amazing", "wonderful", "excellent",
                      "fantastic", "beautiful", "perfect", "awesome"]
    negative_words = ["hate", "terrible", "awful", "horrible", "bad", "worst",
                      "disgusting", "disappointing", "pathetic", "useless"]
    neutral_words = ["okay", "fine", "average", "normal", "decent", "acceptable",
                     "standard", "typical", "ordinary", "fair"]

    samples = []

    for i in range(num_samples):
        # Randomly choose sentiment
        sentiment_type = np.random.choice(['positive', 'negative', 'neutral'],
                                         p=[0.4, 0.3, 0.3])

        # Generate text based on sentiment
        if sentiment_type == 'positive':
            word = np.random.choice(positive_words)
            sentiment_score = np.random.uniform(1.0, 3.0)
            text = f"I think this is {word}!"
        elif sentiment_type == 'negative':
            word = np.random.choice(negative_words)
            sentiment_score = np.random.uniform(-3.0, -1.0)
            text = f"This is so {word}."
        else:
            word = np.random.choice(neutral_words)
            sentiment_score = np.random.uniform(-0.5, 0.5)
            text = f"It's just {word}."

        # Generate features with some correlation to sentiment
        seq_len = np.random.randint(10, 30)

        # Text features (BERT-like, 768-dim)
        # Add sentiment bias to embeddings
        text_features = np.random.randn(seq_len, 768) * 0.1
        text_features += sentiment_score * 0.05  # Slight correlation

        # Audio features (COVAREP, 74-dim)
        # Positive = higher pitch/energy, negative = lower
        audio_features = np.random.randn(seq_len, 74) * 0.5
        audio_features[:, :10] += sentiment_score * 0.1  # Pitch features

        # Video features (Facet, 35-dim)
        # Positive = more facial movement, negative = less
        video_features = np.random.randn(seq_len, 35) * 0.3
        video_features[:, :5] += abs(sentiment_score) * 0.1  # Activity features

        # Generate emotions (6 basic emotions)
        emotions = {
            'happiness': max(0, sentiment_score) if sentiment_type == 'positive' else 0,
            'sadness': abs(min(0, sentiment_score)) if sentiment_type == 'negative' else 0,
            'anger': abs(sentiment_score * 0.5) if sentiment_type == 'negative' else 0,
            'surprise': np.random.uniform(0, 1),
            'disgust': abs(sentiment_score * 0.3) if sentiment_type == 'negative' else 0,
            'fear': np.random.uniform(0, 0.5)
        }

        sample = {
            'utterance_id': f'synthetic_{i:04d}',
            'text': text,
            'speaker_id': f'speaker_{i % 20}',
            'video_id': f'video_{i % 50}',
            'start_time': float(i * 2.5),
            'end_time': float(i * 2.5 + 2.5),
            'text_features': text_features,
            'audio_features': audio_features,
            'video_features': video_features,
            'sentiment_score': float(sentiment_score),
            'sentiment_label': sentiment_type,
            'emotions': emotions
        }

        samples.append(sample)

    # Split into train/val/test
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)

    splits = {
        'train': samples[:train_size],
        'val': samples[train_size:train_size+val_size],
        'test': samples[train_size+val_size:]
    }

    # Save each split
    for split_name, split_samples in splits.items():
        output_file = output_path / f"{split_name}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(split_samples, f)
        print(f"Saved {len(split_samples)} {split_name} samples to {output_file}")

    # Save metadata
    metadata = {
        'num_samples': num_samples,
        'splits': {k: len(v) for k, v in splits.items()},
        'feature_dims': {
            'text': 768,
            'audio': 74,
            'video': 35
        },
        'sentiment_range': [-3.0, 3.0],
        'data_type': 'synthetic'
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")

    # Create a CSV summary for quick inspection
    summary_data = []
    for sample in samples[:20]:  # First 20 samples
        summary_data.append({
            'utterance_id': sample['utterance_id'],
            'text': sample['text'],
            'sentiment_label': sample['sentiment_label'],
            'sentiment_score': f"{sample['sentiment_score']:.2f}",
            'text_seq_len': len(sample['text_features']),
            'audio_seq_len': len(sample['audio_features']),
            'video_seq_len': len(sample['video_features'])
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = output_path / "sample_preview.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved preview to {summary_file}")

    print("\nDataset generation complete!")
    print(f"Total samples: {num_samples}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print(f"\nTo load the data:")
    print(f"  from src.data.cmu_mosei_loader import CMUMOSEILoader")
    print(f"  loader = CMUMOSEILoader(data_dir='{output_dir}')")
    print(f"  data = loader.load_aligned_features(split='train')")


if __name__ == "__main__":
    # Generate 100 samples for quick testing
    generate_synthetic_mosei(num_samples=100, output_dir="data/cmu_mosei")
