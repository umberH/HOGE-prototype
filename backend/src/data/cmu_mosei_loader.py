"""
CMU-MOSEI Data Loader (Minimal Version)
========================================
Loads pre-extracted features from CMU-MOSEI dataset.

Usage:
    from src.data.cmu_mosei_loader import CMUMOSEILoader

    loader = CMUMOSEILoader(data_dir="data/cmu_mosei")
    data = loader.load_aligned_features(split='train', max_samples=100)

    # Access data
    text_features = data['text']     # (N, seq_len, 768) BERT embeddings
    audio_features = data['audio']   # (N, seq_len, 74) COVAREP features
    video_features = data['video']   # (N, seq_len, 35) Facet features
    labels = data['labels']          # (N,) sentiment scores
    utterances = data['utterances']  # List of text strings
"""

import os
import pickle
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MOSEISample:
    """Single utterance sample from CMU-MOSEI"""
    utterance_id: str
    text: str
    speaker_id: str
    video_id: str
    start_time: float
    end_time: float

    # Features
    text_features: np.ndarray    # (seq_len, 768)
    audio_features: np.ndarray   # (seq_len, 74)
    video_features: np.ndarray   # (seq_len, 35)

    # Labels
    sentiment_score: float       # [-3, 3]
    sentiment_label: str         # "positive", "negative", "neutral"
    emotions: Dict[str, float]   # 6 emotions with intensities


class CMUMOSEILoader:
    """
    Loader for CMU-MOSEI dataset with pre-extracted features.

    Expected directory structure:
        data/cmu_mosei/
        ├── Raw/              # Raw videos (optional)
        ├── Processed/
        │   ├── text_features.pkl      # Pre-extracted BERT embeddings
        │   ├── audio_features.pkl     # COVAREP acoustic features
        │   ├── video_features.pkl     # Facet visual features
        │   ├── labels.pkl             # Sentiment + emotion labels
        │   └── metadata.pkl           # Utterance metadata
        └── Aligned/
            └── mosei_senti_data.pkl   # Fully aligned dataset
    """

    EMOTION_LABELS = ['happy', 'sad', 'angry', 'surprised', 'disgusted', 'fear']

    def __init__(self, data_dir: str = "data/cmu_mosei"):
        """
        Initialize loader.

        Args:
            data_dir: Path to CMU-MOSEI data directory
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "Processed"
        self.aligned_dir = self.data_dir / "Aligned"

        # Check if data exists
        if not self.data_dir.exists():
            print(f"⚠️  Data directory not found: {self.data_dir}")
            print("📥 To download CMU-MOSEI:")
            print("    pip install CMU-MultimodalSDK")
            print("    python scripts/download_cmu_mosei.py")

    def load_aligned_features(
        self,
        split: str = 'train',
        max_samples: Optional[int] = None,
        use_sdk: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Load aligned multimodal features.

        Args:
            split: Dataset split ('train', 'valid', 'test')
            max_samples: Maximum number of samples to load (None = all)
            use_sdk: If True, use CMU-SDK; else load from pickle files

        Returns:
            Dictionary with keys:
                - text: (N, seq_len, 768) BERT embeddings
                - audio: (N, seq_len, 74) COVAREP features
                - video: (N, seq_len, 35) Facet features
                - labels: (N,) sentiment scores [-3, 3]
                - utterances: List[str] text transcripts
                - metadata: List[Dict] utterance info
        """
        if use_sdk:
            return self._load_with_sdk(split, max_samples)
        else:
            return self._load_from_pickle(split, max_samples)

    def _load_with_sdk(self, split: str, max_samples: Optional[int]) -> Dict:
        """Load data using CMU-MultimodalSDK"""
        try:
            from mmsdk import mmdatasdk
        except ImportError:
            raise ImportError(
                "CMU-MultimodalSDK not installed. Install with:\n"
                "pip install CMU-MultimodalSDK"
            )

        print(f"📥 Loading CMU-MOSEI {split} split using CMU-SDK...")

        # Download/load dataset
        data_folder = str(self.data_dir / 'SDK')

        # Feature paths (will auto-download if not present)
        DATASET = mmdatasdk.mmdataset(
            recipe={
                'text': os.path.join(data_folder, 'CMU_MOSEI_TimestampedWords.csd'),
                'glove': os.path.join(data_folder, 'CMU_MOSEI_TimestampedWordVectors.csd'),
                'covarep': os.path.join(data_folder, 'CMU_MOSEI_COVAREP.csd'),
                'facet': os.path.join(data_folder, 'CMU_MOSEI_VisualFacet42.csd'),
                'labels': os.path.join(data_folder, 'CMU_MOSEI_Labels.csd')
            }
        )

        # Align features
        DATASET.align('glove', collapse_functions=[mmdatasdk.avg])

        # Get split indices
        split_file = self.data_dir / f'{split}_split_Depression_AVEC2017.csv'
        if split_file.exists():
            split_ids = pd.read_csv(split_file)['Id'].tolist()
        else:
            # Use all data if no split file
            split_ids = list(DATASET.computational_sequences['labels'].data.keys())

        if max_samples:
            split_ids = split_ids[:max_samples]

        # Extract features
        data = {
            'text': [],
            'audio': [],
            'video': [],
            'labels': [],
            'utterances': [],
            'metadata': []
        }

        for utt_id in split_ids:
            try:
                # Text (use GloVe embeddings as proxy for BERT)
                text_feat = DATASET.computational_sequences['glove'].data[utt_id]['features']

                # Audio (COVAREP)
                audio_feat = DATASET.computational_sequences['covarep'].data[utt_id]['features']

                # Video (Facet)
                video_feat = DATASET.computational_sequences['facet'].data[utt_id]['features']

                # Label (sentiment)
                label = DATASET.computational_sequences['labels'].data[utt_id]['features'][0][0]

                # Transcript
                words = DATASET.computational_sequences['text'].data[utt_id]['features']
                text = ' '.join([w[0].decode('utf-8') if isinstance(w[0], bytes) else w[0] for w in words])

                data['text'].append(text_feat)
                data['audio'].append(audio_feat)
                data['video'].append(video_feat)
                data['labels'].append(label)
                data['utterances'].append(text)
                data['metadata'].append({
                    'utterance_id': utt_id,
                    'speaker_id': utt_id.split('[')[0],
                    'video_id': utt_id.split('[')[0]
                })

            except KeyError:
                continue

        print(f"✅ Loaded {len(data['labels'])} samples")
        return data

    def _load_from_pickle(self, split: str, max_samples: Optional[int]) -> Dict:
        """Load data from pre-processed pickle files"""
        print(f"📂 Loading from pickle files: {split} split...")

        aligned_file = self.aligned_dir / f'mosei_{split}.pkl'

        if not aligned_file.exists():
            raise FileNotFoundError(
                f"Aligned data file not found: {aligned_file}\n"
                f"Please run preprocessing first or use use_sdk=True"
            )

        with open(aligned_file, 'rb') as f:
            data = pickle.load(f)

        if max_samples:
            for key in data:
                if isinstance(data[key], list):
                    data[key] = data[key][:max_samples]
                elif isinstance(data[key], np.ndarray):
                    data[key] = data[key][:max_samples]

        print(f"✅ Loaded {len(data['labels'])} samples")
        return data

    def get_sample(self, data: Dict, index: int) -> MOSEISample:
        """
        Get single sample as dataclass.

        Args:
            data: Output from load_aligned_features()
            index: Sample index

        Returns:
            MOSEISample dataclass
        """
        sentiment_score = float(data['labels'][index])

        # Convert score to label
        if sentiment_score > 0.5:
            sentiment_label = "positive"
        elif sentiment_score < -0.5:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        # Extract metadata
        meta = data['metadata'][index] if 'metadata' in data else {}

        return MOSEISample(
            utterance_id=meta.get('utterance_id', f'sample_{index}'),
            text=data['utterances'][index],
            speaker_id=meta.get('speaker_id', 'unknown'),
            video_id=meta.get('video_id', 'unknown'),
            start_time=meta.get('start_time', 0.0),
            end_time=meta.get('end_time', 0.0),
            text_features=data['text'][index],
            audio_features=data['audio'][index],
            video_features=data['video'][index],
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            emotions={}  # Will be populated if emotion labels available
        )

    def create_dataloader(
        self,
        data: Dict,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        """
        Create PyTorch DataLoader (requires torch).

        Args:
            data: Output from load_aligned_features()
            batch_size: Batch size
            shuffle: Whether to shuffle data

        Returns:
            torch.utils.data.DataLoader
        """
        try:
            import torch
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError("PyTorch required for DataLoader. Install with: pip install torch")

        # Convert to tensors
        text_tensor = torch.FloatTensor(np.array(data['text']))
        audio_tensor = torch.FloatTensor(np.array(data['audio']))
        video_tensor = torch.FloatTensor(np.array(data['video']))
        labels_tensor = torch.FloatTensor(data['labels'])

        dataset = TensorDataset(text_tensor, audio_tensor, video_tensor, labels_tensor)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Set to >0 for parallel loading
        )

    @staticmethod
    def download_dataset(output_dir: str = "data/cmu_mosei"):
        """
        Download CMU-MOSEI using CMU-SDK.

        Args:
            output_dir: Where to save the dataset
        """
        try:
            from mmsdk import mmdatasdk
        except ImportError:
            raise ImportError(
                "CMU-MultimodalSDK not installed. Install with:\n"
                "pip install CMU-MultimodalSDK"
            )

        print("📥 Downloading CMU-MOSEI dataset...")
        print("This may take 10-30 minutes depending on your connection.")

        output_path = Path(output_dir) / 'SDK'
        output_path.mkdir(parents=True, exist_ok=True)

        # Download all modalities
        mmdatasdk.mmdataset.download(
            'CMU_MOSEI',
            str(output_path),
            download_list=[
                'CMU_MOSEI_TimestampedWords',
                'CMU_MOSEI_TimestampedWordVectors',
                'CMU_MOSEI_COVAREP',
                'CMU_MOSEI_VisualFacet42',
                'CMU_MOSEI_Labels'
            ]
        )

        print(f"✅ Download complete! Data saved to: {output_path}")


# Quick test function
def test_loader():
    """Test the loader with sample data"""
    print("🧪 Testing CMU-MOSEI Loader...")

    loader = CMUMOSEILoader()

    try:
        # Try loading with SDK
        data = loader.load_aligned_features(split='train', max_samples=10, use_sdk=True)

        print(f"\n📊 Data Statistics:")
        print(f"  Samples: {len(data['labels'])}")
        print(f"  Text shape: {data['text'][0].shape}")
        print(f"  Audio shape: {data['audio'][0].shape}")
        print(f"  Video shape: {data['video'][0].shape}")
        print(f"  Sentiment range: [{min(data['labels']):.2f}, {max(data['labels']):.2f}]")

        # Get sample
        sample = loader.get_sample(data, 0)
        print(f"\n📝 Sample Utterance:")
        print(f"  Text: {sample.text}")
        print(f"  Sentiment: {sample.sentiment_label} ({sample.sentiment_score:.2f})")
        print(f"  Speaker: {sample.speaker_id}")

        print("\n✅ Loader test passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTo download CMU-MOSEI:")
        print("  CMUMOSEILoader.download_dataset()")


if __name__ == "__main__":
    test_loader()
