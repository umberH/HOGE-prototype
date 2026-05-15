"""
Train Multimodal Sentiment Model
=================================
Train simple multimodal model on CMU-MOSEI data for attention analysis.

Usage:
    python backend/src/models/train_multimodal.py --epochs 5 --batch-size 8
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.src.models.multimodal_model import SimpleMultimodalModel
from transformers import BertTokenizer


class MOSEIDataset(Dataset):
    """Simple dataset wrapper for CMU-MOSEI pickle files"""

    def __init__(self, data_path, tokenizer):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Tokenize text
        tokens = self.tokenizer(
            sample['text'],
            padding='max_length',
            max_length=32,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'audio': torch.tensor(sample['audio_features'], dtype=torch.float32).mean(dim=0),  # Average over time
            'video': torch.tensor(sample['video_features'], dtype=torch.float32).mean(dim=0),  # Average over time
            'label': torch.tensor(sample['sentiment_score'], dtype=torch.float32)
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask, audio, video)
        predictions = outputs['sentiment']
        loss = criterion(predictions, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, audio, video)
            predictions = outputs['sentiment']
            loss = criterion(predictions, labels)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train multimodal sentiment model")
    parser.add_argument('--data-dir', type=str, default='data/cmu_mosei',
                       help='Directory containing train.pkl, val.pkl')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='.resources/models/multimodal',
                       help='Where to save trained model')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load datasets
    print("Loading datasets...")
    train_dataset = MOSEIDataset(f"{args.data_dir}/train.pkl", tokenizer)
    val_dataset = MOSEIDataset(f"{args.data_dir}/val.pkl", tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Initialize model
    print("Initializing model...")
    model = SimpleMultimodalModel(
        audio_dim=74,
        video_dim=35,
        fusion_type='concat'
    ).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = output_path / "sentiment_model.pt"
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved best model to {model_path}")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path / 'sentiment_model.pt'}")

    # Test loading
    print("\nVerifying model can be loaded...")
    test_model = SimpleMultimodalModel(audio_dim=74, video_dim=35, fusion_type='concat')
    test_model.load_state_dict(torch.load(output_path / "sentiment_model.pt"))
    print("✓ Model loads successfully!")


if __name__ == "__main__":
    main()
