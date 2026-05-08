"""
Download CMU-MOSEI Dataset
===========================
Downloads the CMU-MOSEI dataset using the official CMU-MultimodalSDK.

Usage:
    python scripts/download_cmu_mosei.py

The dataset will be downloaded to: data/cmu_mosei/SDK/

Dataset size: ~3-5 GB
Time: 10-30 minutes (depending on internet speed)
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cmu_mosei_loader import CMUMOSEILoader


def main():
    """Download CMU-MOSEI dataset"""

    print("="*60)
    print("CMU-MOSEI Dataset Download")
    print("="*60)

    # Check if CMU-SDK is installed
    try:
        import mmsdk
        print("✅ CMU-MultimodalSDK is installed")
    except ImportError:
        print("❌ CMU-MultimodalSDK not found")
        print("\nInstalling CMU-MultimodalSDK...")
        os.system("pip install CMU-MultimodalSDK")

        try:
            import mmsdk
            print("✅ CMU-MultimodalSDK installed successfully")
        except ImportError:
            print("❌ Failed to install CMU-MultimodalSDK")
            print("Please install manually:")
            print("  pip install CMU-MultimodalSDK")
            sys.exit(1)

    # Download dataset
    output_dir = "data/cmu_mosei"

    print(f"\n📁 Output directory: {output_dir}")
    print("\n⚠️  WARNING: This will download ~3-5 GB of data")
    print("Download includes:")
    print("  - Text transcripts (words + timestamps)")
    print("  - GloVe word embeddings (300-dim)")
    print("  - COVAREP audio features (74-dim)")
    print("  - Facet visual features (42-dim)")
    print("  - Sentiment labels ([-3, +3] scale)")

    response = input("\nProceed with download? [y/N]: ")

    if response.lower() != 'y':
        print("❌ Download cancelled")
        sys.exit(0)

    # Download
    try:
        CMUMOSEILoader.download_dataset(output_dir=output_dir)

        print("\n" + "="*60)
        print("✅ DOWNLOAD COMPLETE!")
        print("="*60)

        # Test the loader
        print("\n🧪 Testing data loader...")
        from src.data.cmu_mosei_loader import test_loader
        test_loader()

        print("\n📚 Next Steps:")
        print("  1. Explore the data:")
        print("     jupyter notebook notebooks/01_explore_cmu_mosei.ipynb")
        print("\n  2. Load data in Python:")
        print("     from src.data.cmu_mosei_loader import CMUMOSEILoader")
        print("     loader = CMUMOSEILoader()")
        print("     data = loader.load_aligned_features(split='train', max_samples=100)")

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Check disk space (~5 GB required)")
        print("  3. Try again (download will resume)")
        sys.exit(1)


if __name__ == "__main__":
    main()
