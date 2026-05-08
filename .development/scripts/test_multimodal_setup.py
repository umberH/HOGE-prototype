"""
Test Multimodal Setup
======================
Quick test to verify all dependencies are installed and working.

Run: python test_multimodal_setup.py
"""

import sys

def test_dependencies():
    """Test all required dependencies"""
    print("="*60)
    print("Testing Multimodal HOGE Dependencies")
    print("="*60)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Basic imports
    print("\n1. Testing basic imports...")
    try:
        import numpy
        import pandas
        import h5py
        print(f"   [PASS] NumPy: {numpy.__version__}")
        print(f"   [PASS] Pandas: {pandas.__version__}")
        print(f"   [PASS] h5py: {h5py.__version__}")
        tests_passed += 3
    except ImportError as e:
        print(f"   [FAIL] {e}")
        tests_failed += 1

    # Test 2: CMU-MultimodalSDK
    print("\n2. Testing CMU-MultimodalSDK...")
    try:
        from mmsdk import mmdatasdk
        print("   [PASS] CMU-MultimodalSDK (mmsdk) installed")
        tests_passed += 1
    except ImportError as e:
        print(f"   [FAIL] {e}")
        print("   Install with: pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git")
        tests_failed += 1

    # Test 3: Data loader
    print("\n3. Testing CMU-MOSEI data loader...")
    try:
        from src.data.cmu_mosei_loader import CMUMOSEILoader, MOSEISample
        print("   [PASS] Data loader imported successfully")
        loader = CMUMOSEILoader()
        print(f"   [PASS] Loader initialized: {loader.data_dir}")
        tests_passed += 2
    except Exception as e:
        print(f"   [FAIL] {e}")
        tests_failed += 1

    # Test 4: Optional dependencies
    print("\n4. Testing optional dependencies...")
    optional_libs = {
        'torch': 'PyTorch (for deep learning)',
        'transformers': 'Transformers (for BERT)',
        'neo4j': 'Neo4j driver (for knowledge graph)',
        'openai': 'OpenAI (for LLM explanations)'
    }

    for lib, desc in optional_libs.items():
        try:
            __import__(lib)
            print(f"   [PASS] {desc}")
            tests_passed += 1
        except ImportError:
            print(f"   [SKIP] {desc} (optional)")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")

    if tests_failed == 0:
        print("\n[SUCCESS] All required dependencies installed!")
        print("\nNext steps:")
        print("  1. Download CMU-MOSEI: python scripts/download_cmu_mosei.py")
        print("  2. Read quick start: MULTIMODAL_QUICKSTART.md")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Please install missing dependencies.")
        print("Run: pip install -r requirements_multimodal.txt")
        return 1


if __name__ == "__main__":
    exit_code = test_dependencies()
    sys.exit(exit_code)
