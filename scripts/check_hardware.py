#!/usr/bin/env python3

import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from env_loader import ensure_env_loaded  # noqa: E402

ensure_env_loaded()

from ai_fractals.hardware_config import setup_hardware  # noqa: E402


def main():
    """Check hardware configuration and guide user."""
    print("\n" + "=" * 60)
    print("AI-FRACTALS HARDWARE SETUP")
    print("=" * 60)
    print("\nDetecting hardware configuration...\n")

    # Setup and detect
    config = setup_hardware(verbose=True)

    # Recommendations based on detection
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if config.has_gpu:
        print("\n[OK] Your GPU is ready to use!")
        print("\nOptimal settings for your hardware:")
        print("   - Batch size: 32-64 (depending on model)")
        print("   - Mixed precision: Enabled (faster training)")
        print("   - Memory growth: Enabled (efficient memory use)")
    else:
        print("\n[WARNING] No GPU detected. Running in CPU mode.")
        print("   - Batch size: 4-8 (smaller batches)")
        print("   - More CPU workers for data loading")

    print("\n" + "=" * 60 + "\n")

    return 0 if config.has_gpu else 1


if __name__ == "__main__":
    sys.exit(main())
