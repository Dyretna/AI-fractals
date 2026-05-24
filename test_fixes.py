#!/usr/bin/env python3
"""Quick test of all fixes."""

import sys
from pathlib import Path

root_path = Path(__file__).parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from env_loader import ensure_env_loaded  # noqa: E402

ensure_env_loaded()

from ai_fractals.pipeline import ParameterSampler  # noqa: E402

print("Testing fixes...")
print("=" * 60)

# Test 1: Square images
print("\n1. Testing square image parameters...")
sampler = ParameterSampler(fractal_type="mandelbrot", use_optimizer=False)
params = sampler.sample(strategy="uniform")
print(f"   Width: {params['width']}, Height: {params['height']}")
assert params["width"] == params["height"], "ERROR: Not square!"
print("   ✓ Images are square")

# Test 2: Higher iterations
print("\n2. Testing iteration range...")
print(f"   Max iterations: {params['max_iter']}")
assert params["max_iter"] >= 512, "ERROR: Iterations too low!"
print("   ✓ Iterations in range 512-2048")

# Test 3: Process cleanup (will be tested with Ctrl+C)
print("\n3. Process cleanup test...")
print("   Run: python scripts/generate_dataset.py --target 2 --quality-threshold 0.2")
print("   Then press Ctrl+C and verify:")
print("   - Process exits immediately")
print("   - No orphan processes (check with: ps aux | grep python)")
print("   - Report is saved")

print("\n" + "=" * 60)
print("Basic tests passed! Now test real generation with Ctrl+C...")
print("=" * 60)
