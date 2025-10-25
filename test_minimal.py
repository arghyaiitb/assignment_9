#!/usr/bin/env python3
"""
Minimal test script to verify the entire training pipeline works correctly.
This runs with a tiny subset (100 samples, 10 classes) for ultra-fast testing.
Perfect for CI/CD or quick verification after code changes.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and check its status."""
    print(f"\n{'=' * 60}")
    print(f"📋 {description}")
    print(f"{'=' * 60}")
    print(f"Command: {cmd}\n")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ FAILED: {description}")
        sys.exit(1)
    else:
        print(f"✅ SUCCESS: {description} (took {elapsed:.1f}s)")

    return elapsed


def main():
    """Run minimal pipeline test."""
    print("\n" + "=" * 80)
    print(" MINIMAL PIPELINE TEST")
    print("=" * 80)
    print("\nThis will test the entire training pipeline with minimal data:")
    print("  • 100 samples per split")
    print("  • 10 classes only")
    print("  • 2 training epochs")
    print("  • Expected time: 1-2 minutes total\n")

    total_time = 0

    # Test 1: Import check
    cmd = "python -c 'import torch; import dataset; import train; import models; print(\"✅ All modules importable\")'"
    total_time += run_command(cmd, "Testing imports")

    # Test 2: Convert minimal dataset to FFCV
    ffcv_dir = Path("/tmp/ffcv_minimal_test")
    ffcv_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"""python main.py convert-ffcv \
        --partial-dataset \
        --partial-size 100 \
        --partial-classes 10 \
        --ffcv-dir {ffcv_dir}"""
    total_time += run_command(cmd, "Converting minimal dataset to FFCV")

    # Test 3: Train for 2 epochs
    checkpoint_dir = Path("./checkpoints_test")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"""python main.py train \
        --partial-dataset \
        --partial-size 100 \
        --partial-classes 10 \
        --use-ffcv \
        --ffcv-dir {ffcv_dir} \
        --batch-size 32 \
        --epochs 2 \
        --lr 0.01 \
        --checkpoint-interval 1"""
    total_time += run_command(cmd, "Training for 2 epochs")

    # Test 4: Test without FFCV (standard PyTorch DataLoader)
    cmd = """python main.py train \
        --partial-dataset \
        --partial-size 50 \
        --partial-classes 5 \
        --batch-size 16 \
        --epochs 1 \
        --lr 0.01"""
    total_time += run_command(cmd, "Testing standard DataLoader (no FFCV)")

    # Summary
    print("\n" + "=" * 80)
    print(" ✅ ALL TESTS PASSED!")
    print("=" * 80)
    print(f"\nTotal time: {total_time:.1f} seconds")
    print("\nThe training pipeline is working correctly!")
    print("\nNext steps:")
    print("  1. Run with more data: ./quick_test_partial.sh")
    print("  2. Full dataset: python main.py convert-ffcv")
    print("  3. Full training: python main.py distributed --use-ffcv --epochs 100")
    print("")


if __name__ == "__main__":
    main()
