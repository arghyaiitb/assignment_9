#!/bin/bash
# Quick test script for partial dataset workflow
# This allows you to test the entire training pipeline quickly with a small subset

set -e

echo "=================================================="
echo "Quick Test with Partial Dataset"
echo "=================================================="
echo ""
echo "This script will:"
echo "1. Convert a small subset of ImageNet to FFCV (1% of data)"
echo "2. Run a quick training test for 5 epochs"
echo "3. Validate the trained model"
echo ""

# Configuration
PARTIAL_SIZE=5000  # Number of samples (adjust as needed)
PARTIAL_CLASSES=100  # Number of classes (adjust as needed)
EPOCHS=5  # Quick training epochs

echo "Configuration:"
echo "  Samples per split: $PARTIAL_SIZE"
echo "  Number of classes: $PARTIAL_CLASSES"
echo "  Training epochs: $EPOCHS"
echo ""

# Step 1: Convert partial dataset to FFCV
echo "=================================================="
echo "Step 1: Converting partial dataset to FFCV..."
echo "=================================================="
python main.py convert-ffcv \
    --partial-dataset \
    --partial-size $PARTIAL_SIZE \
    --partial-classes $PARTIAL_CLASSES \
    --ffcv-dir /datasets/ffcv

echo ""
echo "✅ FFCV conversion complete!"
echo ""

# Step 2: Quick training test
echo "=================================================="
echo "Step 2: Running quick training test..."
echo "=================================================="

# Check if multiple GPUs are available
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1 || echo "1")

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Found $GPU_COUNT GPUs - using distributed training"
    python main.py distributed \
        --partial-dataset \
        --partial-size $PARTIAL_SIZE \
        --partial-classes $PARTIAL_CLASSES \
        --use-ffcv \
        --batch-size 128 \
        --epochs $EPOCHS \
        --lr 0.1 \
        --progressive-resize \
        --use-ema \
        --compile
else
    echo "Single GPU detected - using single GPU training"
    python main.py train \
        --partial-dataset \
        --partial-size $PARTIAL_SIZE \
        --partial-classes $PARTIAL_CLASSES \
        --use-ffcv \
        --batch-size 64 \
        --epochs $EPOCHS \
        --lr 0.1 \
        --progressive-resize \
        --use-ema \
        --compile
fi

echo ""
echo "✅ Training complete!"
echo ""

# Step 3: Validate the model
echo "=================================================="
echo "Step 3: Validating trained model..."
echo "=================================================="

# Find the latest checkpoint
LATEST_CHECKPOINT=$(ls -t checkpoints/*.pt 2>/dev/null | head -n1)

if [ ! -z "$LATEST_CHECKPOINT" ]; then
    echo "Validating checkpoint: $LATEST_CHECKPOINT"
    python main.py validate \
        --validate-only "$LATEST_CHECKPOINT" \
        --partial-dataset \
        --partial-size $PARTIAL_SIZE \
        --partial-classes $PARTIAL_CLASSES \
        --use-ffcv
else
    echo "⚠️ No checkpoint found. Skipping validation."
fi

echo ""
echo "=================================================="
echo "✅ Quick test complete!"
echo "=================================================="
echo ""
echo "Summary:"
echo "  - Partial dataset created with $PARTIAL_SIZE samples and $PARTIAL_CLASSES classes"
echo "  - Training ran for $EPOCHS epochs"
echo "  - FFCV files saved to: /datasets/ffcv_partial/"
echo "  - Checkpoints saved to: checkpoints/"
echo ""
echo "To run full training, use:"
echo "  python main.py convert-ffcv  # Convert full dataset"
echo "  python main.py distributed --use-ffcv --epochs 100  # Train on full dataset"
echo ""
