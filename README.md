# ResNet-50 ImageNet Training

Train ResNet-50 on ImageNet-1K to achieve **78% top-1 accuracy** within **$15 budget** using optimized techniques and multi-GPU support.

> **🆕 Quick Testing Feature**: Test the entire training pipeline in ~5 minutes with partial dataset support! Use `--partial-dataset` to train on just 1% of ImageNet for rapid iteration and debugging. See [Quick Testing](#quick-testing-with-partial-dataset-new) below.

## 📋 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (V100, A100, RTX 3090/4090)
- **RAM**: 32GB+ system memory
- **Storage**: 200GB+ for dataset and FFCV files (SSD recommended)
- **Multi-GPU**: 4-8 GPUs for fastest training (optional)

### Software
- **Python**: 3.9+ (3.10 or 3.11 recommended)
- **CUDA**: 11.8+ (12.4 recommended for latest GPUs)
- **PyTorch**: 2.5.0+ (automatically installed)
- **Operating System**: Linux (Ubuntu 20.04/22.04) or WSL2

## 📁 Project Structure

```
.
├── dataset.py      # Dataset management (downloading, FFCV conversion, loaders)
├── models.py       # ResNet-50 model architecture
├── train.py        # Training logic (single & multi-GPU support)
├── main.py         # Main entry point connecting everything
├── requirements.txt # Dependencies
└── README.md       # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

#### Automated Installation (Recommended)
```bash
# Run the installation script (auto-detects CUDA version)
chmod +x install.sh
./install.sh
```

#### Manual Installation
```bash
# For CUDA 12.x GPUs:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# For CUDA 11.8 GPUs:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### Verify Installation
```bash
# Check all package versions
python check_versions.py
```

### 2. Convert Dataset to FFCV (One-time, ~30-45 minutes)

```bash
# Authenticate with HuggingFace (if needed)
huggingface-cli login

# Convert ImageNet to FFCV format
python main.py convert-ffcv --ffcv-dir /datasets/ffcv
```

### 3. Train on Multiple GPUs (Fastest - 60-90 minutes on 8x A100)

```bash
# Automatically uses all available GPUs
python main.py distributed \
    --use-ffcv \
    --batch-size 2048 \
    --epochs 100 \
    --progressive-resize \
    --use-ema \
    --compile
```

## 📋 All Available Commands

### Quick Testing with Partial Dataset (NEW!)

#### Automatic Quick Test (Recommended for First-Time Setup)
```bash
# Run complete test with 1% of data (~5 min total)
chmod +x quick_test_partial.sh
./quick_test_partial.sh
```

#### Manual Partial Dataset Workflow
```bash
# Step 1: Convert partial dataset (1% by default)
python main.py convert-ffcv --partial-dataset

# Step 2: Train on partial dataset
python main.py train \
    --partial-dataset \
    --use-ffcv \
    --batch-size 128 \
    --epochs 5

# Custom partial sizes
python main.py convert-ffcv \
    --partial-dataset \
    --partial-size 10000 \      # 10K samples per split
    --partial-classes 100       # Only first 100 classes

# Train with custom partial
python main.py distributed \
    --partial-dataset \
    --partial-size 10000 \
    --partial-classes 100 \
    --use-ffcv \
    --epochs 10
```

### Dataset Management

#### Convert Full ImageNet to FFCV Format
```bash
python main.py convert-ffcv --ffcv-dir /datasets/ffcv
```

#### Test Setup
```bash
# Quick test to verify everything works
python main.py test --use-ffcv --max-samples 100
```

### Training Commands

#### Single GPU Training
```bash
# Basic training
python main.py train \
    --batch-size 256 \
    --epochs 100 \
    --lr 0.8

# With all optimizations
python main.py train \
    --use-ffcv \
    --batch-size 256 \
    --epochs 100 \
    --progressive-resize \
    --use-ema \
    --compile \
    --cutmix-prob 0.5 \
    --mixup-alpha 0.4
```

#### Multi-GPU Training (Distributed)
```bash
# Use all available GPUs
python main.py distributed \
    --use-ffcv \
    --batch-size 2048 \
    --epochs 100 \
    --progressive-resize \
    --use-ema \
    --compile

# Specify number of GPUs
python main.py distributed \
    --world-size 4 \
    --batch-size 1024 \
    --epochs 100
```

#### Resume Training from Checkpoint
```bash
python main.py train \
    --resume checkpoints/checkpoint_epoch_50.pt \
    --epochs 100
```

### Validation

#### Validate a Checkpoint
```bash
python main.py validate \
    --validate-only checkpoints/best_model.pt \
    --use-ffcv
```

### Advanced Options

#### Training with Budget Constraints
```bash
# Stop after 1.5 hours (for $15 budget on p4d.24xlarge)
python main.py distributed \
    --use-ffcv \
    --batch-size 2048 \
    --epochs 100 \
    --budget-hours 1.5
```

#### Test Mode with Subset
```bash
# Train on small subset for testing
python main.py train \
    --dataset test \
    --max-samples 10000 \
    --epochs 10
```

#### Custom Hyperparameters
```bash
python main.py distributed \
    --use-ffcv \
    --batch-size 2048 \
    --epochs 120 \
    --lr 1.0 \
    --momentum 0.9 \
    --weight-decay 5e-5 \
    --scheduler cosine \
    --label-smoothing 0.1 \
    --cutmix-prob 0.8 \
    --mixup-alpha 0.6 \
    --gradient-clip 1.0 \
    --checkpoint-interval 10 \
    --target-accuracy 78.0
```

## 🎯 Command-Line Arguments

### Mode Selection
- `train`: Single GPU training
- `distributed`: Multi-GPU distributed training
- `convert-ffcv`: Convert ImageNet to FFCV format
- `test`: Run quick tests
- `validate`: Validate a checkpoint

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| **Partial Dataset (NEW)** | | |
| `--partial-dataset` | False | Use partial dataset for quick testing |
| `--partial-size` | 12812 | Number of samples per split (default: 1% of full) |
| `--partial-classes` | 1000 | Number of classes to include |
| **Training** | | |
| `--use-ffcv` | False | Use FFCV for 3-5x faster data loading |
| `--batch-size` | 256 | Total batch size (divided across GPUs) |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 0.8 | Initial learning rate |
| `--progressive-resize` | False | Use progressive resizing (160→192→224) |
| `--use-ema` | False | Use exponential moving average |
| `--compile` | False | Use torch.compile (PyTorch 2.0+) |
| `--cutmix-prob` | 0.5 | CutMix probability |
| `--mixup-alpha` | 0.4 | MixUp alpha parameter |
| **Distributed** | | |
| `--world-size` | -1 | Number of GPUs (-1 for all) |
| `--num-workers` | 8 | Data loading workers per GPU |
| `--target-accuracy` | 78.0 | Early stopping target |

## 💰 Cost-Optimized Training on AWS

### Using p4d.24xlarge (8x A100 80GB)

```bash
# Launch spot instance (~$10-12/hour instead of $32.77)
aws ec2 run-instances \
    --instance-type p4d.24xlarge \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"12.00"}}' \
    --user-data file://startup.sh

# SSH into instance and run
python main.py distributed \
    --use-ffcv \
    --batch-size 2048 \
    --epochs 100 \
    --progressive-resize \
    --use-ema \
    --compile \
    --budget-hours 1.5
```

**Expected Results:**
- Training time: 60-90 minutes
- Cost: $12-15
- Accuracy: 77-78% top-1

### Using p3.8xlarge (4x V100)

```bash
python main.py distributed \
    --world-size 4 \
    --batch-size 1024 \
    --epochs 100
```

**Expected Results:**
- Training time: 2-3 hours
- Cost: $20-25
- Accuracy: 76-77% top-1

## 📊 Performance Tips

### 1. Always Use FFCV
```bash
# Convert once
python main.py convert-ffcv

# Then always train with --use-ffcv
python main.py train --use-ffcv ...
```

### 2. Maximize GPU Utilization
```bash
# Use largest batch size that fits in memory
# For 8x A100: 2048-4096
# For 8x V100: 1024-2048
# For single GPU: 256-512
```

### 3. Enable All Optimizations
```bash
--progressive-resize  # 2x speedup early epochs
--use-ema           # +0.5-1% accuracy
--compile           # 20-30% speedup (PyTorch 2.0+)
--amp              # Mixed precision (default on)
```

## 🔍 Monitoring Training

### Check Training Progress
```bash
# Watch logs in real-time
tail -f logs/train_*.log

# Monitor GPU usage
nvidia-smi -l 1

# Check checkpoint directory
ls -lh checkpoints/
```

### TensorBoard (Optional)
```bash
# If using TensorBoard logging
tensorboard --logdir logs/
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 128

# Or use gradient accumulation (not implemented yet)
```

### FFCV Files Not Found
```bash
# Create them first
python main.py convert-ffcv --ffcv-dir /path/to/ffcv
```

### Distributed Training Hangs
```bash
# Check all GPUs are visible
nvidia-smi

# Try different port
export MASTER_PORT=12356
```

### Slow Data Loading
```bash
# Must use FFCV
--use-ffcv

# Increase workers
--num-workers 12

# Use SSD storage for FFCV files
--ffcv-dir /nvme/ffcv
```

## 📈 Expected Training Progression

| Epoch | Image Size | Top-1 Acc | Time (8x A100) |
|-------|------------|-----------|----------------|
| 10 | 160×160 | 35-40% | 6 min |
| 30 | 192×192 | 55-60% | 18 min |
| 60 | 224×224 | 70-72% | 36 min |
| 80 | 224×224 | 75-76% | 48 min |
| 100 | 224×224 | **77-78%** | 60 min |

## 🎉 Results Summary

With all optimizations on appropriate hardware:
- **Accuracy**: 77-78% top-1, 93-94% top-5
- **Training Time**: 60-90 minutes on 8x A100
- **Cost**: $12-15 with spot instances
- **Key**: FFCV + Progressive Resizing + EMA + Multi-GPU

## 📦 Package Versions (October 2025)

### Core Dependencies
- **PyTorch**: 2.5.0+ (CUDA 12.4 support)
- **FFCV**: 1.0.0 (3-5x faster data loading)
- **NumPy**: 1.26.4 (avoiding v2.0 breaking changes)
- **Albumentations**: 1.4.0+ (stable API)
- **HuggingFace**: datasets 3.2.0+, transformers 4.46.0+

### New Features in Latest Versions
- **PyTorch 2.5**: Improved torch.compile performance, better FSDP support
- **FFCV 1.0**: Stable API, better memory efficiency
- **Datasets 3.2**: Faster streaming, improved caching

### Version Check
Run `python check_versions.py` to verify all packages are correctly installed.

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{resnet50-imagenet-2025,
  title={Efficient ResNet-50 Training on ImageNet},
  year={2025},
  url={https://github.com/yourusername/resnet50-imagenet}
}
```