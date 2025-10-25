#!/bin/bash
# AWS EC2 startup script for automated training
# Use this as user-data when launching instances

set -e

# Update system
apt-get update
apt-get install -y python3-pip git htop nvtop

# Clone repository (replace with your repo)
cd /home/ubuntu
# git clone https://github.com/yourusername/resnet50-imagenet.git
# cd resnet50-imagenet

# For now, create the files locally
mkdir -p resnet50-imagenet
cd resnet50-imagenet

# Install dependencies
pip3 install torch torchvision torchaudio
pip3 install ffcv numpy tqdm matplotlib albumentations opencv-python
pip3 install datasets huggingface_hub

# Create directories
mkdir -p /datasets/ffcv
mkdir -p checkpoints
mkdir -p logs

# Convert dataset to FFCV (if not already done)
python3 main.py convert-ffcv --ffcv-dir /datasets/ffcv

# Start distributed training with all optimizations
python3 main.py distributed \
    --use-ffcv \
    --batch-size 2048 \
    --epochs 100 \
    --progressive-resize \
    --use-ema \
    --compile \
    --budget-hours 1.5 \
    > training.log 2>&1 &

echo "Training started! Monitor with:"
echo "  tail -f training.log"
echo "  nvidia-smi -l 1"
