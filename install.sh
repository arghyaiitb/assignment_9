#!/bin/bash
# Installation script for ResNet-50 ImageNet training dependencies
# Supports both CUDA and CPU installations

set -e

echo "=================================================="
echo "ResNet-50 ImageNet Training - Package Installation"
echo "=================================================="

# Detect Python version
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Python version: $PYTHON_VERSION"

# Check if Python version is 3.9+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "❌ Error: Python 3.9+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Detect CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    
    # Try to detect CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
    echo "CUDA Version: $CUDA_VERSION"
    
    # Determine PyTorch CUDA version
    if [[ "$CUDA_VERSION" == "12"* ]]; then
        echo "Installing PyTorch with CUDA 12.4 support..."
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        CUPY_PACKAGE="cupy-cuda12x"
    elif [[ "$CUDA_VERSION" == "11"* ]]; then
        echo "Installing PyTorch with CUDA 11.8 support..."
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        CUPY_PACKAGE="cupy-cuda11x"
    else
        echo "⚠️ Unknown CUDA version. Installing CUDA 12.4 version..."
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        CUPY_PACKAGE="cupy-cuda12x"
    fi
else
    echo "⚠️ No NVIDIA GPU detected. Installing CPU-only version..."
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    CUPY_PACKAGE=""
fi

# Upgrade pip first
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with the appropriate CUDA/CPU support
echo ""
echo "📦 Installing PyTorch ecosystem..."
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url $TORCH_INDEX

# Install core requirements
echo ""
echo "📦 Installing core dependencies..."
pip install numpy==1.26.4 scipy>=1.14.1

# Install FFCV and dependencies
echo ""
echo "📦 Installing FFCV for fast data loading..."
pip install numba>=0.60.0
pip install ffcv>=1.0.0

# Install CuPy for GPU acceleration (if CUDA is available)
if [ ! -z "$CUPY_PACKAGE" ]; then
    echo ""
    echo "📦 Installing CuPy for GPU acceleration..."
    pip install $CUPY_PACKAGE || echo "⚠️ CuPy installation failed. Continuing without it..."
fi

# Install image processing libraries
echo ""
echo "📦 Installing image processing libraries..."
pip install opencv-python>=4.10.0.84
pip install Pillow>=11.0.0
pip install albumentations>=1.4.0

# Install Hugging Face ecosystem
echo ""
echo "📦 Installing Hugging Face ecosystem..."
pip install datasets>=3.2.0
pip install huggingface_hub>=0.26.0
pip install transformers>=4.46.0
pip install pyarrow>=18.0.0
pip install safetensors>=0.4.5

# Install training utilities
echo ""
echo "📦 Installing training utilities..."
pip install tqdm>=4.67.0
pip install matplotlib>=3.9.2
pip install torchinfo>=1.8.0
pip install pyyaml>=6.0.2

# Optional: Install experiment tracking
echo ""
echo "📦 Installing experiment tracking (optional)..."
pip install wandb>=0.18.0 || echo "⚠️ WandB installation failed"
pip install tensorboard>=2.18.0 tensorboardX>=2.6.2.2 || echo "⚠️ TensorBoard installation failed"

# Optional: Install distributed training support
echo ""
echo "📦 Installing distributed training support (optional)..."
pip install mpi4py>=3.1.6 || echo "⚠️ mpi4py installation failed (MPI not found)"

# Optional: Install Triton for torch.compile optimization
if [ ! -z "$CUPY_PACKAGE" ]; then
    echo ""
    echo "📦 Installing Triton for torch.compile optimization..."
    pip install triton || echo "⚠️ Triton installation failed"
fi

# Verify installation
echo ""
echo "=================================================="
echo "✅ Installation Complete! Verifying packages..."
echo "=================================================="
echo ""

python3 check_versions.py

echo ""
echo "🎉 Setup complete! You can now run:"
echo "  python main.py test  # Quick test"
echo "  python main.py convert-ffcv  # Convert dataset to FFCV"
echo "  python main.py train --use-ffcv  # Start training"
echo ""
