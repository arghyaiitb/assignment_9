#!/bin/bash
# EBS Data Preparation Script for EC2
# Run this on a cheap t3.large instance to prepare ImageNet data

set -e

echo "=================================================="
echo "ImageNet Data Preparation on EBS"
echo "=================================================="

# Configuration
EBS_DEVICE=${EBS_DEVICE:-/dev/xvdf}
MOUNT_POINT=${MOUNT_POINT:-/data}
FFCV_DIR=${FFCV_DIR:-$MOUNT_POINT/ffcv}
HF_CACHE=${HF_CACHE:-$MOUNT_POINT/huggingface_cache}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
    exit 1
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Step 1: Check if EBS is attached
echo ""
echo "Step 1: Checking EBS attachment..."
if [ ! -b "$EBS_DEVICE" ]; then
    # Try alternative device name for Nitro instances
    if [ -b "/dev/nvme1n1" ]; then
        EBS_DEVICE="/dev/nvme1n1"
        print_warning "Using Nitro device: $EBS_DEVICE"
    else
        print_error "EBS device not found at $EBS_DEVICE. Please attach EBS volume first."
    fi
fi
print_status "EBS device found at $EBS_DEVICE"

# Step 2: Check if already mounted
echo ""
echo "Step 2: Checking mount status..."
if mount | grep -q "$MOUNT_POINT"; then
    print_status "EBS already mounted at $MOUNT_POINT"
else
    # Create mount point
    sudo mkdir -p $MOUNT_POINT
    
    # Check if filesystem exists
    if ! sudo file -s $EBS_DEVICE | grep -q "filesystem"; then
        print_warning "No filesystem found. Creating XFS filesystem..."
        sudo mkfs -t xfs $EBS_DEVICE
    fi
    
    # Mount the volume
    print_status "Mounting EBS to $MOUNT_POINT..."
    sudo mount $EBS_DEVICE $MOUNT_POINT
    
    # Change ownership
    sudo chown $(whoami):$(whoami) $MOUNT_POINT
    
    # Make mount persistent
    if ! grep -q "$EBS_DEVICE" /etc/fstab; then
        echo "$EBS_DEVICE $MOUNT_POINT xfs defaults,nofail 0 2" | sudo tee -a /etc/fstab
        print_status "Added persistent mount to /etc/fstab"
    fi
fi

# Step 3: Check available space
echo ""
echo "Step 3: Checking available space..."
AVAILABLE_SPACE=$(df -BG $MOUNT_POINT | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 200 ]; then
    print_error "Insufficient space. Need at least 200GB, only ${AVAILABLE_SPACE}GB available."
fi
print_status "Available space: ${AVAILABLE_SPACE}GB"

# Step 4: Install Python dependencies
echo ""
echo "Step 4: Installing Python dependencies..."

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_warning "pip3 not found. Installing..."
    sudo apt-get update && sudo apt-get install -y python3-pip || \
    sudo yum install -y python3-pip
fi

# Install CPU-only PyTorch (faster for data prep)
print_status "Installing PyTorch (CPU version for data prep)..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
print_status "Installing data processing libraries..."
pip3 install \
    datasets \
    huggingface_hub \
    ffcv \
    numpy \
    tqdm \
    Pillow

# Step 5: Setup directories
echo ""
echo "Step 5: Setting up directories..."
mkdir -p $FFCV_DIR
mkdir -p $HF_CACHE
mkdir -p $MOUNT_POINT/checkpoints
mkdir -p $MOUNT_POINT/logs

print_status "Created directories:"
echo "  - FFCV data: $FFCV_DIR"
echo "  - HF cache: $HF_CACHE"
echo "  - Checkpoints: $MOUNT_POINT/checkpoints"
echo "  - Logs: $MOUNT_POINT/logs"

# Step 6: Set environment variables
echo ""
echo "Step 6: Setting environment variables..."
export HF_HOME=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE/datasets
export FFCV_DIR=$FFCV_DIR

# Add to bashrc for persistence
if ! grep -q "HF_HOME=$HF_CACHE" ~/.bashrc; then
    echo "export HF_HOME=$HF_CACHE" >> ~/.bashrc
    echo "export HF_DATASETS_CACHE=$HF_CACHE/datasets" >> ~/.bashrc
    echo "export FFCV_DIR=$FFCV_DIR" >> ~/.bashrc
    print_status "Added environment variables to ~/.bashrc"
fi

# Step 7: Clone project repository
echo ""
echo "Step 7: Checking project repository..."
if [ ! -d "$MOUNT_POINT/resnet50-imagenet" ]; then
    print_status "Cloning project repository..."
    cd $MOUNT_POINT
    git clone https://github.com/yourusername/resnet50-imagenet.git || {
        print_warning "Could not clone repo. Creating project directory..."
        mkdir -p $MOUNT_POINT/resnet50-imagenet
    }
fi
cd $MOUNT_POINT/resnet50-imagenet

# Step 8: Check HuggingFace authentication
echo ""
echo "Step 8: Checking HuggingFace authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    print_warning "Not logged in to HuggingFace."
    echo "Please run: huggingface-cli login"
    echo "You need this to download ImageNet."
    echo ""
    echo "After logging in, continue with data conversion:"
    echo "  cd $MOUNT_POINT/resnet50-imagenet"
    echo "  python3 main.py convert-ffcv --ffcv-dir $FFCV_DIR"
else
    print_status "HuggingFace authentication OK"
    
    # Step 9: Start data conversion
    echo ""
    echo "Step 9: Ready to convert data!"
    echo ""
    echo "=================================================="
    echo "Next Steps:"
    echo "=================================================="
    echo ""
    echo "1. Convert FULL ImageNet to FFCV (~3-5 hours):"
    echo "   python3 main.py convert-ffcv --ffcv-dir $FFCV_DIR"
    echo ""
    echo "2. Or create a PARTIAL dataset for testing (~10 minutes):"
    echo "   python3 main.py convert-ffcv --partial-dataset --ffcv-dir ${FFCV_DIR}_partial"
    echo ""
    echo "3. Monitor progress:"
    echo "   watch -n 5 'df -h $MOUNT_POINT; ls -lah $FFCV_DIR/'"
    echo ""
    echo "4. When complete, create marker file:"
    echo "   echo 'Data ready: $(date)' > $MOUNT_POINT/DATA_READY.txt"
    echo ""
    echo "5. Then detach EBS and terminate this instance to save money!"
    echo ""
fi

# Create status file
echo "Setup complete: $(date)" > $MOUNT_POINT/SETUP_STATUS.txt
echo "EBS mounted at: $MOUNT_POINT" >> $MOUNT_POINT/SETUP_STATUS.txt
echo "FFCV directory: $FFCV_DIR" >> $MOUNT_POINT/SETUP_STATUS.txt
df -h $MOUNT_POINT >> $MOUNT_POINT/SETUP_STATUS.txt

print_status "Setup complete! EBS is ready for data preparation."
