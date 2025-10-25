#!/bin/bash
# EBS Training Script for GPU EC2 Instance
# Run this on a GPU instance with the prepared EBS attached

set -e

echo "=================================================="
echo "ResNet-50 Training with Prepared EBS Data"
echo "=================================================="

# Configuration
EBS_DEVICE=${EBS_DEVICE:-/dev/nvme1n1}  # Nitro instances use NVMe naming
MOUNT_POINT=${MOUNT_POINT:-/data}
FFCV_DIR=${FFCV_DIR:-$MOUNT_POINT/ffcv}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$MOUNT_POINT/checkpoints}
LOG_DIR=${LOG_DIR:-$MOUNT_POINT/logs}
PROJECT_DIR=${PROJECT_DIR:-$MOUNT_POINT/assignment_9}

# Training configuration
BATCH_SIZE=${BATCH_SIZE:-2048}
EPOCHS=${EPOCHS:-100}
BUDGET_HOURS=${BUDGET_HOURS:-1.5}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Step 1: Check GPU availability
echo ""
echo "Step 1: Checking GPU availability..."
if ! nvidia-smi &> /dev/null; then
    print_error "No GPU detected! This script requires a GPU instance."
fi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n1)
print_status "Found $GPU_COUNT x $GPU_NAME"

# Step 2: Mount EBS with prepared data
echo ""
echo "Step 2: Mounting EBS with prepared data..."

# Check if EBS is attached
if [ ! -b "$EBS_DEVICE" ]; then
    # Try alternative device name for Nitro instances
    if [ -b "/dev/nvme1n1" ]; then
        EBS_DEVICE="/dev/nvme1n1"
        print_warning "Using Nitro device: $EBS_DEVICE"
    else
        print_error "EBS not attached! Please attach the EBS volume with prepared data."
    fi
fi

# Mount if not already mounted
if mount | grep -q "$MOUNT_POINT"; then
    print_status "EBS already mounted at $MOUNT_POINT"
else
    sudo mkdir -p $MOUNT_POINT
    sudo mount $EBS_DEVICE $MOUNT_POINT
    sudo chown $(whoami):$(whoami) $MOUNT_POINT
    print_status "Mounted EBS at $MOUNT_POINT"
fi

# Step 3: Verify prepared data exists
echo ""
echo "Step 3: Verifying prepared data..."

if [ ! -f "$FFCV_DIR/train.ffcv" ] || [ ! -f "$FFCV_DIR/val.ffcv" ]; then
    print_error "FFCV data not found! Expected files:"
    echo "  - $FFCV_DIR/train.ffcv"
    echo "  - $FFCV_DIR/val.ffcv"
    echo ""
    echo "Please run data preparation first on a cheap instance."
    exit 1
fi

TRAIN_SIZE=$(ls -lh $FFCV_DIR/train.ffcv | awk '{print $5}')
VAL_SIZE=$(ls -lh $FFCV_DIR/val.ffcv | awk '{print $5}')
print_status "Found training data:"
echo "  - train.ffcv: $TRAIN_SIZE"
echo "  - val.ffcv: $VAL_SIZE"

# Step 4: Setup project directory
echo ""
echo "Step 4: Setting up project..."

# Clone or update project
if [ ! -d "$PROJECT_DIR" ]; then
    print_info "Cloning project repository..."
    cd ~
    git clone https://github.com/arghyaiitb/assignment_9.git || {
        # If can't clone, try using the one on EBS
        if [ -d "$MOUNT_POINT/assignment_9" ]; then
            cp -r $MOUNT_POINT/assignment_9 $PROJECT_DIR
            print_warning "Using project from EBS"
        else
            print_error "Could not get project files"
        fi
    }
fi

cd $PROJECT_DIR

# Step 5: Setup Python environment and dependencies
echo ""
echo "Step 5: Setting up Python environment..."

# For NVIDIA Deep Learning AMI with PyTorch 2.8
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    print_status "NVIDIA Deep Learning AMI detected. Activating PyTorch environment..."
    source /opt/conda/etc/profile.d/conda.sh
    conda activate pytorch
    
    # Verify PyTorch and CUDA
    python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.cuda.is_available()}')" && \
        print_status "PyTorch 2.8 with CUDA ready" || \
        print_error "PyTorch/CUDA verification failed"
else
    # Fallback for standard AMI
    print_warning "Standard AMI detected. Installing PyTorch with CUDA..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install additional requirements not in AMI
print_info "Installing additional project requirements..."
pip install ffcv datasets huggingface_hub albumentations opencv-python wandb

print_status "Dependencies ready"

# Step 6: Check for existing checkpoints
echo ""
echo "Step 6: Checking for existing checkpoints..."

RESUME_FLAG=""
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/*.pt 2>/dev/null | head -n1)
    if [ ! -z "$LATEST_CHECKPOINT" ]; then
        print_info "Found checkpoint: $LATEST_CHECKPOINT"
        read -p "Resume from this checkpoint? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            RESUME_FLAG="--resume $LATEST_CHECKPOINT"
            print_status "Will resume from checkpoint"
        fi
    fi
fi

# Step 7: Determine training mode
echo ""
echo "Step 7: Configuring training..."

if [ "$GPU_COUNT" -gt 1 ]; then
    TRAINING_MODE="distributed"
    print_status "Using distributed training with $GPU_COUNT GPUs"
    
    # Adjust batch size for multiple GPUs
    BATCH_SIZE=$((BATCH_SIZE * GPU_COUNT / 4))  # Scale from 4-GPU baseline
else
    TRAINING_MODE="train"
    print_status "Using single GPU training"
fi

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="run_${TIMESTAMP}"

# Create directories for this run
mkdir -p "$CHECKPOINT_DIR/$RUN_NAME"
mkdir -p "$LOG_DIR/$RUN_NAME"

# Step 8: Start training
echo ""
echo "=================================================="
echo "Starting Training"
echo "=================================================="
echo "Mode: $TRAINING_MODE"
echo "GPUs: $GPU_COUNT x $GPU_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Budget hours: $BUDGET_HOURS"
echo "FFCV dir: $FFCV_DIR"
echo "Checkpoints: $CHECKPOINT_DIR/$RUN_NAME"
echo "Logs: $LOG_DIR/$RUN_NAME"
echo "=================================================="
echo ""

# Export environment variables for Python to use
export CHECKPOINT_DIR="$CHECKPOINT_DIR/$RUN_NAME"
export LOG_DIR="$LOG_DIR/$RUN_NAME"

# Build training command
TRAINING_CMD="python3 main.py $TRAINING_MODE \
    --use-ffcv \
    --ffcv-dir $FFCV_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr 0.8 \
    --warmup-epochs 8 \
    --scheduler onecycle \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --label-smoothing 0.1 \
    --gradient-clip 1.0 \
    --cutmix-prob 0.0 \
    --mixup-alpha 0.0 \
    --progressive-resize \
    --use-ema \
    --compile \
    --amp \
    --checkpoint-dir $CHECKPOINT_DIR/$RUN_NAME \
    --log-dir $LOG_DIR/$RUN_NAME \
    --checkpoint-interval 10 \
    --auto-resume \
    --budget-hours $BUDGET_HOURS \
    --target-accuracy 78 \
    --num-workers 8 \
    $RESUME_FLAG"

# Ask for confirmation
echo "Training command:"
echo "$TRAINING_CMD"
echo ""
read -p "Start training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Training cancelled"
    exit 0
fi

# Create training script
cat > "$LOG_DIR/$RUN_NAME/training_command.sh" << EOF
#!/bin/bash
# Training command for run $RUN_NAME
cd $PROJECT_DIR
$TRAINING_CMD
EOF
chmod +x "$LOG_DIR/$RUN_NAME/training_command.sh"

# Check if running in tmux
if [ -n "$TMUX" ]; then
    print_info "Already in tmux session. Good!"
    IN_TMUX=true
else
    print_warning "Not in tmux. Training might stop if SSH disconnects!"
    echo ""
    echo "Recommended: Start tmux first:"
    echo "  tmux new -s training"
    echo "  Then run this script again"
    echo ""
    read -p "Continue without tmux? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "To use tmux:"
        echo "  1. Run: tmux new -s training"
        echo "  2. Run this script inside tmux"
        echo "  3. Detach with Ctrl+B, then D"
        echo "  4. Reattach later with: tmux attach -t training"
        exit 0
    fi
    IN_TMUX=false
fi

# Start training
print_status "Starting training..."
echo ""

# Option 1: Run in foreground (recommended when in tmux)
if [ "$1" != "--background" ]; then
    if [ "$IN_TMUX" = true ]; then
        print_info "Training in tmux. You can safely detach with Ctrl+B, D"
    fi
    
    # Save output to log file while displaying
    $TRAINING_CMD 2>&1 | tee "$LOG_DIR/$RUN_NAME/training.log"
    
    # Save final checkpoint location
    echo "$CHECKPOINT_DIR/$RUN_NAME" > "$MOUNT_POINT/LATEST_RUN.txt"
    
    print_status "Training complete!"
    echo "Checkpoints saved to: $CHECKPOINT_DIR/$RUN_NAME"
    echo "Logs saved to: $LOG_DIR/$RUN_NAME"
else
    # Option 2: Run in background (when not using tmux)
    nohup $TRAINING_CMD > "$LOG_DIR/$RUN_NAME/training.log" 2>&1 &
    TRAINING_PID=$!
    
    echo $TRAINING_PID > "$LOG_DIR/$RUN_NAME/training.pid"
    
    print_status "Training started in background (PID: $TRAINING_PID)"
    echo ""
    echo "Monitor progress with:"
    echo "  tail -f $LOG_DIR/$RUN_NAME/training.log"
    echo "  nvidia-smi -l 1"
    echo ""
    echo "Stop training with:"
    echo "  kill $TRAINING_PID"
fi

# Create summary script
cat > "$MOUNT_POINT/view_results.sh" << 'EOF'
#!/bin/bash
echo "Training Results Summary"
echo "========================"
echo ""
echo "Latest runs:"
ls -lt $LOG_DIR | head -5
echo ""
echo "Best checkpoints:"
find $CHECKPOINT_DIR -name "best_model.pt" -exec ls -lh {} \;
echo ""
echo "GPU usage:"
nvidia-smi
EOF
chmod +x "$MOUNT_POINT/view_results.sh"

echo ""
echo "=================================================="
echo "Training Setup Complete!"
echo "=================================================="
echo ""
echo "Useful commands:"
echo "  - Monitor GPU: watch -n 1 nvidia-smi"
echo "  - View logs: tail -f $LOG_DIR/$RUN_NAME/training.log"
echo "  - Check results: $MOUNT_POINT/view_results.sh"
echo "  - List checkpoints: ls -lah $CHECKPOINT_DIR/$RUN_NAME/"
echo ""
echo "Remember to:"
echo "1. Monitor the training progress"
echo "2. Stop the instance when done to save money"
echo "3. Detach the EBS before terminating"
echo "4. Create an EBS snapshot for backup"
echo ""
