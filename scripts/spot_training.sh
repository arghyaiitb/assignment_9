#!/bin/bash
# Spot Instance Training Script - Automatically handles interruptions and resumption
# Perfect for AWS spot instances that can be terminated at any time

set -e

# Configuration
PROJECT_DIR=${PROJECT_DIR:-/data/assignment_9}
FFCV_DIR=${FFCV_DIR:-/data/ffcv}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/data/checkpoints}
LOG_DIR=${LOG_DIR:-$PROJECT_DIR/logs}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Training parameters (optimized for p4d.24xlarge with 8× A100s)
BATCH_SIZE=${BATCH_SIZE:-2048}  # 256 per GPU × 8 GPUs
EPOCHS=${EPOCHS:-60}            # Fewer epochs with larger batch
LR=${LR:-0.8}                   # Linear scaling: 0.1 × (2048/256) = 0.8
TARGET_ACCURACY=${TARGET_ACCURACY:-78.0}

echo -e "${BLUE}=================================================="
echo "      🚀 Spot Instance Training Manager"
echo "=================================================="
echo -e "${NC}"

# Function to check if we're on a spot instance
check_spot_instance() {
    # Check AWS instance metadata for spot instance
    if curl -s -f -m 1 http://169.254.169.254/latest/meta-data/spot/instance-action > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Spot instance termination notice detected!${NC}"
        return 0
    fi
    return 1
}

# Function to setup signal handlers for graceful shutdown
setup_signal_handlers() {
    trap 'echo -e "${YELLOW}Caught interrupt signal. Saving checkpoint...${NC}"; exit 0' INT TERM
}

# Function to mount data volume if not already mounted
mount_data_volume() {
    echo -e "${GREEN}Checking data volume...${NC}"
    
    if ! mountpoint -q /data; then
        echo "Mounting data volume..."
        # Try common device names
        for device in /dev/xvdf /dev/nvme1n1; do
            if [ -b "$device" ]; then
                sudo mount "$device" /data && break
            fi
        done
        sudo chown ubuntu:ubuntu /data
    fi
    
    # Verify FFCV data exists
    if [ ! -f "$FFCV_DIR/train.ffcv" ]; then
        echo -e "${RED}Error: FFCV data not found at $FFCV_DIR${NC}"
        echo "Please ensure your EBS volume is properly attached and contains the FFCV data."
        exit 1
    fi
    
    echo -e "${GREEN}✅ Data volume ready${NC}"
}

# Function to activate PyTorch environment
activate_pytorch() {
    echo -e "${GREEN}Activating PyTorch environment...${NC}"
    
    # Check if using NVIDIA Deep Learning AMI
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source /opt/conda/etc/profile.d/conda.sh
        conda activate pytorch
        echo -e "${GREEN}✅ PyTorch 2.8 environment activated${NC}"
    else
        # Try to activate venv if it exists
        if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
            source "$PROJECT_DIR/venv/bin/activate"
        fi
    fi
}

# Function to check for existing checkpoints
check_existing_progress() {
    echo -e "${GREEN}Checking for existing checkpoints...${NC}"
    
    if [ -f "$CHECKPOINT_DIR/checkpoint_latest.pt" ]; then
        # Extract epoch from checkpoint
        RESUME_EPOCH=$(python -c "
import torch
ckpt = torch.load('$CHECKPOINT_DIR/checkpoint_latest.pt', map_location='cpu')
print(ckpt.get('epoch', -1))
" 2>/dev/null || echo "-1")
        
        if [ "$RESUME_EPOCH" -gt "-1" ]; then
            echo -e "${GREEN}✅ Found checkpoint at epoch $RESUME_EPOCH${NC}"
            echo -e "${GREEN}   Training will automatically resume from this point${NC}"
            
            # Get best accuracy so far
            BEST_ACC=$(python -c "
import torch
ckpt = torch.load('$CHECKPOINT_DIR/checkpoint_latest.pt', map_location='cpu')
print(f\"{ckpt.get('best_accuracy', 0):.2f}\")
" 2>/dev/null || echo "0")
            echo -e "${GREEN}   Best accuracy so far: ${BEST_ACC}%${NC}"
        fi
    else
        echo -e "${YELLOW}No existing checkpoint found. Starting from scratch.${NC}"
    fi
}

# Function to monitor spot instance termination
monitor_spot_termination() {
    while true; do
        if check_spot_instance; then
            echo -e "${RED}Spot instance termination notice received!${NC}"
            echo "Stopping training gracefully..."
            # Send SIGTERM to training process
            pkill -TERM -f "python main.py"
            sleep 30
            exit 0
        fi
        sleep 30
    done
}

# Main training function
run_training() {
    cd $PROJECT_DIR
    
    # Determine if distributed training is possible
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    if [ "$NUM_GPUS" -gt 1 ]; then
        TRAINING_MODE="distributed"
        echo -e "${GREEN}Using distributed training with $NUM_GPUS GPUs${NC}"
        
        # Check if we're on p4d.24xlarge (8 A100s)
        if [ "$NUM_GPUS" -eq 8 ] && nvidia-smi | grep -q "A100"; then
            echo -e "${GREEN}✅ Detected p4d.24xlarge with 8× A100 GPUs!${NC}"
            echo -e "${GREEN}   Training will complete in ~45 minutes${NC}"
        fi
    else
        TRAINING_MODE="train"
        echo -e "${GREEN}Using single GPU training${NC}"
    fi
    
    # Build training command with auto-resume enabled
    TRAINING_CMD="python main.py $TRAINING_MODE \
        --use-ffcv \
        --ffcv-dir $FFCV_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
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
        --checkpoint-dir $CHECKPOINT_DIR \
        --log-dir $LOG_DIR \
        --checkpoint-interval 5 \
        --auto-resume \
        --target-accuracy $TARGET_ACCURACY \
        --num-workers 8"
    
    echo -e "${BLUE}Starting training with command:${NC}"
    echo "$TRAINING_CMD"
    echo ""
    
    # Create checkpoint directory if it doesn't exist
    mkdir -p $CHECKPOINT_DIR
    mkdir -p $LOG_DIR
    
    # Start spot termination monitor in background
    monitor_spot_termination &
    MONITOR_PID=$!
    
    # Run training
    $TRAINING_CMD 2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"
    
    # Kill monitor when training finishes
    kill $MONITOR_PID 2>/dev/null || true
}

# Function to validate final model
validate_model() {
    if [ -f "$CHECKPOINT_DIR/best_model.pt" ]; then
        echo -e "${GREEN}Validating best model...${NC}"
        python main.py validate --validate-only $CHECKPOINT_DIR/best_model.pt
    fi
}

# Main execution
main() {
    echo "Starting at: $(date)"
    echo ""
    
    # Setup signal handlers
    setup_signal_handlers
    
    # Mount data volume
    mount_data_volume
    
    # Activate PyTorch
    activate_pytorch
    
    # Check existing progress
    check_existing_progress
    
    # Verify GPU availability
    echo -e "${GREEN}Checking GPUs...${NC}"
    nvidia-smi
    echo ""
    
    # Run training
    run_training
    
    # Validate if training completed
    validate_model
    
    echo ""
    echo -e "${GREEN}=================================================="
    echo "Training completed at: $(date)"
    echo "=================================================="
    echo -e "${NC}"
}

# Run main function
main "$@"
