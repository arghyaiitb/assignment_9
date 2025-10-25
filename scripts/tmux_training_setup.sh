#!/bin/bash
# Automated tmux setup for ResNet-50 ImageNet training
# Creates a perfect multi-pane environment for training and monitoring

set -e

# Configuration
SESSION_NAME=${SESSION_NAME:-resnet_training}
PROJECT_DIR=${PROJECT_DIR:-~/assignment_9}
FFCV_DIR=${FFCV_DIR:-/data/ffcv}
LOG_DIR=${LOG_DIR:-./logs}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=================================================="
echo "tmux Training Environment Setup"
echo "=================================================="
echo -e "${NC}"

# Check if using NVIDIA Deep Learning AMI
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    echo -e "${GREEN}NVIDIA Deep Learning AMI detected${NC}"
    source /opt/conda/etc/profile.d/conda.sh
    conda activate pytorch
    echo -e "${GREEN}PyTorch 2.8 environment activated${NC}"
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}tmux not found. Installing...${NC}"
    sudo apt-get update && sudo apt-get install -y tmux || \
    sudo yum install -y tmux
fi

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo -e "${YELLOW}Session '$SESSION_NAME' already exists.${NC}"
    read -p "Kill existing session and create new? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t $SESSION_NAME
    else
        echo "Attaching to existing session..."
        tmux attach -t $SESSION_NAME
        exit 0
    fi
fi

# Parse training command arguments
# Auto-detect and optimize for p4d.24xlarge (8× A100s)
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -eq 8 ] && nvidia-smi | grep -q "A100"; then
    # Optimized for p4d.24xlarge
    BATCH_SIZE=${1:-2048}  # 256 per GPU × 8
    EPOCHS=${2:-60}        # Converges faster
    echo -e "${GREEN}Detected p4d.24xlarge: Using optimized settings for 8× A100s${NC}"
else
    # Default for other instances
    BATCH_SIZE=${1:-1024}
    EPOCHS=${2:-90}
fi
USE_DISTRIBUTED=${3:-true}

# Determine training command
if [ "$USE_DISTRIBUTED" = "true" ] && [ $(nvidia-smi -L | wc -l) -gt 1 ]; then
    TRAINING_MODE="distributed"
else
    TRAINING_MODE="train"
fi

TRAINING_CMD="python main.py $TRAINING_MODE \\
    --use-ffcv \\
    --ffcv-dir $FFCV_DIR \\
    --batch-size $BATCH_SIZE \\
    --epochs $EPOCHS \\
    --progressive-resize \\
    --use-ema \\
    --compile \\
    --checkpoint-interval 5 \\
    --auto-resume"

echo -e "${GREEN}Creating tmux session: $SESSION_NAME${NC}"
echo "Training command: $TRAINING_MODE mode with batch_size=$BATCH_SIZE"
echo ""

# Create new session with training in main window
tmux new-session -d -s $SESSION_NAME -n training -c $PROJECT_DIR

# Split window into 4 panes:
# +-------------------+-------------------+
# |                   |                   |
# |     Training      |    GPU Monitor    |
# |      (Pane 0)     |     (Pane 1)      |
# |                   |                   |
# +-------------------+-------------------+
# |                   |                   |
# |     Logs Tail     |    System Stats   |
# |      (Pane 2)     |     (Pane 3)      |
# |                   |                   |
# +-------------------+-------------------+

# Split horizontally (creates pane 1 on right)
tmux split-window -h -t $SESSION_NAME:training -c $PROJECT_DIR

# Split both panes vertically
tmux split-window -v -t $SESSION_NAME:training.0 -c $PROJECT_DIR  # Creates pane 2 bottom-left
tmux split-window -v -t $SESSION_NAME:training.1 -c $PROJECT_DIR  # Creates pane 3 bottom-right

# Setup each pane
# Pane 0 (top-left): Main training
# Activate conda environment if using NVIDIA AMI
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    tmux send-keys -t $SESSION_NAME:training.0 "source /opt/conda/etc/profile.d/conda.sh && conda activate pytorch" C-m
fi
tmux send-keys -t $SESSION_NAME:training.0 "# Training Command (Press Enter to start)" C-m
tmux send-keys -t $SESSION_NAME:training.0 "echo 'Starting training in 5 seconds...'" C-m
tmux send-keys -t $SESSION_NAME:training.0 "sleep 5" C-m
tmux send-keys -t $SESSION_NAME:training.0 "$TRAINING_CMD"

# Pane 1 (top-right): GPU monitoring
tmux send-keys -t $SESSION_NAME:training.1 "watch -n 1 nvidia-smi" C-m

# Pane 2 (bottom-left): Log monitoring
tmux send-keys -t $SESSION_NAME:training.2 "# Waiting for logs..." C-m
tmux send-keys -t $SESSION_NAME:training.2 "sleep 10 && tail -f $LOG_DIR/*.log" C-m

# Pane 3 (bottom-right): System stats
tmux send-keys -t $SESSION_NAME:training.3 "htop || top" C-m

# Create additional window for checkpoints monitoring
tmux new-window -t $SESSION_NAME -n checkpoints -c $PROJECT_DIR
tmux send-keys -t $SESSION_NAME:checkpoints "watch -n 30 'ls -lah checkpoints/ | tail -20'" C-m

# Create window for manual commands
tmux new-window -t $SESSION_NAME -n shell -c $PROJECT_DIR
tmux send-keys -t $SESSION_NAME:shell "# Shell for manual commands" C-m
tmux send-keys -t $SESSION_NAME:shell "# Useful commands:" C-m
tmux send-keys -t $SESSION_NAME:shell "# - python main.py validate --validate-only checkpoints/best_model.pt" C-m
tmux send-keys -t $SESSION_NAME:shell "# - df -h /data  # Check disk space" C-m
tmux send-keys -t $SESSION_NAME:shell "# - ps aux | grep python  # Check processes" C-m

# Create monitoring dashboard window
tmux new-window -t $SESSION_NAME -n dashboard -c $PROJECT_DIR

# Split dashboard into 6 panes for comprehensive monitoring
tmux split-window -h -t $SESSION_NAME:dashboard
tmux split-window -v -t $SESSION_NAME:dashboard.0
tmux split-window -v -t $SESSION_NAME:dashboard.1
tmux split-window -h -t $SESSION_NAME:dashboard.2
tmux split-window -h -t $SESSION_NAME:dashboard.3

# Dashboard pane setup
tmux send-keys -t $SESSION_NAME:dashboard.0 "watch -n 5 'df -h | grep -E \"Filesystem|/$|/data\"'" C-m
tmux send-keys -t $SESSION_NAME:dashboard.1 "watch -n 2 'free -h'" C-m
tmux send-keys -t $SESSION_NAME:dashboard.2 "watch -n 5 'ls -lht logs/*.log | head -5'" C-m
tmux send-keys -t $SESSION_NAME:dashboard.3 "watch -n 10 'tail -n 3 logs/*.log | grep -E \"Loss|Acc|Epoch\"'" C-m
tmux send-keys -t $SESSION_NAME:dashboard.4 "watch -n 1 'sensors 2>/dev/null || echo No sensors available'" C-m
tmux send-keys -t $SESSION_NAME:dashboard.5 "iostat -x 2 2>/dev/null || vmstat 2" C-m

# Switch back to main training window
tmux select-window -t $SESSION_NAME:training

# Create helper script
cat > /tmp/tmux_training_help.txt << 'EOF'
================================================================================
tmux Training Environment Ready!
================================================================================

WINDOWS:
  1. training    - Main training with 4 panes (training, GPU, logs, system)
  2. checkpoints - Monitor checkpoint files
  3. shell       - Manual commands
  4. dashboard   - System monitoring dashboard

NAVIGATION:
  Switch windows : Ctrl+B, then 0-3
  Switch panes   : Ctrl+B, then arrow keys
  Scroll mode    : Ctrl+B, then [ (q to exit)
  Detach session : Ctrl+B, then D

COMMANDS:
  Attach to session : tmux attach -t resnet_training
  List sessions     : tmux ls
  Kill session      : tmux kill-session -t resnet_training
  
START TRAINING:
  Go to pane 0 and press ENTER to start training

================================================================================
EOF

# Display help
clear
cat /tmp/tmux_training_help.txt

echo ""
echo -e "${GREEN}✅ tmux environment created successfully!${NC}"
echo ""
echo "Attaching to session in 3 seconds..."
echo "(Press Ctrl+C to cancel and attach manually later)"
sleep 3

# Attach to the session
tmux attach -t $SESSION_NAME
