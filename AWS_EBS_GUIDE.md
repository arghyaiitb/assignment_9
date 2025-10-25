# AWS EBS Strategy for Cost-Effective ImageNet Training

> **💰 Cost Savings**: Download/prepare data on a $0.10/hour t3.large, then train on GPU instance. Save $30+ on data preparation!

> **🔧 Using NVIDIA Deep Learning AMI**: This guide is optimized for the **NVIDIA Deep Learning AMI with PyTorch 2.8** which has CUDA, cuDNN, and PyTorch pre-installed, saving setup time.

## Overview

This guide shows how to:
1. Create an EBS volume for persistent storage
2. Download & prepare ImageNet on a cheap EC2 instance (~$0.10/hour)
3. Move the EBS to a GPU instance for training (~$3-12/hour)
4. Reuse the EBS for multiple training runs

**Total Cost Estimate:**
- Data preparation: ~$0.50 (5 hours on t3.large)
- Training: ~$15 (1.5 hours on p3.8xlarge spot)
- EBS Storage: ~$13/month for 400GB
- **Total: ~$15.50 one-time + $13/month storage**

---

## 🎯 NVIDIA Deep Learning AMI Advantages

Using the **NVIDIA Deep Learning AMI (PyTorch 2.8)** provides:
- ✅ **PyTorch 2.8** pre-installed with CUDA 12.x support
- ✅ **NVIDIA drivers** and **cuDNN** already configured
- ✅ **Conda environments** for easy package management
- ✅ **Common ML libraries** pre-installed (numpy, scipy, etc.)
- ✅ **Optimized for GPU** training out of the box
- ✅ **No setup time** - start training immediately

**AMI ID (us-east-1)**: `ami-0e3b9734bf8e3d64b` (verify latest version)

---

## Phase 1: Create EBS Volume

### Step 1.1: Create EBS Volume (One-Time)
```bash
# Create a 250GB GP3 SSD in your preferred region (e.g., us-east-1)
aws ec2 create-volume \
  --region us-east-1 \
  --size 400 \
  --volume-type gp3 \
  --availability-zone us-east-1a \
  --iops 10000 \
  --throughput 250 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=imagenet-data}]'

# Note the VolumeId (e.g., vol-0123456789abcdef)
# Save this ID - you'll need it multiple times!
export EBS_VOLUME_ID=vol-0468159ea0a0112aa  # Replace with your actual ID
```

### Step 1.2: Wait for Volume Creation
```bash
# Check volume status
aws ec2 describe-volumes --region us-east-1 --volume-ids $EBS_VOLUME_ID

# Wait until State shows "available"
```

---

## Phase 2: Prepare Data on Cheap Instance

### Step 2.1: Launch t3.large Instance (Data Preparation)
```bash
# Launch a cheap instance in the SAME availability zone as your EBS
# Using NVIDIA Deep Learning AMI (PyTorch 2.8) for consistency
aws ec2 run-instances \
    --region us-east-1 \
    --image-id ami-0e3b9734bf8e3d64b \  # NVIDIA Deep Learning AMI (PyTorch 2.8) - check for latest
    --instance-type t3.large \
    --key-name your-key-pair \
    --subnet-id subnet-xxxxx \  # Subnet in us-east-1a (SAME as EBS!)
    --security-group-ids sg-xxxxx \
    --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":50}}]" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=data-prep-instance}]' \
    --user-data file://data_prep_script.sh

# Note the InstanceId
export PREP_INSTANCE_ID=i-027dd5fb0f01b62b4  # Replace with actual
```

### Step 2.2: Create Data Preparation Script
```bash
cat > data_prep_script.sh << 'EOF'
#!/bin/bash
# This script runs automatically when instance starts
# NVIDIA Deep Learning AMI already has PyTorch 2.8 installed!

# Update system (Ubuntu-based)
sudo apt-get update -y
sudo apt-get install -y git htop tmux

# Activate the pre-installed PyTorch environment
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# Install additional required packages (PyTorch already installed)
pip install datasets huggingface_hub ffcv tqdm

# Create status file
echo "Instance ready. PyTorch 2.8 environment active." > /home/ubuntu/status.txt
echo "Please attach EBS and run manual setup." >> /home/ubuntu/status.txt
EOF
```

### Step 2.3: Attach EBS to Data Prep Instance
```bash
# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $PREP_INSTANCE_ID

# Attach the EBS volume
aws ec2 attach-volume \
    --volume-id $EBS_VOLUME_ID \
    --instance-id $PREP_INSTANCE_ID \
    --device /dev/sdf

# Wait for attachment
aws ec2 wait volume-in-use --volume-ids $EBS_VOLUME_ID
```

### Step 2.4: SSH into Instance and Mount EBS
```bash
# SSH into the instance (ubuntu user for Deep Learning AMI)
ssh -i your-key.pem ubuntu@<instance-public-ip>

# Once logged in, activate PyTorch environment
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# Check PyTorch version
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Mount EBS commands:
# Check device name (might be /dev/xvdf or /dev/nvme1n1)
lsblk

# Format EBS (ONLY first time!)
sudo mkfs -t xfs /dev/xvdf  # Or /dev/nvme1n1 if using Nitro instance

# Create mount point
sudo mkdir /data

# Mount the volume
sudo mount /dev/xvdf /data

# Change ownership
sudo chown ubuntu:ubuntu /data

# Make mount permanent
echo '/dev/xvdf /data xfs defaults,nofail 0 2' | sudo tee -a /etc/fstab

# Verify
df -h /data  # Should show ~400GB available (you increased to 400GB)
```

### Step 2.5: Download and Prepare ImageNet
```bash
# Ensure PyTorch environment is active
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# Clone the project
cd /data
git clone https://github.com/yourusername/resnet50-imagenet.git  # Replace with your repo
cd resnet50-imagenet

# Install additional requirements (PyTorch already installed)
# Skip torch installation since we have PyTorch 2.8
pip install datasets huggingface_hub ffcv numpy tqdm albumentations opencv-python

# Authenticate with HuggingFace (if not already done)
huggingface-cli login

# IMPORTANT: Set data directories to EBS
export FFCV_DIR=/data/ffcv
export HF_HOME=/data/huggingface_cache
export HF_DATASETS_CACHE=/data/huggingface_cache/datasets

# Create directories
mkdir -p $FFCV_DIR $HF_HOME

# Download and convert FULL ImageNet to FFCV (This takes 3-5 hours)
python main.py convert-ffcv --ffcv-dir $FFCV_DIR

# Optional: Also create partial dataset for testing (recommended first!)
python main.py convert-ffcv --partial-dataset --partial-size 5000 --ffcv-dir ${FFCV_DIR}_partial

# Verify data
ls -lah $FFCV_DIR/
# Should show:
# train.ffcv (~140GB)
# val.ffcv (~6GB)

# Create a marker file with environment info
echo "Data preparation complete: $(date)" > /data/READY.txt
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')" >> /data/READY.txt
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')" >> /data/READY.txt
```

### Step 2.6: Detach EBS and Terminate Cheap Instance
```bash
# Exit SSH
exit

# Stop the instance
aws ec2 stop-instances --instance-ids $PREP_INSTANCE_ID

# Wait for instance to stop
aws ec2 wait instance-stopped --instance-ids $PREP_INSTANCE_ID

# Detach the EBS volume
aws ec2 detach-volume --volume-id $EBS_VOLUME_ID

# Wait for detachment
aws ec2 wait volume-available --volume-ids $EBS_VOLUME_ID

# Terminate the cheap instance (save money!)
aws ec2 terminate-instances --instance-ids $PREP_INSTANCE_ID

echo "✅ Data preparation complete! EBS volume $EBS_VOLUME_ID is ready for training."
```

---

## Phase 3: Training on GPU Instance

### Step 3.1: Launch GPU Instance (p3.8xlarge or p4d.24xlarge)
```bash
# For p3.8xlarge (4x V100, ~$12/hour spot)
aws ec2 run-instances \
    --region us-east-1 \
    --image-id ami-0e3b9734bf8e3d64b \  # NVIDIA Deep Learning AMI (PyTorch 2.8)
    --instance-type p3.8xlarge \
    --key-name your-key-pair \
    --subnet-id subnet-xxxxx \  # SAME availability zone as EBS! (us-east-1a)
    --security-group-ids sg-xxxxx \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"4.00"}}' \
    --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":100}}]" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=training-gpu}]' \
    --user-data file://training_script.sh

# Note the InstanceId
export GPU_INSTANCE_ID=i-9876543210fedcba  # Replace with actual
```

### Step 3.2: Create Training Script
```bash
cat > training_script.sh << 'EOF'
#!/bin/bash
# Auto-setup for GPU training instance with PyTorch 2.8 pre-installed

# Activate PyTorch environment
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# Clone project (will be on root volume)
cd /home/ubuntu
git clone https://github.com/yourusername/resnet50-imagenet.git
cd resnet50-imagenet

# Install additional requirements (PyTorch already installed)
pip install datasets huggingface_hub ffcv numpy tqdm albumentations opencv-python wandb

# Create mount point for data
sudo mkdir -p /data

# Test GPU availability
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"

echo "Instance ready with PyTorch 2.8. Attach EBS and start training." > /home/ubuntu/READY.txt
EOF
```

### Step 3.3: Attach EBS to GPU Instance
```bash
# Wait for GPU instance to be running
aws ec2 wait instance-running --instance-ids $GPU_INSTANCE_ID

# Attach the EBS volume with data
aws ec2 attach-volume \
    --volume-id $EBS_VOLUME_ID \
    --instance-id $GPU_INSTANCE_ID \
    --device /dev/sdf

# Wait for attachment
aws ec2 wait volume-in-use --volume-ids $EBS_VOLUME_ID
```

### Step 3.4: SSH and Start Training
```bash
# SSH into GPU instance
ssh -i your-key.pem ubuntu@<gpu-instance-public-ip>

# Activate PyTorch 2.8 environment
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# Verify PyTorch and GPU
python -c "import torch; print(f'PyTorch {torch.__version__}, GPUs: {torch.cuda.device_count()}')"

# tmux should already be installed, if not:
# sudo apt-get update && sudo apt-get install -y tmux

# Start tmux session for persistent training
tmux new -s training

# === Inside tmux session ===

# Make sure conda environment is active in tmux
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# Mount the EBS with prepared data
sudo mount /dev/xvdf /data  # Or /dev/nvme1n1 for Nitro instances
sudo chown ubuntu:ubuntu /data

# Verify data is there
ls -lah /data/ffcv/
# Should show train.ffcv and val.ffcv

# Navigate to project
cd ~/resnet50-imagenet

# Split tmux window (Ctrl+B, then %)
# Left pane: Start training
python main.py distributed \
    --use-ffcv \
    --ffcv-dir /data/ffcv \
    --batch-size 2048 \
    --epochs 100 \
    --progressive-resize \
    --use-ema \
    --compile \
    --checkpoint-interval 10 \
    --budget-hours 1.5

# Right pane: Monitor GPU (Ctrl+B, arrow to switch)
watch -n 1 nvidia-smi

# Create new window for logs (Ctrl+B, then C)
tail -f logs/train_*.log

# Detach from tmux (Ctrl+B, then D)
# Training continues even if SSH disconnects!

# === Back in SSH (not in tmux) ===

# You can now safely disconnect
exit

# Later, reconnect and check progress
ssh ubuntu@<gpu-instance-public-ip>
tmux attach -t training
```

#### Alternative: Multiple tmux Sessions for Monitoring
```bash
# Session 1: Training
tmux new -s training -d "cd ~/resnet50-imagenet && python main.py distributed --use-ffcv --ffcv-dir /data/ffcv"

# Session 2: GPU monitoring  
tmux new -s gpu -d "watch -n 1 nvidia-smi"

# Session 3: Logs
tmux new -s logs -d "tail -f ~/resnet50-imagenet/logs/train_*.log"

# View all sessions
tmux ls

# Attach to specific session
tmux attach -t training  # or gpu, or logs
```

### Step 3.5: Save Results and Clean Up
```bash
# After training completes, copy checkpoints to EBS
cp -r checkpoints/ /data/checkpoints_$(date +%Y%m%d_%H%M%S)/

# Exit SSH
exit

# Stop GPU instance
aws ec2 stop-instances --instance-ids $GPU_INSTANCE_ID

# Detach EBS
aws ec2 detach-volume --volume-id $EBS_VOLUME_ID

# Terminate GPU instance to stop billing
aws ec2 terminate-instances --instance-ids $GPU_INSTANCE_ID
```

---

## Phase 4: Reusing the EBS for Future Training

### Option A: Resume Training
```bash
# Launch new GPU instance
aws ec2 run-instances ... # Same as Step 3.1

# Attach existing EBS
aws ec2 attach-volume \
    --volume-id $EBS_VOLUME_ID \
    --instance-id $NEW_GPU_INSTANCE_ID \
    --device /dev/sdf

# SSH and mount
ssh ubuntu@<new-instance-ip>
sudo mount /dev/xvdf /data

# Resume from checkpoint
python main.py train \
    --use-ffcv \
    --ffcv-dir /data/ffcv \
    --resume /data/checkpoints_20240315_120000/checkpoint_epoch_50.pt \
    --epochs 100
```

### Option B: New Experiments
```bash
# The data is already prepared! Just:
# 1. Launch GPU instance
# 2. Attach EBS
# 3. Start training with different hyperparameters

# No need to download/convert data again!
# Saves 3-5 hours and ~150GB download each time
```

---

## 💰 Cost Breakdown

### One-Time Costs:
| Component | Time | Rate | Cost |
|-----------|------|------|------|
| t3.large (data prep) | 5 hrs | $0.10/hr | $0.50 |
| p3.8xlarge spot (training) | 1.5 hrs | $10/hr | $15.00 |
| Data transfer | 150GB | $0.00 | $0.00 |
| **Total One-Time** | | | **$15.50** |

### Ongoing Storage:
| Component | Size | Rate | Monthly Cost |
|-----------|------|------|--------------|
| EBS GP3 SSD | 400GB | $0.032/GB-month | $12.80 |
| Snapshots (optional) | 400GB | $0.05/GB-month | $20.00 |

### Per Training Run (Reusing EBS):
| Component | Time | Rate | Cost |
|-----------|------|------|------|
| p3.8xlarge spot | 1.5 hrs | $10/hr | $15.00 |
| **Total Per Run** | | | **$15.00** |

---

## 🔒 Important Security & Best Practices

### 1. EBS Snapshots for Backup
```bash
# Create snapshot after data preparation
aws ec2 create-snapshot \
    --volume-id $EBS_VOLUME_ID \
    --description "ImageNet FFCV data ready for training" \
    --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=imagenet-ffcv-backup}]'
```

### 2. Use Same Availability Zone
- **CRITICAL**: EBS volumes can only be attached to instances in the SAME availability zone
- Always specify `--availability-zone` when creating volumes
- Always specify `--subnet-id` for the same AZ when launching instances

### 3. Encryption (Optional but Recommended)
```bash
# Create encrypted EBS
aws ec2 create-volume \
    --encrypted \
    --kms-key-id alias/aws/ebs \
    --size 250 \
    --volume-type gp3 \
    --availability-zone us-east-1a
```

### 4. Auto-Stop for Cost Control
```bash
# Set CloudWatch alarm to stop instance after 2 hours
aws cloudwatch put-metric-alarm \
    --alarm-name "gpu-training-timeout" \
    --alarm-actions arn:aws:automate:region:ec2:stop \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 1 \
    --comparison-operator LessThanThreshold \
    --evaluation-periods 24  # 2 hours
```

---

## 🔧 Troubleshooting PyTorch 2.8 AMI

### Common Issues and Solutions:

**Conda environment not activated:**
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch
```

**PyTorch not finding GPUs:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
# If false, restart instance or check nvidia-smi
```

**Permission issues with conda:**
```bash
# Run as ubuntu user, not root
sudo chown -R ubuntu:ubuntu /opt/conda/envs/pytorch
```

**Package conflicts:**
```bash
# Create a fresh environment if needed
conda create -n training python=3.10
conda activate training
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## 🚀 Quick Reference Commands

```bash
# Variables to set once
export EBS_VOLUME_ID=vol-0468159ea0a0112aa  # Your actual volume ID
export PREP_INSTANCE_ID=i-xxxxx  
export GPU_INSTANCE_ID=i-xxxxx

# Attach EBS to instance
aws ec2 attach-volume --volume-id $EBS_VOLUME_ID --instance-id $INSTANCE_ID --device /dev/sdf

# Detach EBS from instance
aws ec2 stop-instances --instance-ids $INSTANCE_ID
aws ec2 detach-volume --volume-id $EBS_VOLUME_ID

# Check EBS status
aws ec2 describe-volumes --volume-ids $EBS_VOLUME_ID --query 'Volumes[0].State'

# List all your EBS volumes
aws ec2 describe-volumes --query 'Volumes[*].[VolumeId,Size,State,Tags[?Key==`Name`].Value|[0]]' --output table

# Delete EBS when completely done (CAREFUL!)
aws ec2 delete-volume --volume-id $EBS_VOLUME_ID
```

---

## 📝 Summary

**Advantages of this approach:**
1. ✅ Data preparation on cheap instance ($0.50 vs $50)
2. ✅ Reusable data - prepare once, train many times
3. ✅ Persistent storage - survives instance termination
4. ✅ Snapshot backups - never lose prepared data
5. ✅ Flexibility - attach to any GPU instance type

**Key Points:**
- Use **NVIDIA Deep Learning AMI** with PyTorch 2.8 pre-installed
- Prepare data ONCE on t3.large (~$0.50)  
- Store on 400GB EBS (~$13/month)
- Train MULTIPLE times on GPU (~$15 each)
- Total first run: ~$15.50
- Subsequent runs: ~$15.00 each
- No setup overhead - PyTorch ready to go!

This strategy saves significant money and time for multiple training runs!
