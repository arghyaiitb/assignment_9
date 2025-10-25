# Phase 1: CPU Data Preparation (1 Hour, $0.62)

> **Purpose**: Download ImageNet, convert to FFCV format, prepare everything for GPU training
> **Time**: 1 hour with c5a.4xlarge (vs 5 hours with t3.large)
> **Cost**: $0.62 total
> **Output**: 400GB EBS volume with all data ready for training

## 📋 Prerequisites

1. **AWS Account** with EC2 access
2. **HuggingFace Account** with ImageNet access approved
3. **AWS CLI** configured locally
4. **Your project code** in a Git repository

## 🚀 Quick Start - Complete Setup in 1 Hour

### Step 1: Create EBS Volume (One-Time)

```bash
# Create 400GB EBS volume for data storage
aws ec2 create-volume \
  --region us-east-1 \
  --size 400 \
  --volume-type gp3 \
  --availability-zone us-east-1a \
  --iops 10000 \
  --throughput 250 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=imagenet-data}]'

# Note the VolumeId - you'll need this!
export EBS_VOLUME_ID=vol-xxxxxxxxxxxxx  # Replace with your actual ID
```

### Step 2: Launch c5a.4xlarge Instance (CPU-Optimized)

```bash
# Set your configuration
export KEY_NAME=your-key-pair        # Your SSH key
export SUBNET_ID=subnet-xxxxx       # Must be in us-east-1a (same as EBS!)
export SECURITY_GROUP=sg-xxxxx      # Your security group

# Launch c5a.4xlarge with Ubuntu 22.04 (CPU-optimized AMI)
INSTANCE_ID=$(aws ec2 run-instances \
  --region us-east-1 \
  --image-id ami-0866a3c8686eaeeba \  # Ubuntu 22.04 LTS for us-east-1
  --instance-type c5a.4xlarge \
  --key-name $KEY_NAME \
  --subnet-id $SUBNET_ID \
  --security-group-ids $SECURITY_GROUP \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","Iops":10000}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=imagenet-data-prep}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance ID: $INSTANCE_ID"
```

### Step 3: Attach EBS and Get IP

```bash
# Wait for instance to start
aws ec2 wait instance-running --region us-east-1 --instance-ids $INSTANCE_ID

# Attach EBS volume
aws ec2 attach-volume \
  --region us-east-1 \
  --volume-id $EBS_VOLUME_ID \
  --instance-id $INSTANCE_ID \
  --device /dev/sdf

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --region us-east-1 \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "==============================================="
echo "Instance ready! SSH with:"
echo "ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo "==============================================="
```

### Step 4: SSH and Run Automated Setup Script

```bash
# SSH into the instance
ssh -i your-key.pem ubuntu@$PUBLIC_IP
```

Once logged in, run these commands:

```bash
# 1. Quick system setup
sudo apt-get update && sudo apt-get install -y git tmux htop

# 2. Mount the EBS volume
# First, find the device name (Nitro instances use NVMe naming)
lsblk  # Look for your 400GB volume (likely /dev/nvme1n1)

# Format and mount (replace nvme1n1 with your actual device)
sudo mkfs -t xfs /dev/nvme1n1  # Only first time! Skip if already formatted
sudo mkdir /data
sudo mount -o noatime,nodiratime /dev/nvme1n1 /data
sudo chown ubuntu:ubuntu /data

# 3. Clone your project (or download the script)
cd /data
git clone https://github.com/arghyaiitb/assignment_9.git
cd assignment_9

# 4. Run the automated setup script
bash scripts/ebs_data_prep.sh
```

### Step 5: Manual Steps After Script

The script will guide you, but here are the key commands:

```bash
# 1. Login to HuggingFace (required for ImageNet)
huggingface-cli login
# Enter your token when prompted

# 2. Start tmux for persistent session
tmux new -s ffcv

# 3. Convert ImageNet to FFCV (this is the main task - takes ~1 hour)
cd /data/assignment_9
python main.py convert-ffcv --ffcv-dir /data/ffcv

# Monitor progress in another tmux pane (Ctrl-B %)
watch -n 5 'df -h /data; ls -lah /data/ffcv/'

# 4. When complete, verify the files
ls -lah /data/ffcv/
# Should show:
# train.ffcv (~140GB)
# val.ffcv (~6GB)

# 5. Create completion marker
echo "Data ready: $(date)" > /data/DATA_READY.txt

# 6. Exit tmux
exit  # or Ctrl-D
```

### Step 6: Cleanup - IMPORTANT! (Save Money)

```bash
# Exit SSH
exit

# From your local machine:
# Stop the instance
aws ec2 stop-instances --region us-east-1 --instance-ids $INSTANCE_ID
aws ec2 wait instance-stopped --region us-east-1 --instance-ids $INSTANCE_ID

# Detach the EBS (keeps your data safe)
aws ec2 detach-volume --region us-east-1 --volume-id $EBS_VOLUME_ID

# Terminate the instance (stop paying!)
aws ec2 terminate-instances --region us-east-1 --instance-ids $INSTANCE_ID

echo "✅ Phase 1 complete! Your data is on EBS: $EBS_VOLUME_ID"
```

## 📊 Why c5a.4xlarge?

| Instance | vCPUs | Time | Cost | Why Choose? |
|----------|-------|------|------|-------------|
| t3.large | 2 | 5 hrs | $0.40 | ❌ Too slow, CPU throttles |
| t3.2xlarge | 8 | 2 hrs | $0.67 | ❌ Burstable = unreliable |
| c5.2xlarge | 8 | 1.5 hrs | $0.51 | ✅ Good option |
| **c5a.4xlarge** | **16** | **1 hr** | **$0.62** | **🏆 BEST - 5x faster!** |

**c5a.4xlarge** gives you:
- 16 vCPUs for parallel FFCV conversion
- Consistent performance (no throttling)
- 10 Gbps network for fast downloads
- Completes in 1 hour instead of 5
- Only $0.22 more than slow option

## 🔧 What the Script Does

The `scripts/ebs_data_prep.sh` script automates:

1. **EBS Setup**: Detects, formats, and mounts your EBS volume
2. **Python Environment**: Installs Python 3.10 and pip
3. **Dependencies**: Installs PyTorch (CPU), FFCV, datasets, etc.
4. **Directory Structure**: Creates organized folders for data
5. **Environment Variables**: Sets up paths for FFCV and HuggingFace
6. **Optimization**: Configures for maximum CPU utilization

## 📝 Alternative: Manual Setup

If you prefer manual setup instead of the script:

```bash
# System packages (including OpenCV dependencies for FFCV)
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    tmux \
    htop \
    libopencv-dev \
    python3-opencv \
    pkg-config \
    libturbojpeg-dev \
    libopenjp2-7-dev \
    libjpeg-dev

# Python packages
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install datasets huggingface_hub numpy tqdm
pip3 install ffcv  # Install after OpenCV system libs
pip3 install albumentations opencv-python wandb

# Environment setup
export FFCV_DIR=/data/ffcv
export HF_HOME=/data/huggingface_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export NUMBA_NUM_THREADS=16

# Create directories
mkdir -p $FFCV_DIR $HF_HOME
```

## ⚡ Speed Optimizations

The setup is optimized for speed:

1. **16 CPU cores**: c5a.4xlarge processes data in parallel
2. **XFS filesystem**: Faster than ext4 for large files
3. **noatime mount**: Skips unnecessary file access updates
4. **10K IOPS EBS**: Fast disk I/O
5. **HF_TRANSFER**: Accelerated HuggingFace downloads
6. **16 FFCV workers**: Maximum parallel conversion

## 🆘 Troubleshooting

### "No such file or directory" for /dev/xvdf
- **Cause**: Nitro instances (c5a, c5, m5, etc.) use NVMe device naming
- **Solution**: Run `lsblk` to find your device (usually `/dev/nvme1n1`)
- **Fix**: Use `export EBS_DEVICE=/dev/nvme1n1` before running the script

### "No space left on device"
- Check: `df -h /data`
- Solution: Ensure 400GB EBS is attached and mounted

### "FFCV conversion stuck"
- Check CPU usage: `htop`
- Solution: Should show ~100% CPU usage across all cores
- If not, restart with more workers

### "Could not find required package: opencv" (FFCV install error)
- **Cause**: Missing OpenCV system libraries
- **Solution**: Install system dependencies first:
  ```bash
  sudo apt-get install -y libopencv-dev python3-opencv pkg-config libturbojpeg-dev libopenjp2-7-dev libjpeg-dev
  pip install ffcv
  ```

### "HuggingFace authentication failed"
- Run: `huggingface-cli login`
- Make sure you have ImageNet access approved

### "Instance terminated unexpectedly"
- Your data is safe on EBS!
- Just launch a new instance and reattach the EBS
- Resume from where you left off

## 💡 Pro Tips

1. **Test First**: Try with partial dataset (5 minutes):
   ```bash
   python main.py convert-ffcv --partial-dataset --ffcv-dir /data/ffcv_test
   ```

2. **Monitor Progress**: Use tmux split panes:
   - Pane 1: Run conversion
   - Pane 2: `watch -n 5 'ls -lah /data/ffcv/'`
   - Pane 3: `htop` to monitor CPU

3. **Budget Control**: Set a CloudWatch alarm to auto-stop after 2 hours

## ✅ Completion Checklist

- [ ] Created 400GB EBS volume
- [ ] Launched c5a.4xlarge instance
- [ ] Attached EBS and mounted at /data
- [ ] Installed dependencies with script
- [ ] Logged into HuggingFace
- [ ] Converted ImageNet to FFCV format
- [ ] Verified train.ffcv and val.ffcv exist
- [ ] Stopped and terminated instance
- [ ] Detached EBS volume

## 📚 Next Step

Once Phase 1 is complete, your EBS volume contains everything needed for training.
Proceed to **[AWS_PHASE2_GPU_TRAINING.md](AWS_PHASE2_GPU_TRAINING.md)** to start GPU training!

---

**Remember**: The entire Phase 1 takes just 1 hour with c5a.4xlarge. Don't waste 5 hours with a slow instance!
