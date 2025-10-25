# Phase 2: GPU Training (1.5 Hours, $5.25)

> **Prerequisites**: Completed Phase 1 with EBS volume containing FFCV data
> **Time**: 1.5 hours on p3.8xlarge
> **Cost**: ~$5.25 with spot instances
> **Result**: ResNet-50 trained to 78% accuracy

## 🚀 Quick Start - Train in 90 Minutes

### Step 1: Launch GPU Instance with Spot Pricing

```bash
# Use same configuration as Phase 1
export KEY_NAME=your-key-pair
export SUBNET_ID=subnet-xxxxx       # MUST be same AZ as your EBS!
export SECURITY_GROUP=sg-xxxxx
export EBS_VOLUME_ID=vol-xxxxxxxxxxxxx  # Your data volume from Phase 1

# Launch p3.8xlarge with NVIDIA Deep Learning AMI (has PyTorch 2.8 + CUDA)
GPU_INSTANCE_ID=$(aws ec2 run-instances \
  --region us-east-1 \
  --image-id ami-0e3b9734bf8e3d64b \  # NVIDIA Deep Learning AMI
  --instance-type p3.8xlarge \
  --key-name $KEY_NAME \
  --subnet-id $SUBNET_ID \
  --security-group-ids $SECURITY_GROUP \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"4.00","SpotInstanceType":"one-time"}}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=resnet50-training}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "GPU Instance: $GPU_INSTANCE_ID"
```

### Step 2: Attach EBS and Connect

```bash
# Wait for instance
aws ec2 wait instance-running --region us-east-1 --instance-ids $GPU_INSTANCE_ID

# Attach your data EBS
aws ec2 attach-volume \
  --region us-east-1 \
  --volume-id $EBS_VOLUME_ID \
  --instance-id $GPU_INSTANCE_ID \
  --device /dev/sdf

# Get IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --region us-east-1 \
  --instance-ids $GPU_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "==============================================="
echo "GPU Instance ready! SSH with:"
echo "ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
echo "==============================================="
```

### Step 3: SSH and Start Training

```bash
# SSH into GPU instance
ssh -i your-key.pem ubuntu@$PUBLIC_IP
```

Once logged in, run the automated training script:

```bash
# Run the training setup script
bash /data/resnet50-imagenet/scripts/ebs_training.sh
```

Or manually:

```bash
# 1. Activate PyTorch environment (pre-installed on NVIDIA AMI)
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

# 2. Mount your data
sudo mount /dev/xvdf /data  # or /dev/nvme1n1 for Nitro
sudo chown ubuntu:ubuntu /data

# 3. Verify data and GPUs
ls -lah /data/ffcv/  # Should show train.ffcv and val.ffcv
nvidia-smi  # Should show 4x V100 GPUs

# 4. Navigate to project
cd /data/resnet50-imagenet

# 5. Start training with tmux monitoring
bash scripts/tmux_training_setup.sh
```

### Step 4: Training Command

The `tmux_training_setup.sh` script will set up multiple panes and start training:

```bash
# Main training command (runs automatically in tmux)
python main.py distributed \
  --use-ffcv \
  --ffcv-dir /data/ffcv \
  --batch-size 1024 \
  --epochs 90 \
  --lr 1.6 \
  --warmup-epochs 5 \
  --progressive-resize \
  --use-ema \
  --compile \
  --mixed-precision \
  --checkpoint-dir /data/checkpoints \
  --checkpoint-interval 10 \
  --budget-hours 1.5
```

### Step 5: Monitor Training

The tmux setup creates these panes for you:
- **Window 1**: Training progress
- **Window 2**: GPU monitoring (`nvidia-smi`)
- **Window 3**: Training logs
- **Window 4**: System stats
- **Window 5**: Checkpoint monitoring

Navigate between windows:
- `Ctrl-B` then `0-5`: Switch windows
- `Ctrl-B` then `D`: Detach (training continues)
- `tmux attach -t training`: Reattach

### Step 6: After Training Completes

```bash
# Training will save best model automatically
# Check results
ls -lah /data/checkpoints/
cat /data/checkpoints/training_summary.txt

# Exit tmux
exit  # From tmux
exit  # From SSH
```

### Step 7: Cleanup - CRITICAL! (Stop Billing)

```bash
# From your local machine:
# Stop the GPU instance
aws ec2 stop-instances --region us-east-1 --instance-ids $GPU_INSTANCE_ID
aws ec2 wait instance-stopped --region us-east-1 --instance-ids $GPU_INSTANCE_ID

# Detach EBS (preserves your trained model)
aws ec2 detach-volume --region us-east-1 --volume-id $EBS_VOLUME_ID

# Terminate GPU instance
aws ec2 terminate-instances --region us-east-1 --instance-ids $GPU_INSTANCE_ID

echo "✅ Training complete! Model saved on EBS: $EBS_VOLUME_ID"
```

## 📊 Training Configuration Explained

### Hyperparameters for 4× V100 GPUs

```python
--batch-size 1024        # 256 per GPU × 4 GPUs
--lr 1.6                 # Scaled linearly with batch size
--epochs 90              # Target 78% accuracy
--warmup-epochs 5        # Gradual warmup
--progressive-resize     # 160→192→224 resolution
--use-ema               # Smoother convergence
--compile               # PyTorch 2.0 optimization
--mixed-precision       # FP16 for speed
```

### Progressive Training Schedule

| Phase | Epochs | Resolution | Batch Size | Est. Time |
|-------|--------|------------|------------|-----------|
| Warmup | 0-5 | 160×160 | 2048 | 5 min |
| Stage 1 | 5-30 | 160×160 | 2048 | 20 min |
| Stage 2 | 30-60 | 192×192 | 1536 | 30 min |
| Stage 3 | 60-90 | 224×224 | 1024 | 35 min |
| **Total** | **90** | | | **~90 min** |

## 📈 Expected Accuracy Progress

| Checkpoint | Top-1 | Top-5 | Time |
|------------|-------|-------|------|
| Epoch 10 | ~45% | ~70% | 10 min |
| Epoch 30 | ~65% | ~85% | 30 min |
| Epoch 60 | ~73% | ~91% | 60 min |
| **Epoch 90** | **~78%** | **~94%** | **90 min** |

## 🔧 What the Scripts Do

### `scripts/ebs_training.sh`
- Mounts EBS with your data
- Activates PyTorch environment
- Verifies GPU availability
- Sets up training directories
- Provides training commands

### `scripts/tmux_training_setup.sh`
- Creates multi-pane tmux layout
- Starts distributed training
- Sets up GPU monitoring
- Tracks logs and checkpoints
- Provides real-time dashboard

## 💰 Cost Optimization

### GPU Instance Options

| Instance | GPUs | Spot $/hr | Time | Total Cost | Recommendation |
|----------|------|-----------|------|------------|----------------|
| **p3.8xlarge** | 4× V100 | ~$3.50 | 1.5 hrs | **$5.25** | 🏆 Best value |
| g5.12xlarge | 4× A10G | ~$2.00 | 2 hrs | $4.00 | Good alternative |
| p3.2xlarge | 1× V100 | ~$0.90 | 6 hrs | $5.40 | Too slow |

### Why Spot Instances?
- 70% cheaper than on-demand
- Perfect for training jobs
- Just save checkpoints frequently

## 🆘 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 512  # Instead of 1024
```

### Training Too Slow
```bash
# Check GPU utilization
nvidia-smi  # Should show >95% usage

# Ensure FFCV is working
ls /data/ffcv/  # Must have .ffcv files
```

### Spot Instance Terminated
```bash
# Resume from checkpoint
--resume /data/checkpoints/checkpoint_60.pt
--start-epoch 61
```

### Poor Accuracy
```bash
# Lower learning rate
--lr 0.8  # If loss diverging

# Check data augmentation
--cutmix-prob 0.0  # Disable if causing issues
```

## 📋 Training Checklist

- [ ] Launched p3.8xlarge spot instance
- [ ] Attached EBS with FFCV data
- [ ] Mounted at /data
- [ ] Verified 4 GPUs available
- [ ] Started tmux monitoring
- [ ] Training running with >95% GPU usage
- [ ] Checkpoints saving every 10 epochs
- [ ] Achieved target accuracy (~78%)
- [ ] Stopped and terminated instance
- [ ] Detached EBS volume

## 💾 Accessing Your Trained Model

To use your trained model later:

```bash
# Launch any instance (even CPU)
# Attach the EBS
# Mount at /data
# Your model is at:
/data/checkpoints/best_model.pt
/data/checkpoints/checkpoint_epoch_90.pt
```

## 🏁 Summary

**After 90 epochs (~1.5 hours):**
- ✅ Top-1 Accuracy: **77-78%**
- ✅ Top-5 Accuracy: **93-94%**
- ✅ Total Cost: **~$5.25** (spot pricing)
- ✅ Model saved to EBS for future use

**Total Project Cost (Phase 1 + 2): ~$5.87**

---

**Congratulations!** You've trained ResNet-50 to 78% accuracy for under $6! 🎉
