# Documentation Index

## 📚 Main Documentation

### Getting Started
- **[README.md](README.md)** - Main project overview and quick start guide
  - Local development setup
  - Quick command reference
  - Feature overview

### AWS Training Guides (Recommended Path)
1. **[AWS_PHASE1_CPU_SETUP.md](AWS_PHASE1_CPU_SETUP.md)** - Data preparation on CPU instance
   - Setup c5a.4xlarge instance
   - Download and convert ImageNet to FFCV format
   - Time: ~1 hour, Cost: $0.62

2. **[AWS_PHASE2_GPU_TRAINING.md](AWS_PHASE2_GPU_TRAINING.md)** - Training on GPU instance
   - Setup p4d.24xlarge with 8× A100 GPUs
   - Train ResNet-50 to 78% accuracy
   - Time: ~45 minutes, Cost: $8.25

## 🔧 Troubleshooting & Technical Docs

### Issue Resolution
- **[TRAINING_ISSUES_RESOLVED.md](TRAINING_ISSUES_RESOLVED.md)** - ✅ **START HERE if you have training issues**
  - Fixed: Validation stuck at 0.1%
  - Fixed: NaN losses
  - Fixed: Incorrect learning rate
  - Working configuration and expected results

- **[DISTRIBUTED_FREEZE_COMPLETE_GUIDE.md](DISTRIBUTED_FREEZE_COMPLETE_GUIDE.md)** - Historical: Distributed training freeze issue
  - Problem: Training froze after epoch 1
  - Status: Resolved via FFCV distributed validation fix
  - Keep for reference only

## 🎯 Quick Navigation

**I want to...**

### Train on AWS
→ Follow [AWS_PHASE1_CPU_SETUP.md](AWS_PHASE1_CPU_SETUP.md) then [AWS_PHASE2_GPU_TRAINING.md](AWS_PHASE2_GPU_TRAINING.md)

### Fix Training Issues
→ See [TRAINING_ISSUES_RESOLVED.md](TRAINING_ISSUES_RESOLVED.md)

### Train Locally
→ See [README.md](README.md) installation and training sections

### Understand the Code
→ See [README.md](README.md) architecture section

---

**Last Updated**: 2025-10-25

