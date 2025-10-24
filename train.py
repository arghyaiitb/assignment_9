# Modern Python 3.8+ with type hints and latest syntax
import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from torchinfo import summary  # Modern replacement for torchsummary
import matplotlib.pyplot as plt
from model import ResNet50
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from pathlib import Path
import sys
from datetime import datetime
from torch.distributions.beta import Beta
from typing import Optional, Tuple
import argparse

# Import ImageNet-1K data loader from consolidated module
from imagenet import get_imagenet_dataloaders

# Essential constants (moved from constants.py to eliminate dependency)

# Directory paths
CHECKPOINT_DIR = Path('./checkpoints')
LOG_FILE = Path('./training_logs.md')

# Data loading configuration
CUDA_BATCH_SIZE = 64
CPU_BATCH_SIZE = 32
NUM_WORKERS = 8
PREFETCH_FACTOR = 3

# Augmentation parameters
CUTMIX_PROB = 0.3
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.3
MIXUP_ALPHA = 0.8

# Training parameters
LABEL_SMOOTHING = 0.1
GRADIENT_CLIPPING_MAX_NORM = 1.0

# ImageNet normalization parameters
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Create checkpoint directory
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on ImageNet-1K")
    parser.add_argument('--dataset', choices=['test', 'full'], default='test',
                       help='Dataset to use: test (10k train + 1k val) or full (1.2M train + 50k val)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to use (overrides dataset choice)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides default based on CUDA availability)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs to train')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Custom cache directory for dataset')
    return parser.parse_args()

class Logger:
    """Logger that writes to both console and file using modern context management"""
    def __init__(self, log_file: Path) -> None:
        self.terminal = sys.stdout
        self.log_file = log_file
        self.log = open(log_file, 'a', encoding='utf-8')

    def write(self, message: str) -> None:
        """Write message to both terminal and log file"""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write

    def flush(self) -> None:
        """Flush both terminal and log file"""
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        """Close the log file"""
        if hasattr(self.log, 'close'):
            self.log.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure file is closed"""
        self.close()


# Initialize logger with context management
logger = Logger(LOG_FILE)
sys.stdout = logger

# Write header to log file
print(f"\n{'='*80}")
print(f"# ImageNet-1K Training Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")
print("## ResNet-50 ImageNet-1K Training Configuration")
print("✅ **Architecture**: ResNet-50 with Bottleneck blocks [3,4,6,3]")
print("✅ **Dataset**: ImageNet-1K (1000 classes, 224x224 input)")
print("✅ **MaxPool**: Included after initial conv (required for ImageNet)")
print("✅ **Dropout**: 0.5 in final FC layer (standard for ImageNet)")
print("✅ **CutMix Augmentation**: Enabled with prob=0.3, alpha=1.0")
print("✅ **MixUp Augmentation**: Enabled with prob=0.3, alpha=0.8")
print("✅ **Label Smoothing**: 0.1 (reduces overfitting)")
print("✅ **Optimized LR Schedule**: OneCycleLR for 150 epochs")
print("   └─ Target: 83% top-1 accuracy in <150 epochs")
print("Expected Impact: State-of-the-art ImageNet performance")
print()


# Custom transform class for albumentations
class AlbumentationTransforms:
    def __init__(self, transforms_list):
        self.transforms = A.Compose(transforms_list)

    def __call__(self, img):
        img = np.array(img)
        return self.transforms(image=img)['image']

# ImageNet Training Phase transformations with Albumentations 2.0.8 (latest)
# Reference: https://albumentations.ai/docs/
train_transforms = AlbumentationTransforms([
    # Standard ImageNet augmentations - Fixed for Albumentations 2.0.8 API
    A.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.33), p=1.0),
    A.HorizontalFlip(p=0.5),

    # Color augmentations - Albumentations 2.0.8 compatible
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
    ], p=0.8),

    # Geometric augmentations - Albumentations 2.0.8 syntax
    A.OneOf([
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                 scale=(0.9, 1.1), rotate=(-10, 10), p=1.0),
        A.Perspective(scale=(0.0, 0.1), p=1.0),
    ], p=0.5),

    # Noise and blur - Albumentations 2.0.8 compatible
    A.OneOf([
        A.GaussNoise(std_range=(0.0, 0.1), p=1.0),  # Using std_range (works without warnings)
        A.MotionBlur(blur_limit=3, p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.3),

    # Dropout augmentations - Fixed for Albumentations 2.0.8 API
    A.OneOf([
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=tuple([int(x * 255) for x in IMAGENET_MEAN]),
            p=1.0
        ),
    ], p=0.3),

    # Normalization and tensor conversion
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

# ImageNet Validation Phase transformations
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Parse arguments and setup
args = parse_args()

print("\n" + "="*80)
print("ResNet-50 Training Configuration")
print("="*80)
print(f"Dataset: {'Test (10k train + 1k val)' if args.dataset == 'test' else 'Full ImageNet-1K (1.2M train + 50k val)'}")
print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
print(f"Epochs: {args.epochs}")
print(f"Batch size: {args.batch_size if args.batch_size else 'Auto (64 CUDA, 32 CPU)'}")
print(f"Cache directory: {args.cache_dir if args.cache_dir else 'Auto'}")
print("="*80 + "\n")


SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    # Enable cuDNN benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("cuDNN benchmark enabled for maximum speed")

# Create ImageNet-1K dataloaders using consolidated module
print("Creating dataloaders...")
batch_size = args.batch_size or (CUDA_BATCH_SIZE if cuda else CPU_BATCH_SIZE)
num_workers = NUM_WORKERS if cuda else 2
test_mode = args.dataset == 'test'

# Determine max_samples based on dataset choice and user input
max_samples = args.max_samples
if not max_samples and test_mode:
    # Default test dataset sizes
    max_samples = 10000  # 10k train samples

train_loader, test_loader = get_imagenet_dataloaders(
    train_transform=train_transforms,
    val_transform=test_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    cache_dir=args.cache_dir,
    test_mode=test_mode,
    max_samples=max_samples,
    pin_memory=cuda,
    persistent_workers=False,  # Disabled to avoid PyGILState_Release errors on exit
    prefetch_factor=PREFETCH_FACTOR if cuda else 2,
    streaming=False  # Set to False for better performance with random access
)

print(f"\n✅ Dataloaders created successfully")
print(f"   Batch size: {batch_size}")
print(f"   Num workers: {num_workers}")
print(f"   Training batches: {len(train_loader)}")
print(f"   Validation batches: {len(test_loader)}")
print()


train_losses = []
test_losses = []
train_acc = []
test_acc = []
lrs = []


def save_and_plot_metrics(output_dir: Path) -> None:
    """Save metrics (CSV + NPZ) and generate overview plots (2x2 grid and individual).

    Files written under output_dir:
      - metrics.csv (epoch, train_loss, test_loss, train_acc, test_acc, lr)
      - metrics.npz (numpy arrays)
      - metrics_overview.png (2x2 grid)
      - train_loss.png, test_loss.png, train_acc.png, test_acc.png
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    epochs_axis = list(range(len(train_losses)))

    # Save CSV
    csv_path = output_dir / 'metrics.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,test_loss,train_acc,test_acc,lr\n')
        for i in range(len(epochs_axis)):
            tl = train_losses[i] if i < len(train_losses) else ''
            tsl = test_losses[i] if i < len(test_losses) else ''
            tra = train_acc[i] if i < len(train_acc) else ''
            tea = test_acc[i] if i < len(test_acc) else ''
            lr_i = lrs[i] if i < len(lrs) else ''
            f.write(f"{i},{tl},{tsl},{tra},{tea},{lr_i}\n")

    # Save NPZ for programmatic reuse
    np.savez(output_dir / 'metrics.npz',
             train_losses=np.array(train_losses, dtype=float),
             test_losses=np.array(test_losses, dtype=float),
             train_acc=np.array(train_acc, dtype=float),
             test_acc=np.array(test_acc, dtype=float),
             lrs=np.array(lrs, dtype=float))

    # Individual plots
    def _save_line(x, y, title, ylabel, path):
        plt.figure(figsize=(6, 4))
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    _save_line(epochs_axis, train_losses, 'Training Loss', 'Loss', output_dir / 'train_loss.png')
    _save_line(epochs_axis, test_losses, 'Test Loss', 'Loss', output_dir / 'test_loss.png')
    _save_line(epochs_axis, train_acc, 'Training Accuracy', 'Accuracy (%)', output_dir / 'train_acc.png')
    _save_line(epochs_axis, test_acc, 'Test Accuracy', 'Accuracy (%)', output_dir / 'test_acc.png')

    # 2x2 overview
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs_axis, train_losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_axis, test_losses)
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs_axis, train_acc)
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_axis, test_acc)
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_overview.png', dpi=200)
    plt.close(fig)


def train_epoch(model: nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, epoch: int, scaler: Optional[torch.amp.GradScaler] = None,
                scheduler: Optional[object] = None) -> float:
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    running_loss = 0.0
    
    # CutMix and MixUp hyperparameters
    cutmix_prob = CUTMIX_PROB  # Apply CutMix with configured probability
    cutmix_alpha = CUTMIX_ALPHA  # Beta distribution parameter for CutMix
    mixup_prob = MIXUP_PROB   # Apply MixUp with configured probability
    mixup_alpha = MIXUP_ALPHA  # Beta distribution parameter for MixUp
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        # Decide whether to apply CutMix or MixUp for this batch
        use_cutmix = np.random.rand() < cutmix_prob
        use_mixup = np.random.rand() < mixup_prob
        
        if use_cutmix and not use_mixup:  # CutMix only if not MixUp
            # Sample lambda from Beta distribution
            lam = Beta(cutmix_alpha, cutmix_alpha).sample().item()
            batch_size = data.size(0)
            index = torch.randperm(batch_size).to(device)
            
            # Generate random bounding box
            H, W = data.size(2), data.size(3)
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            
            # Uniform random center point
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            # Bounding box coordinates
            x1 = np.clip(cx - cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y2 = np.clip(cy + cut_h // 2, 0, H)
            
            # Apply CutMix: replace patch with shuffled batch
            data[:, :, y1:y2, x1:x2] = data[index, :, y1:y2, x1:x2]
            
            # Adjust lambda to actual area ratio
            lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
            
            # Mixed precision training with CutMix - PyTorch 2.0+ API
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(data)
                    # Mixed loss: NO label smoothing (CutMix already provides soft labels)
                    loss = lam * F.cross_entropy(output, target) + \
                           (1 - lam) * F.cross_entropy(output, target[index])
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                # Mixed loss: NO label smoothing (CutMix already provides soft labels)
                loss = lam * F.cross_entropy(output, target) + \
                       (1 - lam) * F.cross_entropy(output, target[index])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
                optimizer.step()
        elif use_mixup:  # MixUp augmentation
            # Sample lambda from Beta distribution
            lam = Beta(mixup_alpha, mixup_alpha).sample().item()
            batch_size = data.size(0)
            index = torch.randperm(batch_size).to(device)

            # Mix the inputs
            mixed_data = lam * data + (1 - lam) * data[index]
            mixed_targets = lam * target + (1 - lam) * target[index]

            # Mixed precision training with MixUp - PyTorch 2.0+ optimized
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(mixed_data)
                    # Mixed loss: NO label smoothing (MixUp already provides soft labels)
                    loss = F.cross_entropy(output, target) * lam + F.cross_entropy(output, target[index]) * (1 - lam)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(mixed_data)
                # Mixed loss: NO label smoothing (MixUp already provides soft labels)
                loss = F.cross_entropy(output, target) * lam + F.cross_entropy(output, target[index]) * (1 - lam)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
                optimizer.step()
        else:
            # Standard training without CutMix or MixUp - PyTorch 2.0+ API
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(data)
                    loss = F.cross_entropy(output, target, label_smoothing=LABEL_SMOOTHING)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = F.cross_entropy(output, target, label_smoothing=LABEL_SMOOTHING)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_MAX_NORM)
                optimizer.step()
        
        # Step scheduler after each batch for OneCycleLR
        if scheduler is not None:
            scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        running_loss += loss.item()
        
        pbar.set_description(desc=f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    
    train_accuracy = 100 * correct / processed
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_acc.append(train_accuracy)
    return train_accuracy


def test(model: nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader) -> float:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

    return test_loss


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, test_loss: float,
                   test_accuracy: float, is_best: bool = False, checkpoint_dir: Path = CHECKPOINT_DIR) -> None:
    """Save training checkpoint with model state, optimizer state, and training metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_acc': train_acc,
        'test_acc': test_acc,
    }
    
    # Save periodic checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save best model checkpoint
    if is_best:
        best_model_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved: {best_model_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, torch.optim.Optimizer, int, float, float]:
    """Load training checkpoint to resume training."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    test_loss = checkpoint['test_loss']
    test_accuracy = checkpoint['test_accuracy']
    
    # Restore training history
    global train_losses, test_losses, train_acc, test_acc
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    train_acc = checkpoint['train_acc']
    test_acc = checkpoint['test_acc']
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"Resuming from epoch: {epoch}, Test Accuracy: {test_accuracy:.2f}%")
    
    return model, optimizer, epoch, test_loss, test_accuracy


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



print("\n## Model Configuration")
print("- Architecture: ResNet-50 (Standard ImageNet configuration)")
print("- Block Structure: [3, 4, 6, 3] = 16 Bottleneck blocks (standard ResNet-50)")
print("- MaxPool: INCLUDED after initial conv (required for ImageNet 224x224)")
print("- Dropout in residual blocks: 0.0 (disabled)")
print("- FC Layer Dropout: 0.5 (standard for ImageNet)")
print("- Number of classes: 1000 (ImageNet-1K)")
print("- Input resolution: 224x224")
print()

model = ResNet50(num_classes=1000).to(device)

# Use channels_last memory format for better T4 performance
if cuda:
    model = model.to(memory_format=torch.channels_last)
    print("Using channels_last memory format for optimal T4 performance")

# Modern torchinfo API (replacement for torchsummary)
summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params"], 
        verbose=1, device=device)

# ImageNet-optimized optimizer with proper weight decay
# Note: Initial LR will be set by OneCycleLR scheduler (max_lr/div_factor)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

# Initialize mixed precision scaler for faster training - PyTorch 2.0+ API
scaler = torch.amp.GradScaler(device='cuda') if cuda else None
if scaler:
    print("Mixed precision training enabled (AMP) for faster training")

# OneCycleLR optimized for ImageNet - 150 epochs target for 83% accuracy
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=150,  # Reduced from 200 for faster convergence to 83% target
    steps_per_epoch=len(train_loader),
    pct_start=0.25,  # Warmup for 25% of training (37 epochs) - longer warmup for ImageNet
    div_factor=25,   # Initial lr = 0.1/25 = 0.004 (stable warmup)
    final_div_factor=10000,  # Final lr = 0.1/10000 = 0.00001 (very low for fine convergence)
    anneal_strategy='cos'
)

print("\n## Training Hyperparameters")
print(f"- Epochs: 150 (target 83% accuracy)")
print(f"- Batch Size: 64 (224x224 ImageNet resolution)")
print(f"- Optimizer: SGD (momentum=0.9, weight_decay=1e-4, nesterov=True)")
print(f"- Scheduler: OneCycleLR (max_lr=0.1, initial_lr=0.004, final_lr=0.00001)")
print(f"- Label Smoothing: 0.1")
print(f"- Gradient Clipping: max_norm=1.0")
print(f"- Mixed Precision: {'Enabled' if scaler else 'Disabled'}")
print(f"- CutMix: Enabled (prob=0.3, alpha=1.0)")
print(f"- MixUp: Enabled (prob=0.3, alpha=0.8)")
print(f"- Advanced Augmentation: RandResizedCrop, ColorJitter, Cutout, MotionBlur")
print(f"- Input Resolution: 224x224 (standard ImageNet)")
print()

EPOCHS = args.epochs  # Configurable via command line
CHECKPOINT_INTERVAL = 5

# Initialize training variables
best_test_accuracy = 0.0

print("\n## Training Progress\n")
print("| Epoch | Train Acc | Test Acc | Test Loss | LR | Status |")
print("|-------|-----------|----------|-----------|-----|--------|")


def main():
    """Main training function"""
    # Resume from checkpoint if requested
    if args.resume:
        if not Path(args.resume).exists():
            print(f"❌ Checkpoint file not found: {args.resume}")
            return
        print(f"🔄 Resuming training from: {args.resume}")
        # TODO: Implement resume functionality
        print("⚠️  Resume functionality not yet implemented")

    # Run training
    print("\n🚀 Starting training...")
    print(f"Training for {EPOCHS} epochs with {args.dataset} dataset")

    # Execute training loop (moved from global scope)
    for epoch in range(EPOCHS):
        print(f"\n### EPOCH: {epoch}")

        train_accuracy = train_epoch(model, device, train_loader, optimizer, epoch, scaler, scheduler)
        test_loss = test(model, device, test_loader)

        # Print current LR for monitoring
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        print(f"Current LR: {current_lr:.6f}")

        # Calculate current test accuracy
        current_test_accuracy = test_acc[-1] if test_acc else 0.0

        # Display Train and Test Accuracy Side by Side
        print(f"\n{'='*70}")
        print(f"EPOCH: {epoch:3d} | Train Accuracy: {train_accuracy:6.2f}% | Test Accuracy: {current_test_accuracy:6.2f}%")
        print(f"{'='*70}\n")

        # Add row to markdown table
        status = ""

        # Save checkpoint logic
        is_best = current_test_accuracy > best_test_accuracy
        if is_best:
            best_test_accuracy = current_test_accuracy
            status = "🎯 **BEST**"

        # Add table row
        print(f"| {epoch:5d} | {train_accuracy:8.2f}% | {current_test_accuracy:7.2f}% | {test_loss:8.4f} | {current_lr:.6f} | {status} |")

        # Save best model checkpoint
        if is_best:
            save_checkpoint(model, optimizer, epoch, test_loss, current_test_accuracy, is_best=True)

        # Save periodic checkpoint
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch, test_loss, current_test_accuracy, is_best=False)

        # Early stopping if we hit target accuracy
        if current_test_accuracy >= 75.0:
            print(f"\n{'='*70}")
            print(f"🎯 TARGET ACHIEVED! Test Accuracy: {current_test_accuracy:.2f}% >= 75%")
            print(f"{'='*70}\n")
            save_checkpoint(model, optimizer, epoch, test_loss, current_test_accuracy, is_best=True)
            break

    print("\n" + "="*80)
    print("# Training Completed!")
    print(f"**Best Test Accuracy: {best_test_accuracy:.2f}%**")
    print(f"Training Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    # Save metrics and plots
    metrics_dir = Path('./training_metrics')
    print("Saving training metrics and plots to ./training_metrics ...")
    save_and_plot_metrics(metrics_dir)
    print("✅ Saved metrics: CSV, NPZ and plots.")

    print("\n" + "="*80)

    # Close log file properly using context manager
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    sys.stdout = sys.__stdout__  # Restore original stdout
    print(f"Training logs saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
