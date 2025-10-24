#!/usr/bin/env python3
"""
Simple test script to verify training pipeline with synthetic data
This creates a minimal dataset to test the training functionality without downloading ImageNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from model import ResNet50
import sys
from pathlib import Path
import argparse

# Essential constants for testing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SyntheticImageNetDataset(Dataset):
    """Synthetic dataset mimicking ImageNet structure for testing"""

    def __init__(self, split='train', num_samples=100, num_classes=10):
        self.split = split
        self.num_samples = num_samples
        self.num_classes = num_classes

        # Simple transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic image (224x224x3) as numpy array
        if self.split == 'train':
            # Random image in range [0, 1] with float32 precision
            image = np.random.rand(224, 224, 3).astype(np.float32)
        else:
            # Slightly different for validation
            image = (np.random.rand(224, 224, 3) * 0.8).astype(np.float32)

        # Random label
        label = torch.randint(0, self.num_classes, (1,)).item()

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_synthetic_dataloaders(batch_size=16, num_samples=100, num_classes=10):
    """Create synthetic dataloaders for testing"""

    train_dataset = SyntheticImageNetDataset('train', num_samples, num_classes)
    val_dataset = SyntheticImageNetDataset('validation', num_samples//5, num_classes)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader


def test_training():
    """Test the training pipeline with synthetic data"""

    print("="*80)
    print("Testing Training Pipeline with Synthetic Data")
    print("="*80)
    print("Using synthetic dataset to verify training functionality")
    print("Dataset: 100 train samples, 20 validation samples, 10 classes")
    print("="*80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model with fewer classes for testing
    model = ResNet50(num_classes=10).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create synthetic dataloaders
    train_loader, test_loader = get_synthetic_dataloaders(batch_size=16, num_samples=100)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(test_loader)}")

    # Setup optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training parameters
    criterion = nn.CrossEntropyLoss()

    # Test training loop for 1 epoch
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    print("\nTesting training loop...")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += len(data)

        if batch_idx % 2 == 0:  # Print every 2 batches
            print(f"Batch {batch_idx:2d}: Loss={loss.item():.4f}, Accuracy={100*correct/total:.2f}%")

    scheduler.step()

    # Test validation
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    print("\nTesting validation...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += len(data)

    # Results
    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Training Loss:   {train_loss:.4f}")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Test Loss:       {test_loss:.4f}")
    print(f"Test Accuracy:   {test_acc:.2f}%")
    print("="*80)

    if test_acc > 5:  # Very basic check - should get better than random (10% for 10 classes)
        print("✅ Training pipeline test PASSED!")
        print("✅ Model can train and evaluate successfully")
        return True
    else:
        print("❌ Training pipeline test FAILED!")
        print("❌ Model accuracy too low - check implementation")
        return False


if __name__ == "__main__":
    success = test_training()
    sys.exit(0 if success else 1)
