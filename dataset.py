#!/usr/bin/env python3
"""
Dataset management for ImageNet-1K training.
Handles dataset downloading, FFCV conversion, and data loading.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
import argparse

# PyTorch imports
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# HuggingFace datasets
from datasets import load_dataset, DownloadConfig
from PIL import Image

# Albumentations for augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# FFCV imports (optional but recommended)
try:
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage
    from ffcv.fields.decoders import (
        IntDecoder,
        RandomResizedCropRGBImageDecoder,
        CenterCropRGBImageDecoder,
    )
    from ffcv.transforms.common import Squeeze
    from ffcv.writer import DatasetWriter
    from ffcv.fields import IntField, RGBImageField

    FFCV_AVAILABLE = True
except ImportError:
    FFCV_AVAILABLE = False
    print("⚠️ FFCV not installed. Install with: pip install ffcv")

# ImageNet constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 1000


class ImageNetConfig:
    """Configuration for ImageNet dataset operations."""

    # Cache directories
    DEFAULT_CACHE_DIR = "./datasets/imagenet_cache"
    FFCV_DIR = "/datasets/ffcv"

    # Dataset sizes
    FULL_SIZE = {"train": 1281167, "validation": 50000}

    TEST_SIZE = {"train": 10000, "validation": 1000}

    # Download settings
    DOWNLOAD_CONFIG = DownloadConfig(
        num_proc=8,
        resume_download=True,
        max_retries=10,
    )


class HuggingFaceImageNet(Dataset):
    """Wrapper for HuggingFace ImageNet-1K dataset."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[object] = None,
        cache_dir: Optional[str] = None,
        test_mode: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.split = split
        self.transform = transform
        self.test_mode = test_mode
        self.max_samples = max_samples

        if cache_dir is None:
            cache_dir = ImageNetConfig.DEFAULT_CACHE_DIR

        print(f"Loading ImageNet-1K {split} split from HuggingFace...")

        try:
            self.dataset = load_dataset(
                "ILSVRC/imagenet-1k",
                split=split,
                cache_dir=cache_dir,
                streaming=False,
                download_config=ImageNetConfig.DOWNLOAD_CONFIG,
                num_proc=8,
            )

            if max_samples:
                self.dataset = self.dataset.select(
                    range(min(max_samples, len(self.dataset)))
                )

            print(f"✅ Loaded {len(self.dataset)} images")

        except Exception as e:
            print(f"❌ Error loading ImageNet: {e}")
            print("Make sure to authenticate with HuggingFace first:")
            print("  huggingface-cli login")
            raise

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["label"]

        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def convert_to_ffcv(
    output_dir: str = "/datasets/ffcv",
    max_resolution: int = 256,
    jpeg_quality: int = 90,
    max_samples_per_split: Optional[int] = None,
    max_classes: Optional[int] = None,
    num_workers: Optional[int] = None,
):
    """Convert HuggingFace ImageNet to FFCV format for fast loading.

    Args:
        output_dir: Directory to save FFCV files
        max_resolution: Maximum image resolution for FFCV
        jpeg_quality: JPEG compression quality
        max_samples_per_split: Maximum samples per split (for partial dataset)
        max_classes: Maximum number of classes to include (for partial dataset)
        num_workers: Number of workers for parallel processing (default: auto-detect)
    """

    if not FFCV_AVAILABLE:
        raise RuntimeError("FFCV not installed. Run: pip install ffcv")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log partial dataset settings
    if max_samples_per_split or max_classes:
        print(f"\n🔄 Creating PARTIAL dataset:")
        if max_samples_per_split:
            print(f"  Max samples per split: {max_samples_per_split}")
        if max_classes:
            print(f"  Max classes: {max_classes}")
        print()

    for split, output_name in [("train", "train.ffcv"), ("validation", "val.ffcv")]:
        output_path = output_dir / output_name

        if output_path.exists():
            print(f"✅ {output_path} already exists, skipping...")
            continue

        print(f"\nConverting {split} split to FFCV format...")
        print(f"Output: {output_path}")

        # Load dataset
        dataset = load_dataset("ILSVRC/imagenet-1k", split=split)
        print(f"Loaded {len(dataset)} samples from HuggingFace")

        # Apply filtering for partial dataset
        if max_samples_per_split or max_classes:
            # Filter by classes first if specified
            if max_classes and max_classes < NUM_CLASSES:
                print(f"Filtering to first {max_classes} classes...")
                # Filter to only include samples from first N classes
                filtered_indices = []
                for i, sample in enumerate(dataset):
                    if sample["label"] < max_classes:
                        filtered_indices.append(i)
                dataset = dataset.select(filtered_indices)
                print(
                    f"  Filtered to {len(dataset)} samples with {max_classes} classes"
                )

            # Then limit number of samples
            if max_samples_per_split and len(dataset) > max_samples_per_split:
                print(f"Limiting to {max_samples_per_split} samples...")
                dataset = dataset.select(range(max_samples_per_split))

        print(f"Final dataset size: {len(dataset)} samples")

        # Determine number of workers
        if num_workers is None:
            import multiprocessing

            actual_workers = min(multiprocessing.cpu_count(), 16)
        else:
            actual_workers = num_workers
        print(f"Using {actual_workers} workers for FFCV conversion")

        # Create FFCV writer
        writer = DatasetWriter(
            str(output_path),
            {
                "image": RGBImageField(
                    max_resolution=max_resolution,
                    jpeg_quality=jpeg_quality,
                    compress_probability=0.5,
                ),
                "label": IntField(),
            },
            num_workers=actual_workers,
        )

        # Convert dataset
        class HFWrapper:
            def __init__(self, hf_dataset):
                self.dataset = hf_dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                sample = self.dataset[idx]
                image = np.array(sample["image"])
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[2] == 4:
                    image = image[:, :, :3]
                return (image, sample["label"])

        wrapper = HFWrapper(dataset)
        writer.from_indexed_dataset(wrapper)

        file_size = output_path.stat().st_size / (1024**3)
        print(f"✅ Created {output_path} ({file_size:.2f} GB)")

    print("\n✅ FFCV conversion complete!")
    return output_dir


def get_ffcv_loaders(
    batch_size: int,
    num_workers: int = 8,
    image_size: int = 224,
    ffcv_dir: str = "/datasets/ffcv",
    distributed: bool = False,
    seed: Optional[int] = None,
    world_size: int = 1,
) -> Tuple[Loader, Loader]:
    """Create FFCV data loaders for fast training."""

    if not FFCV_AVAILABLE:
        raise RuntimeError("FFCV not installed. Run: pip install ffcv")

    train_path = Path(ffcv_dir) / "train.ffcv"
    val_path = Path(ffcv_dir) / "val.ffcv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"FFCV files not found in {ffcv_dir}. Run convert_to_ffcv() first."
        )

    # Training pipeline with progressive resizing support
    train_pipeline = [
        RandomResizedCropRGBImageDecoder((image_size, image_size)),
        ToTensor(),
        ToDevice(torch.device("cuda"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(
            mean=np.array(IMAGENET_MEAN) * 255,
            std=np.array(IMAGENET_STD) * 255,
            type=np.float16,
        ),
    ]

    # Validation pipeline (always 224x224)
    val_pipeline = [
        CenterCropRGBImageDecoder((224, 224), ratio=224 / 256),
        ToTensor(),
        ToDevice(torch.device("cuda"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(
            mean=np.array(IMAGENET_MEAN) * 255,
            std=np.array(IMAGENET_STD) * 255,
            type=np.float32,  # Use fp32 for validation to avoid NaN issues
        ),
    ]

    # Label pipeline
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda"), non_blocking=True),
    ]

    # Create loaders
    train_loader = Loader(
        str(train_path),
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.RANDOM,
        drop_last=True,
        pipelines={"image": train_pipeline, "label": label_pipeline},
        os_cache=True,
        distributed=distributed,
        seed=seed if seed is not None else 42,
    )

    # CRITICAL FIX: FFCV has a bug in distributed validation mode where ranks 1-7
    # hang during iteration. Workaround: Always use distributed=False for validation.
    # Only rank 0 will actually validate (handled in train.py), others will skip.
    val_loader = Loader(
        str(val_path),
        batch_size=batch_size * world_size if distributed else batch_size,
        num_workers=num_workers,
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
        pipelines={"image": val_pipeline, "label": label_pipeline},
        os_cache=True,
        distributed=False,  # CRITICAL: Always False - FFCV distributed validation is broken
    )

    return train_loader, val_loader


def get_pytorch_loaders(
    batch_size: int,
    num_workers: int = 8,
    cache_dir: Optional[str] = None,
    test_mode: bool = False,
    max_samples: Optional[int] = None,
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create standard PyTorch data loaders (fallback when FFCV not available)."""

    # Training transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Validation transforms
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Create datasets
    train_dataset = HuggingFaceImageNet(
        split="train",
        transform=train_transform,
        cache_dir=cache_dir,
        test_mode=test_mode,
        max_samples=max_samples,
    )

    val_dataset = HuggingFaceImageNet(
        split="validation",
        transform=val_transform,
        cache_dir=cache_dir,
        test_mode=test_mode,
        max_samples=max_samples if test_mode else None,
    )

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def test_ffcv_integration():
    """Test FFCV integration to ensure everything works."""

    if not FFCV_AVAILABLE:
        print("❌ FFCV not installed")
        return False

    try:
        print("Testing FFCV integration...")

        # Check for FFCV files
        ffcv_dir = Path(ImageNetConfig.FFCV_DIR)
        if not (ffcv_dir / "train.ffcv").exists():
            print("❌ FFCV files not found. Run convert_to_ffcv() first.")
            return False

        # Create test loader
        train_loader, val_loader = get_ffcv_loaders(
            batch_size=32, image_size=224, ffcv_dir=str(ffcv_dir), num_workers=4
        )

        # Test loading a batch
        batch = next(iter(train_loader))
        images = batch["image"]
        labels = batch["label"]

        print(f"✅ FFCV working! Batch shape: {images.shape}")
        return True

    except Exception as e:
        print(f"❌ FFCV test failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNet Dataset Management")
    parser.add_argument(
        "--convert-ffcv", action="store_true", help="Convert ImageNet to FFCV format"
    )
    parser.add_argument(
        "--test-ffcv", action="store_true", help="Test FFCV integration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/datasets/ffcv",
        help="Output directory for FFCV files",
    )

    args = parser.parse_args()

    if args.convert_ffcv:
        convert_to_ffcv(args.output_dir)
    elif args.test_ffcv:
        test_ffcv_integration()
    else:
        parser.print_help()
