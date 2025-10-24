"""
ImageNet Dataset Management with CLI Interface

This module provides comprehensive ImageNet-1K dataset management including:
- Authentication and setup
- Dataset downloading and caching
- Download status monitoring
- Dataset verification and validation
- Support for smaller test datasets
- Command line interface for all operations

Usage:
    python imagenet.py download --help
    python imagenet.py status --help
    python imagenet.py verify --help
    python imagenet.py setup --help
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DownloadConfig
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import json
import time


class ImageNetConfig:
    """Configuration for ImageNet dataset operations"""

    # Cache directories
    DEFAULT_CACHE_DIR = './datasets/imagenet_cache'
    TEST_CACHE_DIR = './datasets/imagenet_test_cache'

    # Dataset sizes
    FULL_SIZE = {
        'train': 1281167,
        'validation': 50000
    }

    TEST_SIZE = {
        'train': 10000,  # 10k samples for quick testing
        'validation': 1000  # 1k samples for validation
    }

    # Download settings
    DOWNLOAD_CONFIG = DownloadConfig(
        num_proc=8,
        resume_download=True,
        max_retries=10,
    )


class ImageNetDataset(Dataset):
    """Wrapper for Hugging Face ImageNet-1K dataset"""

    def __init__(
        self,
        split: str = 'train',
        transform: Optional[object] = None,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        test_mode: bool = False,
        max_samples: Optional[int] = None
    ):
        self.split = split
        self.transform = transform
        self.test_mode = test_mode
        self.max_samples = max_samples

        # Set cache directory based on mode
        if cache_dir is None:
            cache_dir = ImageNetConfig.TEST_CACHE_DIR if test_mode else ImageNetConfig.DEFAULT_CACHE_DIR

        cache_dir = str(Path(cache_dir).absolute())

        print(f"\n{'='*80}")
        print(f"Loading ImageNet-1K ({split}) from Hugging Face...")
        print(f"Dataset: ILSVRC/imagenet-1k")
        print(f"Cache directory: {cache_dir}")
        print(f"Mode: {'Test' if test_mode else 'Full'}")
        print(f"Streaming mode: {streaming}")
        print(f"Max samples: {max_samples if max_samples else 'All'}")
        print(f"{'='*80}\n")

        try:
            self.dataset = load_dataset(
                "ILSVRC/imagenet-1k",
                split=split,
                cache_dir=cache_dir,
                streaming=streaming,
                download_config=ImageNetConfig.DOWNLOAD_CONFIG,
                num_proc=8 if not streaming else None
            )

            # If streaming, convert to list for random access
            if streaming:
                print("Converting streaming dataset to list for random access...")
                self.dataset = list(self.dataset)

            # Limit samples for test mode
            if test_mode and max_samples:
                print(f"Limiting to {max_samples} samples for test mode...")
                self.dataset = list(self.dataset)[:max_samples]

            print(f"✅ Successfully loaded {len(self.dataset)} images from {split} split")

        except Exception as e:
            print(f"❌ Error loading ImageNet-1K dataset: {e}")
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.dataset[idx]

        # Extract image and label
        image = sample['image']
        label = sample['label']

        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                image = Image.fromarray(np.array(image))

        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_imagenet_dataloaders(
    train_transform=None,
    val_transform=None,
    batch_size: int = 64,
    num_workers: int = 8,
    cache_dir: Optional[str] = None,
    test_mode: bool = False,
    max_samples: Optional[int] = None,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    # Create datasets
    train_dataset = ImageNetDataset(
        split='train',
        transform=train_transform,
        cache_dir=cache_dir,
        test_mode=test_mode,
        max_samples=max_samples
    )

    val_dataset = ImageNetDataset(
        split='validation',
        transform=val_transform,
        cache_dir=cache_dir,
        test_mode=test_mode,
        max_samples=max_samples
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None,
        drop_last=False
    )

    return train_loader, val_loader


def verify_imagenet_access() -> bool:
    """Verify that ImageNet-1K dataset is accessible"""
    try:
        print("Verifying ImageNet-1K dataset access...")
        test_dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split='train',
            streaming=True,
            download_config=ImageNetConfig.DOWNLOAD_CONFIG
        )
        # Try to get one sample
        sample = next(iter(test_dataset))
        print("✅ ImageNet-1K dataset is accessible")
        return True
    except Exception as e:
        print(f"❌ Cannot access ImageNet-1K dataset: {e}")
        return False


def check_huggingface_auth() -> bool:
    """Check if Hugging Face authentication is available"""
    # Check environment variables
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        print("✅ Hugging Face token found in environment variables")
        return True

    # Check if already logged in via CLI
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✅ Already logged in via huggingface-cli")
            return True
    except Exception:
        pass

    print("❌ No Hugging Face authentication found")
    return False


def authenticate_huggingface() -> bool:
    """Interactive authentication with Hugging Face"""
    print("\n" + "="*80)
    print("Hugging Face Authentication Setup")
    print("="*80)
    print("\nTo access the ImageNet-1K dataset, you need to:")
    print("1. Create a Hugging Face account: https://huggingface.co/join")
    print("2. Accept dataset terms: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    print("3. Create an access token: https://huggingface.co/settings/tokens")
    print("   (Make sure to create a 'Read' token)")
    print("\nChoose authentication method:")
    print("1. Login via CLI (recommended)")
    print("2. Enter token manually")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        print("\nRunning: huggingface-cli login")
        os.system("huggingface-cli login")
        return check_huggingface_auth()

    elif choice == "2":
        token = input("\nPaste your Hugging Face token: ").strip()
        if not token:
            print("❌ No token provided")
            return False

        try:
            from huggingface_hub import login
            login(token=token)
            print("✅ Successfully authenticated")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False

    else:
        print("Exiting...")
        return False


def setup_imagenet() -> bool:
    """Setup ImageNet dataset access"""
    print("="*80)
    print("ImageNet-1K Dataset Setup")
    print("="*80)

    # Check authentication
    if not check_huggingface_auth():
        print("\nAuthentication required...")
        if not authenticate_huggingface():
            print("\n❌ Setup failed. Please try again.")
            return False

    # Verify dataset access
    if not verify_imagenet_access():
        print("\n❌ Setup incomplete. Please resolve the issues above.")
        return False

    print("\n" + "="*80)
    print("✅ Setup Complete!")
    print("="*80)
    print("\nYou can now use the dataset with:")
    print("  python imagenet.py download --test  # Download test dataset")
    print("  python imagenet.py download        # Download full dataset")
    print("  python train.py --dataset test    # Train on test dataset")
    print("  python train.py --dataset full    # Train on full dataset")
    print("="*80 + "\n")

    return True


def check_download_status(cache_dir: str, test_mode: bool = False) -> Dict[str, Any]:
    """Check the status of dataset download"""
    print("="*80)
    print("ImageNet-1K Download Status")
    print("="*80)
    print(f"\nCache directory: {cache_dir}")
    print(f"Mode: {'Test' if test_mode else 'Full'}\n")

    cache_path = Path(cache_dir)

    if not cache_path.exists():
        return {
            'status': 'not_started',
            'cache_exists': False,
            'total_files': 0,
            'total_size': 0,
            'parquet_files': 0
        }

    # Calculate total size and file count
    total_size = 0
    file_count = 0
    for f in cache_path.rglob('*'):
        if f.is_file():
            total_size += f.stat().st_size
            file_count += 1

    # Look for dataset directory
    dataset_path = cache_path / "ILSVRC___imagenet-1k"
    if not dataset_path.exists():
        return {
            'status': 'not_started',
            'cache_exists': True,
            'total_files': file_count,
            'total_size': total_size,
            'parquet_files': 0
        }

    # Check for parquet files
    parquet_files = list(dataset_path.rglob("*.parquet"))
    expected_files = 50 if test_mode else 300

    if len(parquet_files) == 0:
        return {
            'status': 'no_files',
            'cache_exists': True,
            'total_files': file_count,
            'total_size': total_size,
            'parquet_files': 0
        }

    elif len(parquet_files) < expected_files:
        return {
            'status': 'in_progress',
            'cache_exists': True,
            'total_files': file_count,
            'total_size': total_size,
            'parquet_files': len(parquet_files),
            'expected_files': expected_files,
            'progress': len(parquet_files) / expected_files
        }

    else:
        return {
            'status': 'complete',
            'cache_exists': True,
            'total_files': file_count,
            'total_size': total_size,
            'parquet_files': len(parquet_files),
            'expected_files': expected_files
        }


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def verify_dataset(cache_dir: str, test_mode: bool = False) -> bool:
    """Verify dataset integrity"""
    print("="*80)
    print("ImageNet-1K Dataset Verification")
    print("="*80)

    status = check_download_status(cache_dir, test_mode)

    if status['status'] != 'complete':
        print(f"❌ Dataset download not complete. Status: {status['status']}")
        return False

    print("✅ Dataset download complete")
    print(f"   Files: {status['parquet_files']}")
    print(f"   Size: {format_size(status['total_size'])}")

    # Try to load and test data
    try:
        print("\n🧪 Testing data loading...")
        train_dataset = ImageNetDataset(
            split='train',
            cache_dir=cache_dir,
            test_mode=test_mode,
            max_samples=100  # Test with small sample
        )

        print(f"✅ Successfully loaded {len(train_dataset)} training samples")

        # Test a few samples
        for i in range(min(5, len(train_dataset))):
            try:
                image, label = train_dataset[i]
                print(f"   Sample {i}: shape={image.shape}, label={label}")
            except Exception as e:
                print(f"❌ Error loading sample {i}: {e}")
                return False

        print("\n✅ Dataset verification complete!")
        return True

    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
        return False


def download_dataset(test_mode: bool = False, max_samples: Optional[int] = None) -> bool:
    """Download ImageNet dataset"""
    cache_dir = ImageNetConfig.TEST_CACHE_DIR if test_mode else ImageNetConfig.DEFAULT_CACHE_DIR

    print("="*80)
    print("ImageNet-1K Dataset Download")
    print("="*80)
    print(f"Mode: {'Test' if test_mode else 'Full'}")
    print(f"Cache directory: {cache_dir}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print("="*80)

    # Check authentication first
    if not check_huggingface_auth():
        print("\n❌ Authentication required!")
        if not authenticate_huggingface():
            return False

    # Check if dataset already exists
    status = check_download_status(cache_dir, test_mode)
    if status['status'] == 'complete':
        print("\n✅ Dataset already downloaded and complete!")
        return True

    try:
        print("\n📥 Starting download...")
        if test_mode:
            print(f"Downloading test dataset ({max_samples or ImageNetConfig.TEST_SIZE['train']} train + {max_samples or ImageNetConfig.TEST_SIZE['validation']} validation samples)")
        else:
            print("Downloading full ImageNet-1K dataset (1.2M train + 50K validation images)")

        # Download train set
        print("\n📥 Downloading training set...")
        train_dataset = ImageNetDataset(
            split='train',
            cache_dir=cache_dir,
            test_mode=test_mode,
            max_samples=max_samples
        )

        # Download validation set
        print("\n📥 Downloading validation set...")
        val_dataset = ImageNetDataset(
            split='validation',
            cache_dir=cache_dir,
            test_mode=test_mode,
            max_samples=max_samples
        )

        print("\n✅ Download complete!")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Cache location: {cache_dir}")

        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="ImageNet-1K Dataset Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup ImageNet dataset access')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download ImageNet dataset')
    download_parser.add_argument('--test', action='store_true', help='Download test dataset (10k train + 1k validation)')
    download_parser.add_argument('--max-samples', type=int, help='Maximum number of samples to download')

    # Status command
    status_parser = subparsers.add_parser('status', help='Check download status')
    status_parser.add_argument('--test', action='store_true', help='Check test dataset status')

    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify dataset integrity')
    verify_parser.add_argument('--test', action='store_true', help='Verify test dataset')

    # Cache directory option for all commands
    for subparser in [download_parser, status_parser, verify_parser]:
        subparser.add_argument('--cache-dir', type=str, help='Custom cache directory')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute commands
    if args.command == 'setup':
        setup_imagenet()

    elif args.command == 'download':
        cache_dir = args.cache_dir or (ImageNetConfig.TEST_CACHE_DIR if args.test else ImageNetConfig.DEFAULT_CACHE_DIR)
        success = download_dataset(test_mode=args.test, max_samples=args.max_samples)
        sys.exit(0 if success else 1)

    elif args.command == 'status':
        cache_dir = args.cache_dir or (ImageNetConfig.TEST_CACHE_DIR if args.test else ImageNetConfig.DEFAULT_CACHE_DIR)
        status = check_download_status(cache_dir, test_mode=args.test)

        if status['status'] == 'complete':
            print("\n✅ Status: Download complete!")
        elif status['status'] == 'in_progress':
            print(f"\n🔄 Status: Download in progress ({status['progress']*100:.1f}% complete)")
        elif status['status'] == 'no_files':
            print("\n❌ Status: No files found")
        else:
            print("\n❌ Status: Download not started")

        print(f"   Files: {status['parquet_files']}")
        print(f"   Size: {format_size(status['total_size'])}")

    elif args.command == 'verify':
        cache_dir = args.cache_dir or (ImageNetConfig.TEST_CACHE_DIR if args.test else ImageNetConfig.DEFAULT_CACHE_DIR)
        success = verify_dataset(cache_dir, test_mode=args.test)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
