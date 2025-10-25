#!/usr/bin/env python3
"""
Main entry point for ResNet-50 ImageNet training.
Connects all components and provides a unified interface for training.
"""

import os
import sys
import argparse
import logging
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import Dict, Any

# Import our modules
from train import train_single_gpu, train_distributed, Trainer
from dataset import convert_to_ffcv, test_ffcv_integration, FFCV_AVAILABLE
from models import ResNet50


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ResNet-50 ImageNet Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "mode",
        choices=["train", "distributed", "convert-ffcv", "test", "validate"],
        help="Operation mode",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        choices=["full", "test"],
        default="full",
        help="Dataset variant (full ImageNet or test subset)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="/datasets", help="Base directory for datasets"
    )
    parser.add_argument(
        "--ffcv-dir",
        type=str,
        default="/datasets/ffcv",
        help="Directory for FFCV files",
    )
    parser.add_argument(
        "--use-ffcv",
        action="store_true",
        help="Use FFCV data loaders for faster training",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to use (for testing)"
    )

    # Quick testing with partial dataset
    parser.add_argument(
        "--partial-dataset",
        action="store_true",
        help="Use partial dataset for quick testing (1%% of full dataset)",
    )
    parser.add_argument(
        "--partial-size",
        type=int,
        default=None,
        help="Number of samples for partial dataset (default: 1%% of full)",
    )
    parser.add_argument(
        "--partial-classes",
        type=int,
        default=None,
        help="Number of classes to use (default: all 1000)",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Total batch size (will be divided across GPUs)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.8,
        help="Initial learning rate (will be scaled for distributed)",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--label-smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    parser.add_argument(
        "--scheduler",
        choices=["onecycle", "cosine", "none"],
        default="onecycle",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for learning rate scheduler",
    )

    # Augmentation arguments
    parser.add_argument(
        "--cutmix-prob", type=float, default=0.5, help="CutMix probability"
    )
    parser.add_argument(
        "--mixup-alpha", type=float, default=0.4, help="MixUp alpha parameter"
    )
    parser.add_argument(
        "--progressive-resize",
        action="store_true",
        help="Use progressive resizing (160→192→224)",
    )

    # Optimization arguments
    parser.add_argument(
        "--amp", action="store_true", default=True, help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        dest="amp",
        help="Alias for --amp (automatic mixed precision)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for optimization (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use exponential moving average of weights",
    )
    parser.add_argument(
        "--gradient-clip", type=float, default=1.0, help="Gradient clipping value"
    )

    # Distributed training arguments
    parser.add_argument(
        "--world-size",
        type=int,
        default=-1,
        help="Number of GPUs to use (-1 for all available)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers per GPU",
    )
    parser.add_argument(
        "--dist-backend", type=str, default="nccl", help="Distributed backend"
    )

    # Checkpoint arguments
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=True,
        help="Automatically resume from latest checkpoint if available (default: True)",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_false",
        dest="auto_resume",
        help="Disable automatic resume from latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/data/checkpoints",
        help="Directory to save checkpoints (default: /data/checkpoints)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/data/logs",
        help="Directory to save training logs (default: /data/logs)",
    )
    parser.add_argument(
        "--validate-only", type=str, help="Path to checkpoint for validation only"
    )

    # Other arguments
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=78.0,
        help="Target validation accuracy for early stopping",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        help="Maximum training time in hours (for budget control)",
    )

    return parser.parse_args()


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def prepare_config(args) -> Dict[str, Any]:
    """Prepare configuration dictionary from arguments."""

    # Handle partial dataset settings
    max_samples = args.max_samples
    partial_classes = args.partial_classes

    if args.partial_dataset:
        # Set default partial sizes if not specified
        if args.partial_size:
            max_samples = args.partial_size
        else:
            # Use 1% of full dataset by default
            max_samples = (
                12812 if args.dataset == "full" else 100
            )  # 1% of 1.28M for train

        if not args.partial_classes:
            partial_classes = 1000  # Use all classes by default

    config = {
        # Dataset
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "ffcv_dir": args.ffcv_dir,
        "use_ffcv": args.use_ffcv,
        "max_samples": max_samples,
        "test_mode": args.dataset == "test" or args.partial_dataset,
        "partial_dataset": args.partial_dataset,
        "partial_classes": partial_classes,
        # Training
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        # Augmentation
        "cutmix_prob": args.cutmix_prob,
        "mixup_alpha": args.mixup_alpha,
        "progressive_resize": args.progressive_resize,
        # Optimization
        "amp": args.amp,
        "compile": args.compile,
        "use_ema": args.use_ema,
        "gradient_clip": args.gradient_clip,
        # Distributed
        "num_workers": args.num_workers,
        "dist_backend": args.dist_backend,
        # Checkpointing
        "resume": args.resume,
        "auto_resume": args.auto_resume,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir": args.log_dir,
        # Other
        "seed": args.seed,
        "target_accuracy": args.target_accuracy,
        "budget_hours": args.budget_hours,
    }

    return config


def mode_train(args, logger):
    """Single GPU training mode."""
    logger.info("Starting single GPU training...")

    # Prepare configuration
    config = prepare_config(args)

    # Log if using partial dataset
    if args.partial_dataset:
        logger.info("🔄 Using PARTIAL dataset for quick testing")
        logger.info(f"  Max samples: {config['max_samples']}")
        logger.info(f"  Classes: {config['partial_classes']}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! Training requires GPU.")
        return

    # Check FFCV if requested
    if config["use_ffcv"]:
        if not FFCV_AVAILABLE:
            logger.error("FFCV is not installed! Install with: pip install ffcv")
            return

        # Adjust FFCV directory for partial dataset
        if args.partial_dataset:
            config["ffcv_dir"] = config["ffcv_dir"].rstrip("/") + "_partial"
            logger.info(f"Using partial FFCV directory: {config['ffcv_dir']}")

        # Check if FFCV files exist
        ffcv_dir = Path(config["ffcv_dir"])
        if not (ffcv_dir / "train.ffcv").exists():
            logger.warning(f"FFCV files not found in {ffcv_dir}")
            logger.warning("Please run with mode='convert-ffcv' first:")
            if args.partial_dataset:
                logger.warning("  python main.py convert-ffcv --partial-dataset")
            else:
                logger.warning("  python main.py convert-ffcv")
            return

    # Start training
    train_single_gpu(config)


def mode_distributed(args, logger):
    """Multi-GPU distributed training mode."""
    logger.info("Starting distributed training...")

    # Determine world size
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()

    if args.world_size < 2:
        logger.warning(
            "Less than 2 GPUs available. Falling back to single GPU training."
        )
        mode_train(args, logger)
        return

    logger.info(f"Using {args.world_size} GPUs for distributed training")

    # Prepare configuration
    config = prepare_config(args)

    # Log if using partial dataset
    if args.partial_dataset:
        logger.info("🔄 Using PARTIAL dataset for quick testing")
        logger.info(f"  Max samples: {config['max_samples']}")
        logger.info(f"  Classes: {config['partial_classes']}")

        # Adjust FFCV directory for partial dataset
        if config["use_ffcv"]:
            config["ffcv_dir"] = config["ffcv_dir"].rstrip("/") + "_partial"
            logger.info(f"Using partial FFCV directory: {config['ffcv_dir']}")

    # Check FFCV if requested
    if config["use_ffcv"] and not FFCV_AVAILABLE:
        logger.error("FFCV is not installed! Install with: pip install ffcv")
        return

    # Spawn processes for distributed training
    mp.spawn(
        train_distributed,
        args=(args.world_size, config),
        nprocs=args.world_size,
        join=True,
    )


def mode_convert_ffcv(args, logger):
    """Convert ImageNet to FFCV format."""

    # Prepare configuration to handle partial dataset
    config = prepare_config(args)

    if args.partial_dataset:
        logger.info(f"Converting PARTIAL ImageNet to FFCV format...")
        logger.info(f"  Samples: {config['max_samples']} per split")
        logger.info(f"  Classes: {config['partial_classes']}")
    else:
        logger.info("Converting FULL ImageNet to FFCV format...")

    if not FFCV_AVAILABLE:
        logger.error("FFCV is not installed! Install with: pip install ffcv")
        return

    try:
        # Check HuggingFace authentication
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if not token:
            logger.warning("HuggingFace token not found. You may need to login:")
            logger.warning("  huggingface-cli login")
    except ImportError:
        pass

    # Determine output directory for partial dataset
    output_dir = args.ffcv_dir
    if args.partial_dataset:
        # Use a different directory for partial datasets to avoid confusion
        output_dir = output_dir.rstrip("/") + "_partial"
        logger.info(f"Partial dataset will be saved to: {output_dir}")

    # Convert to FFCV
    convert_to_ffcv(
        output_dir=output_dir,
        max_samples_per_split=config.get("max_samples"),
        max_classes=config.get("partial_classes"),
    )


def mode_test(args, logger):
    """Test mode for quick validation of setup."""
    logger.info("Running tests...")

    # Test 1: Check CUDA
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.get_device_name(0)}")

    # Test 2: Check FFCV
    if FFCV_AVAILABLE:
        logger.info("FFCV is installed")
        if test_ffcv_integration():
            logger.info("FFCV integration test passed!")
    else:
        logger.warning("FFCV not installed")

    # Test 3: Quick training test
    logger.info("Running quick training test...")
    config = prepare_config(args)
    config["epochs"] = 1
    config["max_samples"] = 100
    config["test_mode"] = True

    try:
        trainer = Trainer(config)
        train_acc, train_loss = trainer.train_epoch()
        val_acc, val_loss = trainer.validate()
        logger.info(
            f"Quick test completed! Train: {train_acc:.2f}%, Val: {val_acc:.2f}%"
        )
    except Exception as e:
        logger.error(f"Training test failed: {e}")


def mode_validate(args, logger):
    """Validation mode for evaluating a checkpoint."""
    if not args.validate_only:
        logger.error("Please provide checkpoint path with --validate-only")
        return

    checkpoint_path = Path(args.validate_only)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Validating checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", prepare_config(args))

    # Create trainer and load model
    trainer = Trainer(config)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])

    if "ema_state_dict" in checkpoint and config.get("use_ema"):
        trainer.ema_model.load_state_dict(checkpoint["ema_state_dict"])

    # Run validation
    val_acc, val_loss = trainer.validate()
    logger.info(f"Validation Results: Accuracy: {val_acc:.2f}%, Loss: {val_loss:.4f}")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    logger = setup_logging()

    # Print banner
    logger.info("=" * 80)
    logger.info("ResNet-50 ImageNet Training")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    logger.info("=" * 80)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Execute mode
    if args.mode == "train":
        mode_train(args, logger)
    elif args.mode == "distributed":
        mode_distributed(args, logger)
    elif args.mode == "convert-ffcv":
        mode_convert_ffcv(args, logger)
    elif args.mode == "test":
        mode_test(args, logger)
    elif args.mode == "validate":
        mode_validate(args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method
    if torch.cuda.is_available():
        mp.set_start_method("spawn", force=True)

    main()
