#!/usr/bin/env python3
"""
Training module for ResNet-50 on ImageNet.
Supports both single-GPU and multi-GPU distributed training.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel
from torch.cuda.amp import GradScaler, autocast
from torch.distributions.beta import Beta

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
import logging
from datetime import datetime

# Import our modules
from models import ResNet50
from dataset import (
    get_ffcv_loaders,
    get_pytorch_loaders,
    FFCV_AVAILABLE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
)

# Training constants
CHECKPOINT_DIR = Path("./checkpoints")
LOG_DIR = Path("./logs")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


class Trainer:
    """Main trainer class that handles both single and distributed training."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.ema_model = None
        self.train_loader = None
        self.val_loader = None
        self.best_accuracy = 0.0
        self.current_epoch = 0

        # Distributed training state
        self.distributed = config.get("distributed", False)
        self.rank = config.get("rank", 0)
        self.world_size = config.get("world_size", 1)

        # Setup device
        self._setup_device()

        # Setup logging
        self._setup_logging()

        # Initialize components
        self._build_model()
        self._build_dataloaders()
        self._build_optimizer()
        self._build_scheduler()

        # Mixed precision
        if config.get("amp", True):
            self.scaler = GradScaler()

        # EMA model
        if config.get("use_ema", False) and self.rank == 0:
            self.ema_model = AveragedModel(
                self.model.module if self.distributed else self.model,
                avg_fn=lambda avg, model, num: 0.9999 * avg + 0.0001 * model,
            )

    def _setup_device(self):
        """Setup CUDA device for training."""
        if self.distributed:
            self.device = torch.device(f"cuda:{self.rank}")
            torch.cuda.set_device(self.rank)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Using device: {self.device}")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO if self.rank == 0 else logging.WARNING

        logging.basicConfig(
            level=log_level,
            format="[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(
                    LOG_DIR / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
                ),
                logging.StreamHandler(),
            ],
        )

        self.logger = logging.getLogger(__name__)

        if self.rank == 0:
            self.logger.info("=" * 80)
            self.logger.info("ResNet-50 ImageNet Training")
            self.logger.info("=" * 80)
            self.logger.info(f"Configuration: {self.config}")

    def _build_model(self):
        """Build and initialize the model."""
        self.logger.info("Building ResNet-50 model...")

        # Create model
        self.model = ResNet50(num_classes=NUM_CLASSES).to(self.device)

        # Use channels_last memory format
        self.model = self.model.to(memory_format=torch.channels_last)

        # Compile model if requested (PyTorch 2.0+)
        if self.config.get("compile", False) and torch.__version__ >= "2.0":
            self.logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="default")

        # Wrap with DDP for distributed training
        if self.distributed:
            self.model = DDP(
                self.model, device_ids=[self.rank], output_device=self.rank
            )

        # Log model info
        if self.rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def _build_dataloaders(self):
        """Build data loaders for training and validation."""
        batch_size = self.config["batch_size"]
        num_workers = self.config.get("num_workers", 8)

        # Adjust batch size for distributed training
        if self.distributed:
            batch_size = batch_size // self.world_size
            self.logger.info(f"Batch size per GPU: {batch_size}")

        # Use FFCV if available and requested
        if self.config.get("use_ffcv", False) and FFCV_AVAILABLE:
            self.logger.info("Using FFCV data loaders...")
            self.train_loader, self.val_loader = get_ffcv_loaders(
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=self._get_image_size(0),
                ffcv_dir=self.config.get("ffcv_dir", "/datasets/ffcv"),
                distributed=self.distributed,
                seed=self.config.get("seed", 42) + self.rank,
            )
            self.use_ffcv = True
        else:
            self.logger.info("Using standard PyTorch data loaders...")
            self.train_loader, self.val_loader = get_pytorch_loaders(
                batch_size=batch_size,
                num_workers=num_workers,
                cache_dir=self.config.get("cache_dir"),
                test_mode=self.config.get("test_mode", False),
                max_samples=self.config.get("max_samples"),
                distributed=self.distributed,
            )
            self.use_ffcv = False

        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")

    def _build_optimizer(self):
        """Build optimizer."""
        # Scale learning rate by world size for distributed training
        base_lr = self.config.get("lr", 0.1)
        if self.distributed:
            base_lr *= self.world_size

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=base_lr,
            momentum=self.config.get("momentum", 0.9),
            weight_decay=self.config.get("weight_decay", 1e-4),
            nesterov=True,
        )

        self.logger.info(
            f"Optimizer: SGD (lr={base_lr}, momentum={self.config.get('momentum', 0.9)})"
        )

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        epochs = self.config["epochs"]
        scheduler_type = self.config.get("scheduler", "onecycle")

        if scheduler_type == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]["lr"],
                epochs=epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.25,
                div_factor=25,
                final_div_factor=10000,
                anneal_strategy="cos",
            )
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=epochs * len(self.train_loader), eta_min=1e-6
            )
        else:
            self.scheduler = None

        self.logger.info(f"Scheduler: {scheduler_type}")

    def _get_image_size(self, epoch: int) -> int:
        """Get image size for progressive resizing."""
        if not self.config.get("progressive_resize", False):
            return 224

        # Progressive resizing schedule
        if epoch < 30:
            return 160
        elif epoch < 60:
            return 192
        else:
            return 224

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        # Progress bar only on rank 0
        pbar = tqdm(self.train_loader) if self.rank == 0 else self.train_loader

        for batch_idx, batch in enumerate(pbar):
            # Handle both FFCV and standard dataloaders
            if self.use_ffcv:
                images = batch["image"]
                labels = batch["label"].long()
            else:
                images, labels = batch
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

            # Apply mixup/cutmix augmentation
            if (
                self.config.get("cutmix_prob", 0) > 0
                and np.random.rand() < self.config["cutmix_prob"]
            ):
                images, labels = self._cutmix(images, labels)
            elif self.config.get("mixup_alpha", 0) > 0 and np.random.rand() < 0.5:
                images, labels = self._mixup(images, labels)

            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    if isinstance(labels, tuple):  # Mixed labels from cutmix/mixup
                        loss = labels[2] * F.cross_entropy(outputs, labels[0]) + (
                            1 - labels[2]
                        ) * F.cross_entropy(outputs, labels[1])
                    else:
                        loss = F.cross_entropy(
                            outputs,
                            labels,
                            label_smoothing=self.config.get("label_smoothing", 0.1),
                        )

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.get("gradient_clip", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clip"]
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without mixed precision
                outputs = self.model(images)
                if isinstance(labels, tuple):  # Mixed labels
                    loss = labels[2] * F.cross_entropy(outputs, labels[0]) + (
                        1 - labels[2]
                    ) * F.cross_entropy(outputs, labels[1])
                else:
                    loss = F.cross_entropy(
                        outputs,
                        labels,
                        label_smoothing=self.config.get("label_smoothing", 0.1),
                    )

                self.optimizer.zero_grad()
                loss.backward()

                if self.config.get("gradient_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clip"]
                    )

                self.optimizer.step()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Update EMA
            if self.ema_model is not None and batch_idx % 10 == 0:
                self.ema_model.update_parameters(
                    self.model.module if self.distributed else self.model
                )

            # Update metrics
            total_loss += loss.item()
            if not isinstance(labels, tuple):
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Update progress bar
            if self.rank == 0 and hasattr(pbar, "set_description"):
                if total > 0:
                    pbar.set_description(
                        f"Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.2f}%"
                    )

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0

        return accuracy, avg_loss

    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, disable=self.rank != 0):
                if self.use_ffcv:
                    images = batch["image"]
                    labels = batch["label"].long()
                else:
                    images, labels = batch
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return accuracy, avg_loss

    def _cutmix(self, images, labels):
        """Apply CutMix augmentation."""
        batch_size = images.size(0)
        beta = Beta(1.0, 1.0)
        lam = beta.sample().item()

        index = torch.randperm(batch_size).to(self.device)

        # Generate random box
        H, W = images.size(2), images.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply CutMix
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Adjust lambda
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

        return images, (labels, labels[index], lam)

    def _mixup(self, images, labels):
        """Apply MixUp augmentation."""
        batch_size = images.size(0)
        alpha = self.config.get("mixup_alpha", 0.2)

        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(batch_size).to(self.device)
        mixed_images = lam * images + (1 - lam) * images[index]

        return mixed_images, (labels, labels[index], lam)

    def save_checkpoint(self, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": (
                self.model.module if self.distributed else self.model
            ).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_accuracy": self.best_accuracy,
            "config": self.config,
        }

        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

        # Save regular checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(
                f"Saved best model with accuracy: {self.best_accuracy:.2f}%"
            )

    def train(self):
        """Main training loop."""
        epochs = self.config["epochs"]

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Update data loaders for progressive resizing
            if self.config.get("progressive_resize", False) and self.use_ffcv:
                new_size = self._get_image_size(epoch)
                if epoch == 0 or new_size != self._get_image_size(epoch - 1):
                    self.logger.info(f"Updating image size to {new_size}")
                    self._build_dataloaders()

            # Set epoch for distributed sampler
            if self.distributed and hasattr(self.train_loader, "sampler"):
                self.train_loader.sampler.set_epoch(epoch)

            # Train for one epoch
            if self.rank == 0:
                self.logger.info(f"\n{'=' * 80}")
                self.logger.info(f"Epoch {epoch + 1}/{epochs}")
                self.logger.info(f"{'=' * 80}")

            train_acc, train_loss = self.train_epoch()

            # Validation
            if self.rank == 0:
                val_acc, val_loss = self.validate()

                # Log results
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
                )
                self.logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                self.logger.info(f"Learning Rate: {current_lr:.6f}")

                # Save checkpoint
                is_best = val_acc > self.best_accuracy
                if is_best:
                    self.best_accuracy = val_acc

                if (epoch + 1) % self.config.get(
                    "checkpoint_interval", 5
                ) == 0 or is_best:
                    self.save_checkpoint(is_best)

                # Early stopping
                if val_acc >= self.config.get("target_accuracy", 100):
                    self.logger.info(
                        f"Target accuracy {self.config['target_accuracy']}% reached!"
                    )
                    break

            # Synchronize processes
            if self.distributed:
                dist.barrier()

        if self.rank == 0:
            self.logger.info(
                f"\nTraining completed! Best accuracy: {self.best_accuracy:.2f}%"
            )


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training environment."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    torch.cuda.set_device(rank)
    dist.barrier()


def cleanup_distributed():
    """Cleanup distributed training environment."""
    dist.destroy_process_group()


def train_distributed(rank: int, world_size: int, config: Dict):
    """Training function for distributed training."""
    # Setup distributed environment
    setup_distributed(rank, world_size)

    # Update config with distributed settings
    config["distributed"] = True
    config["rank"] = rank
    config["world_size"] = world_size

    try:
        # Create trainer and start training
        trainer = Trainer(config)
        trainer.train()
    finally:
        cleanup_distributed()


def train_single_gpu(config: Dict):
    """Training function for single GPU."""
    config["distributed"] = False
    config["rank"] = 0
    config["world_size"] = 1

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    # Example usage
    config = {
        "batch_size": 256,
        "epochs": 100,
        "lr": 0.1,
        "use_ffcv": True,
        "progressive_resize": True,
        "use_ema": True,
        "compile": True,
        "cutmix_prob": 0.5,
        "mixup_alpha": 0.4,
        "label_smoothing": 0.1,
        "gradient_clip": 1.0,
        "checkpoint_interval": 5,
        "target_accuracy": 78.0,
    }

    # Single GPU training
    train_single_gpu(config)
