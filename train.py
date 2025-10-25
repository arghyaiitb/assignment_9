#!/usr/bin/env python3
"""
Training module for ResNet-50 on ImageNet.
Supports both single-GPU and multi-GPU distributed training.
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel
from torch.amp import GradScaler, autocast
from torch.distributions.beta import Beta

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
import logging

# Import our modules
from models import ResNet50
from dataset import (
    get_ffcv_loaders,
    get_pytorch_loaders,
    FFCV_AVAILABLE,
    NUM_CLASSES,
)

# Training constants - can be overridden by environment variables or config
# These will be set dynamically based on config in the Trainer class
DEFAULT_CHECKPOINT_DIR = Path("./checkpoints")
DEFAULT_LOG_DIR = Path("./logs")
# Global variables for backwards compatibility
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", str(DEFAULT_CHECKPOINT_DIR)))
LOG_DIR = Path(os.environ.get("LOG_DIR", str(DEFAULT_LOG_DIR)))


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
        self.start_epoch = 0  # For resuming training
        self.global_step = 0  # Track global step for scheduler

        # Setup checkpoint and log directories from config or environment
        self.checkpoint_dir = Path(
            config.get("checkpoint_dir")
            or os.environ.get("CHECKPOINT_DIR", str(DEFAULT_CHECKPOINT_DIR))
        )
        self.log_dir = Path(
            config.get("log_dir") or os.environ.get("LOG_DIR", str(DEFAULT_LOG_DIR))
        )
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Distributed training state
        self.distributed = config.get("distributed", False)
        self.rank = config.get("rank", 0)
        self.world_size = config.get("world_size", 1)

        # Setup logging first (before device setup which logs)
        self._setup_logging()

        # Setup device
        self._setup_device()

        # Initialize components
        self._build_model()
        self._build_dataloaders()
        self._build_optimizer()
        self._build_scheduler()

        # Mixed precision
        if config.get("amp", True):
            self.scaler = GradScaler("cuda")

        # EMA model
        if config.get("use_ema", False) and self.rank == 0:
            self.ema_model = AveragedModel(
                self.model.module if self.distributed else self.model,
                avg_fn=lambda avg, model, num: 0.9999 * avg + 0.0001 * model,
            )

        # Resume from checkpoint if specified
        self._resume_from_checkpoint()

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

        # Create a persistent log file name (append mode for spot instances)
        # Use the node rank in filename for multi-node setups
        log_filename = "train.log" if self.rank == 0 else f"train_rank{self.rank}.log"
        log_path = self.log_dir / log_filename

        # Also create an epoch-specific log for easy tracking
        epoch_log_path = self.log_dir / "epoch_progress.log"

        handlers = [
            logging.FileHandler(log_path, mode="a"),  # Append mode for persistence
            logging.StreamHandler(),
        ]

        # Add epoch progress logger for rank 0
        if self.rank == 0:
            self.epoch_logger = logging.getLogger("epoch_progress")
            self.epoch_logger.setLevel(logging.INFO)
            epoch_handler = logging.FileHandler(epoch_log_path, mode="a")
            epoch_handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
                )
            )
            self.epoch_logger.addHandler(epoch_handler)
            self.epoch_logger.propagate = False

        logging.basicConfig(
            level=log_level,
            format="[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
            force=True,  # Override any existing configuration
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

        # Log multi-GPU setup details
        if self.distributed and self.rank == 0:
            self.logger.info("\n🚀 Distributed Training Configuration:")
            self.logger.info(f"   World Size: {self.world_size} GPUs")
            self.logger.info(f"   Backend: {self.config.get('dist_backend', 'nccl')}")
            self.logger.info(f"   Total Batch Size: {self.config['batch_size']}")
            self.logger.info(
                f"   Per-GPU Batch Size: {self.config['batch_size'] // self.world_size}"
            )
            self.logger.info(f"   Base LR: {self.config.get('lr', 0.1):.4f}")
            self.logger.info(f"   Effective LR (scaled): {base_lr:.4f}")

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        epochs = self.config["epochs"]
        scheduler_type = self.config.get("scheduler", "onecycle")
        warmup_epochs = self.config.get("warmup_epochs", 5)

        if scheduler_type == "onecycle":
            total_steps = epochs * len(self.train_loader)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]["lr"],
                total_steps=total_steps,
                pct_start=warmup_epochs / epochs if warmup_epochs > 0 else 0.25,
                div_factor=25,
                final_div_factor=10000,
                anneal_strategy="cos",
            )
            self.logger.info(
                f"Using OneCycleLR scheduler with {warmup_epochs} warmup epochs"
            )
        elif scheduler_type == "cosine":
            # For cosine scheduler, we'll implement warmup manually
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=(epochs - warmup_epochs) * len(self.train_loader),
                eta_min=1e-6,
            )
            self.warmup_steps = warmup_epochs * len(self.train_loader)
            self.warmup_scheduler = None
            if warmup_epochs > 0:
                # Create a linear warmup scheduler
                def warmup_lambda(step):
                    if step < self.warmup_steps:
                        return float(step) / float(max(1, self.warmup_steps))
                    return 1.0

                self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=warmup_lambda
                )
            self.logger.info(
                f"Using CosineAnnealingLR scheduler with {warmup_epochs} warmup epochs"
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

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        # Progress bar only on rank 0
        pbar = tqdm(self.train_loader) if self.rank == 0 else self.train_loader

        for batch_idx, batch in enumerate(pbar):
            # Handle both FFCV and standard dataloaders
            # Check batch type instead of just relying on flag
            if isinstance(batch, dict):
                # FFCV loader returns dict
                images = batch["image"]
                labels = batch["label"].long()
            else:
                # Standard PyTorch loader returns tuple
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
                with autocast("cuda"):
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

            # Update scheduler (must be after optimizer.step())
            if self.scheduler is not None:
                self.scheduler.step()
                self.global_step += 1

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

        # Aggregate training metrics across all GPUs if distributed
        if self.distributed:
            # Convert metrics to tensors for all-reduce
            metrics = torch.tensor(
                [correct, total, total_loss, len(self.train_loader)],
                dtype=torch.float32,
                device=self.device,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

            # Extract aggregated values
            correct = metrics[0].item()
            total = metrics[1].item()
            total_loss = metrics[2].item()
            num_batches = metrics[3].item()

            # Recalculate with global values
            avg_loss = total_loss / num_batches
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
                # Check batch type instead of just relying on flag
                if isinstance(batch, dict):
                    # FFCV loader returns dict
                    images = batch["image"]
                    labels = batch["label"].long()
                else:
                    # Standard PyTorch loader returns tuple
                    images, labels = batch
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                # Use autocast for validation to match training precision
                if self.scaler is not None:
                    with autocast("cuda"):
                        outputs = model(images)
                        loss = F.cross_entropy(outputs, labels)
                else:
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0

        # Aggregate metrics across all GPUs if distributed
        if self.distributed:
            # Convert to tensors for all-reduce
            metrics = torch.tensor(
                [correct, total, total_loss], dtype=torch.float32, device=self.device
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

            # Extract aggregated values
            correct = metrics[0].item()
            total = metrics[1].item()
            total_loss = metrics[2].item()

            # Recalculate with global values
            avg_loss = total_loss / (len(self.val_loader) * self.world_size)
            accuracy = 100.0 * correct / total if total > 0 else 0

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
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
        }

        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

        # Save regular checkpoint
        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Also save as 'latest' for easy auto-resume
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(
                f"Saved best model with accuracy: {self.best_accuracy:.2f}%"
            )

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint file for auto-resume."""
        # First check for explicit latest checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest_path.exists():
            return latest_path

        # Otherwise find the highest epoch checkpoint
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
            return checkpoints[-1]

        return None

    def _resume_from_checkpoint(self):
        """Resume training from checkpoint if specified or auto-detect."""
        checkpoint_path = None

        # Check if explicit resume path is provided
        if self.config.get("resume"):
            checkpoint_path = Path(self.config["resume"])
            if not checkpoint_path.exists():
                self.logger.error(f"Checkpoint not found: {checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Auto-resume: check for latest checkpoint if enabled
        elif self.config.get("auto_resume", True):
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path:
                self.logger.info(f"Auto-resume: Found checkpoint at {checkpoint_path}")

        # Load checkpoint if found
        if checkpoint_path and checkpoint_path.exists():
            self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            model_state = checkpoint["model_state_dict"]
            if self.distributed:
                self.model.module.load_state_dict(model_state)
            else:
                self.model.load_state_dict(model_state)

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load scheduler state if available
            if self.scheduler and checkpoint.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Load scaler state if available
            if self.scaler and checkpoint.get("scaler_state_dict"):
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            # Load EMA model state if available
            if self.ema_model and checkpoint.get("ema_state_dict"):
                self.ema_model.load_state_dict(checkpoint["ema_state_dict"])

            # Restore training state
            self.start_epoch = checkpoint["epoch"] + 1
            self.current_epoch = checkpoint["epoch"]
            self.best_accuracy = checkpoint.get("best_accuracy", 0.0)

            self.logger.info(f"Resumed from epoch {self.start_epoch}")
            self.logger.info(f"Best accuracy so far: {self.best_accuracy:.2f}%")

    def train(self):
        """Main training loop."""
        epochs = self.config["epochs"]

        for epoch in range(self.start_epoch, epochs):
            self.current_epoch = epoch

            # Update data loaders for progressive resizing
            if self.config.get("progressive_resize", False) and self.use_ffcv:
                new_size = self._get_image_size(epoch)
                if epoch == 0 or new_size != self._get_image_size(epoch - 1):
                    if self.rank == 0:
                        self.logger.info(f"Updating image size to {new_size}")
                    self._build_dataloaders()
                    # Ensure all ranks finish rebuilding loaders before continuing
                    if self.distributed:
                        dist.barrier()

            # Set epoch for distributed sampler
            if self.distributed and hasattr(self.train_loader, "sampler"):
                self.train_loader.sampler.set_epoch(epoch)

            # Synchronize all ranks before starting epoch
            if self.distributed:
                dist.barrier()

            # Train for one epoch
            if self.rank == 0:
                self.logger.info(f"\n{'=' * 80}")
                self.logger.info(f"Epoch {epoch + 1}/{epochs}")
                self.logger.info(f"{'=' * 80}")

            train_acc, train_loss = self.train_epoch()

            # Validation - all ranks participate but only rank 0 saves
            val_acc, val_loss = self.validate()

            # Synchronize all ranks after validation before any rank-specific operations
            if self.distributed:
                dist.barrier()

            if self.rank == 0:
                # Log results
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
                )
                self.logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                self.logger.info(f"Learning Rate: {current_lr:.6f}")

                # Also log to epoch progress file for easy monitoring
                if hasattr(self, "epoch_logger"):
                    self.epoch_logger.info(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Train: {train_acc:.2f}% (loss: {train_loss:.4f}) | "
                        f"Val: {val_acc:.2f}% (loss: {val_loss:.4f}) | "
                        f"LR: {current_lr:.6f} | "
                        f"Best: {self.best_accuracy:.2f}%"
                    )

                # Save checkpoint
                is_best = val_acc > self.best_accuracy
                if is_best:
                    self.best_accuracy = val_acc

                if (epoch + 1) % self.config.get(
                    "checkpoint_interval", 5
                ) == 0 or is_best:
                    self.save_checkpoint(is_best)

            # Early stopping check - only rank 0 decides
            if self.rank == 0:
                should_stop = val_acc >= self.config.get("target_accuracy", 100)
                if should_stop:
                    self.logger.info(
                        f"Target accuracy {self.config['target_accuracy']}% reached!"
                    )
            else:
                should_stop = False

            # Broadcast the stopping decision to all ranks
            if self.distributed:
                # Create tensor on all ranks
                if self.rank == 0:
                    stop_tensor = torch.tensor(
                        [1.0 if should_stop else 0.0], device=self.device
                    )
                else:
                    stop_tensor = torch.tensor([0.0], device=self.device)

                # All ranks participate in broadcast
                dist.broadcast(stop_tensor, src=0)

                # All ranks get the result
                should_stop = stop_tensor.item() > 0.5

            if should_stop:
                break

        if self.rank == 0:
            self.logger.info(
                f"\nTraining completed! Best accuracy: {self.best_accuracy:.2f}%"
            )


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training environment."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    # Set device before init_process_group to avoid warnings
    torch.cuda.set_device(rank)

    # Initialize process group
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    # Initial synchronization
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
