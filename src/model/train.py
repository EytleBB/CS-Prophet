"""Training loop for BombSiteTransformer — Focal Loss, AMP, gradient accumulation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.features.dataset import RoundSequenceDataset, split_files
from src.model.transformer import BombSiteTransformer

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017).

    Down-weights easy (well-classified) examples so training focuses on
    hard, misclassified ones. Reduces to cross-entropy when gamma=0.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean focal loss.

        Args:
            logits: (N, C) raw class scores.
            targets: (N,) integer class indices.

        Returns:
            Scalar mean loss.
        """
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float]:
    """Compute mean loss and top-1 accuracy over a DataLoader.

    Args:
        model: The model to evaluate.
        loader: DataLoader yielding (x, y) batches.
        criterion: Loss function.
        device: Target device.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        (mean_loss, accuracy) as Python floats.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(x)
                loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

    n = max(total, 1)
    return total_loss / n, correct / n


def train(config_path: str = "configs/train_config.yaml") -> None:
    """Run the full training loop using the YAML config at config_path.

    Saves the best checkpoint (by validation loss) to ``cfg.logging.save_dir/best.pt``.
    """
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.training.seed)
    logger.info("Training on %s", device)

    processed_dir = Path(cfg.data.processed_dir)
    parquet_files = sorted(processed_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {processed_dir}")

    train_files, val_files, _ = split_files(
        parquet_files,
        val_frac=cfg.training.val_split,
        test_frac=cfg.training.test_split,
        seed=cfg.training.seed,
    )

    train_ds = RoundSequenceDataset(train_files, cfg.data.sequence_length)
    val_ds = RoundSequenceDataset(val_files, cfg.data.sequence_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)

    model = BombSiteTransformer(
        input_dim=cfg.model.input_dim,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        num_classes=cfg.model.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = FocalLoss(gamma=cfg.training.focal_loss_gamma)
    use_amp = cfg.training.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum_steps: int = cfg.training.gradient_accumulation_steps

    save_dir = Path(cfg.logging.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(cfg.training.epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, criterion,
                                scaler, device, use_amp, accum_steps)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss},
                save_dir / "best.pt",
            )

        if (epoch + 1) % cfg.logging.log_interval == 0:
            logger.info(
                "Epoch %d | train=%.4f val=%.4f acc=%.3f",
                epoch + 1, train_loss, val_loss, val_acc,
            )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    accum_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            unscaled_loss = criterion(model(x), y)
            loss = unscaled_loss / accum_steps

        scaler.scale(loss).backward()

        if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += unscaled_loss.item() * len(y)

    return total_loss / max(len(loader.dataset), 1)
