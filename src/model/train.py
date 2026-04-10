"""Training loop for BombSiteTransformer — Focal Loss, AMP, gradient accumulation."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    # Allow `python src/model/train.py` by exposing the repo root as an import base.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

            src_key_padding_mask = (x.abs().sum(dim=-1) == 0).to(device)
            with torch.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(x, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

    n = max(total, 1)
    return total_loss / n, correct / n


def _label_counts(ds: RoundSequenceDataset) -> dict[str, int]:
    counts = {"A": 0, "B": 0}
    for label in ds._labels:
        value = int(label.item())
        if value == 0:
            counts["A"] += 1
        elif value == 1:
            counts["B"] += 1
    return counts


def train(config_path: str = "configs/train_config.yaml") -> None:
    """Run the full training loop using the YAML config at config_path.

    Saves the best checkpoint (by validation loss) to ``cfg.logging.save_dir/best.pt``.
    """
    config_path = str(Path(config_path))
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.training.seed)
    logger.info("Training on %s", device)
    logger.info("Config: %s", config_path)

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
    train_counts = _label_counts(train_ds)
    val_counts = _label_counts(val_ds)
    batches_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(
        batches_per_epoch / cfg.training.gradient_accumulation_steps
    )
    logger.info(
        "Split: files train=%d val=%d test=%d | rounds train=%d val=%d",
        len(train_files),
        len(val_files),
        len(parquet_files) - len(train_files) - len(val_files),
        len(train_ds),
        len(val_ds),
    )
    logger.info(
        "Labels: train A=%d B=%d | val A=%d B=%d",
        train_counts["A"],
        train_counts["B"],
        val_counts["A"],
        val_counts["B"],
    )
    logger.info(
        "Batches/epoch=%d | optimizer updates/epoch=%d | grad_accum=%d",
        batches_per_epoch,
        updates_per_epoch,
        cfg.training.gradient_accumulation_steps,
    )

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
    best_epoch = -1
    patience: int = cfg.training.get("early_stop_patience", 0)
    no_improve = 0

    for epoch in range(cfg.training.epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, criterion,
                                scaler, device, use_amp, accum_steps)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "model_config": OmegaConf.to_container(cfg.model, resolve=True),
                },
                save_dir / "best.pt",
            )
        else:
            no_improve += 1

        if (
            (epoch + 1) % cfg.logging.log_interval == 0
            or epoch == 0
            or no_improve == 0
            or (patience > 0 and no_improve >= max(1, patience - 2))
        ):
            logger.info(
                "Epoch %d | train=%.4f val=%.4f acc=%.3f | best=%.4f@%d | patience=%d/%s",
                epoch + 1,
                train_loss,
                val_loss,
                val_acc,
                best_val_loss,
                best_epoch + 1,
                no_improve,
                patience if patience > 0 else "-",
            )

        if patience > 0 and no_improve >= patience:
            logger.info("Early stopping at epoch %d (no val improvement for %d epochs)",
                        epoch + 1, patience)
            break


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

        src_key_padding_mask = (x.abs().sum(dim=-1) == 0).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            unscaled_loss = criterion(model(x, src_key_padding_mask=src_key_padding_mask), y)
            loss = unscaled_loss / accum_steps

        scaler.scale(loss).backward()

        if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += unscaled_loss.item() * len(y)

    return total_loss / max(len(loader.dataset), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BombSiteTransformer")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to the training config YAML",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    train(args.config)
