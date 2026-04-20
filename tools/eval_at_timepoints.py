#!/usr/bin/env python3
"""Honest evaluation: accuracy vs. how early in the round we predict.

For each cutoff T (seconds since freeze-end), include only rounds whose
actual plant happens AFTER T, truncate the feature sequence at T, and
measure the model's accuracy on the truncated view. This avoids the
"read the last frame = plant moment" shortcut that inflates val acc.

Usage:
    python tools/eval_at_timepoints.py \
        --config configs/train_config_v2_2hz.yaml \
        --checkpoint checkpoints/v2_2hz/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.dataset_v2 import LABEL_MAP, split_files
from src.features.state_vector_v2 import FEATURE_DIM, FEATURE_NAMES, build_state_matrix
from src.model.transformer import BombSiteTransformer
from src.utils.paths import resolve_path_input

TICK_RATE = 64
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model at multiple pre-plant cutoffs.")
    p.add_argument("--config", default="configs/train_config_v2_2hz.yaml")
    p.add_argument("--checkpoint", default="checkpoints/v2_2hz/best.pt")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument(
        "--cutoffs",
        default="15,30,45,60,75,90,105,120",
        help="Comma-separated cutoff seconds since freeze-end.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--out-csv", default=None, help="Optional path to save per-cutoff CSV.")
    p.add_argument("--out-plot", default=None, help="Optional path to save matplotlib PNG.")
    return p.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> BombSiteTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BombSiteTransformer(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def iter_round_matrices(
    parquet_files: list[Path],
    sequence_length: int,
) -> list[tuple[np.ndarray, int, float]]:
    """Load every labeled A/B round as (features, label, plant_sec).

    plant_sec is seconds from freeze-end to plant (== last-tick - first-tick).
    """
    out: list[tuple[np.ndarray, int, float]] = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        for (demo_name, round_num), group in df.groupby(
            ["demo_name", "round_num"], sort=False
        ):
            site = str(group["bomb_site"].iloc[0])
            if site not in LABEL_MAP:
                continue
            group = group.sort_values("step")
            ticks = group["tick"].to_numpy()
            if len(ticks) < 2:
                continue
            plant_sec = float(ticks[-1] - ticks[0]) / TICK_RATE

            mat = build_state_matrix(group.loc[:, FEATURE_NAMES])
            out.append((mat, LABEL_MAP[site], plant_sec))
    return out


def evaluate_cutoff(
    rounds: list[tuple[np.ndarray, int, float]],
    cutoff_sec: float,
    sequence_length: int,
    target_tick_rate: float,
    model: BombSiteTransformer,
    device: torch.device,
    feature_dim: int,
    batch_size: int = 32,
) -> tuple[int, int]:
    """Return (correct, included) count for this cutoff.

    Only includes rounds where plant_sec > cutoff_sec. Truncates sequences
    to the first ``cutoff_sec`` seconds since freeze-end. One step spans
    ``1/target_tick_rate`` seconds, so keep = floor(cutoff_sec * rate) + 1.
    """
    kept: list[tuple[np.ndarray, int]] = []
    max_keep_sec = int(cutoff_sec * target_tick_rate) + 1
    for mat, label, plant_sec in rounds:
        if plant_sec <= cutoff_sec:
            continue
        n = len(mat)
        if n < 2:
            continue
        keep = min(n, max_keep_sec, sequence_length)
        padded = np.zeros((sequence_length, feature_dim), dtype=np.float32)
        padded[:keep] = mat[:keep]
        kept.append((padded, label))

    if not kept:
        return 0, 0

    correct = 0
    for i in range(0, len(kept), batch_size):
        chunk = kept[i : i + batch_size]
        x = torch.from_numpy(np.stack([c[0] for c in chunk])).to(device)
        y = torch.tensor([c[1] for c in chunk], dtype=torch.long, device=device)
        mask = (x.abs().sum(dim=-1) == 0)
        with torch.no_grad():
            logits = model(x, src_key_padding_mask=mask)
            preds = logits.argmax(dim=-1)
        correct += int((preds == y).sum().item())
    return correct, len(kept)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    processed_dir = resolve_path_input(str(cfg.data.processed_dir))
    parquet_files = sorted(processed_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {processed_dir}")

    train_files, val_files, test_files = split_files(
        parquet_files,
        val_frac=cfg.training.val_split,
        test_frac=cfg.training.test_split,
        seed=cfg.training.seed,
    )
    eval_files = test_files if args.split == "test" else val_files
    logger.info("Split=%s | files=%d", args.split, len(eval_files))

    ckpt_path = resolve_path_input(args.checkpoint)
    model = load_model(ckpt_path, device)
    feature_dim = int(cfg.model.input_dim)
    if feature_dim != FEATURE_DIM:
        logger.warning("Config input_dim=%d != FEATURE_DIM=%d", feature_dim, FEATURE_DIM)

    rounds = iter_round_matrices(eval_files, cfg.data.sequence_length)
    logger.info("Loaded %d labeled rounds", len(rounds))
    if not rounds:
        return 1

    cutoffs = [float(s) for s in args.cutoffs.split(",")]
    target_rate = float(cfg.data.get("target_tick_rate", 2))
    results: list[dict] = []
    for T in cutoffs:
        correct, n = evaluate_cutoff(
            rounds,
            T,
            int(cfg.data.sequence_length),
            target_rate,
            model,
            device,
            feature_dim,
        )
        acc = correct / n if n > 0 else float("nan")
        results.append({"cutoff_sec": T, "n_rounds": n, "correct": correct, "accuracy": acc})
        logger.info(
            "cutoff=%5.1fs  n=%4d  acc=%s",
            T,
            n,
            f"{acc:.4f}" if n > 0 else "N/A",
        )

    print()
    print(f"{'cutoff_s':>10} {'n':>6} {'acc':>8}")
    for r in results:
        acc_str = f"{r['accuracy']:.4f}" if r["n_rounds"] > 0 else "N/A"
        print(f"{r['cutoff_sec']:>10.1f} {r['n_rounds']:>6d} {acc_str:>8}")

    if args.out_csv:
        pd.DataFrame(results).to_csv(args.out_csv, index=False)
        logger.info("Saved CSV: %s", args.out_csv)

    if args.out_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping plot")
        else:
            df = pd.DataFrame(results)
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(df["cutoff_sec"], df["accuracy"], marker="o", color="tab:blue")
            ax1.set_xlabel("Seconds into round (cutoff)")
            ax1.set_ylabel("Accuracy", color="tab:blue")
            ax1.set_ylim(0.4, 1.0)
            ax1.grid(alpha=0.3)
            ax2 = ax1.twinx()
            ax2.bar(
                df["cutoff_sec"],
                df["n_rounds"],
                alpha=0.15,
                color="tab:gray",
                width=5.0,
            )
            ax2.set_ylabel("Rounds in sample", color="tab:gray")
            plt.title(f"Accuracy vs. prediction cutoff ({args.split} split)")
            plt.tight_layout()
            plt.savefig(args.out_plot, dpi=120)
            logger.info("Saved plot: %s", args.out_plot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
