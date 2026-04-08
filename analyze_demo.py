#!/usr/bin/env python3
"""Offline demo analyzer — replay a parsed parquet and output bomb-site predictions every 10 s.

Usage:
    python analyze_demo.py data/processed/some_demo.parquet
    python analyze_demo.py data/processed/some_demo.parquet --rounds 1 3 5
    python analyze_demo.py data/processed/some_demo.parquet --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.state_vector import FEATURE_DIM, build_state_vector
from src.inference.predictor import RoundPredictor

_TARGET_RATE = 8    # ticks/sec (after downsampling)
_MAX_STEPS   = 720  # 90 s × 8
_INTERVALS   = [0, 80, 240, 480, 719]  # steps corresponding to t=0,10,30,60,90 s


def _build_partial(group: pd.DataFrame, up_to_step: int) -> np.ndarray:
    """Build a (MAX_STEPS, FEATURE_DIM) matrix with only the first up_to_step+1 rows filled."""
    mat = np.zeros((_MAX_STEPS, FEATURE_DIM), dtype=np.float32)
    rows = group.sort_values("step")
    for i, (_, row) in enumerate(rows.iterrows()):
        if i > up_to_step or i >= _MAX_STEPS:
            break
        mat[i] = build_state_vector(row)
    return mat


def analyze(
    parquet_path: Path,
    checkpoint: Path,
    device: str = "cpu",
    rounds: list[int] | None = None,
) -> None:
    predictor = RoundPredictor(checkpoint, device=device)
    df = pd.read_parquet(parquet_path)
    demo_name = df["demo_name"].iloc[0] if "demo_name" in df.columns else parquet_path.stem

    print(f"\nDemo : {demo_name}")
    print(f"Rounds: {df['round_num'].nunique()}  |  checkpoint: {checkpoint}")
    print("=" * 62)

    for round_num, group in df.groupby("round_num", sort=True):
        if rounds and round_num not in rounds:
            continue

        group = group.sort_values("step").reset_index(drop=True)
        actual = group["bomb_site"].iloc[-1]
        n_steps = len(group)

        print(f"\nRound {round_num:>2}  actual={actual}  ({n_steps} steps)")
        print(f"  {'time':>4}  {'P(A)':>6}  {'P(B)':>6}  prediction")
        print(f"  {'-'*4}  {'-'*6}  {'-'*6}  ----------")

        for step in _INTERVALS:
            available = min(step, n_steps - 1)
            partial = _build_partial(group, available)
            probs = predictor.predict(partial)

            pred = "A" if probs["A"] >= probs["B"] else "B"
            correct = "✓" if pred == actual else "✗"
            t_sec = available // _TARGET_RATE

            print(
                f"  {t_sec:>3}s"
                f"  {probs['A']:>5.1%}"
                f"  {probs['B']:>5.1%}"
                f"  {pred} {correct}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline bomb-site prediction replay.")
    parser.add_argument("parquet", help="Path to parsed .parquet file")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Model checkpoint")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--rounds", type=int, nargs="+", metavar="N",
                        help="Only show these round numbers (default: all)")
    args = parser.parse_args()

    analyze(
        Path(args.parquet),
        Path(args.checkpoint),
        device=args.device,
        rounds=args.rounds,
    )


if __name__ == "__main__":
    main()
