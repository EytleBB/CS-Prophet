"""Dataset classes and file-split utilities for the bomb-site prediction task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.state_vector import FEATURE_DIM, build_state_vector

LABEL_MAP: dict[str, int] = {"A": 0, "B": 1}


def split_files(
    parquet_files: list[Path],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split file paths into (train, val, test) at the demo level.

    Splitting at file level prevents rounds from the same demo appearing
    in multiple sets (data leakage).

    Args:
        parquet_files: All available parquet paths.
        val_frac: Fraction for validation.
        test_frac: Fraction for test.
        seed: RNG seed for reproducibility.

    Returns:
        (train_files, val_files, test_files) — non-overlapping lists.
    """
    if len(parquet_files) < 3:
        raise ValueError(
            f"split_files requires at least 3 files; got {len(parquet_files)}"
        )
    rng = np.random.default_rng(seed)
    files = list(parquet_files)
    rng.shuffle(files)
    n = len(files)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    return files[n_test + n_val:], files[n_test: n_test + n_val], files[:n_test]


class RoundSequenceDataset(Dataset):
    """Dataset of per-round padded state sequences from parquet files.

    Each item is ``(sequence_tensor, label)`` where:
    - ``sequence_tensor``: float32 of shape ``(sequence_length, FEATURE_DIM)``
    - ``label``: torch.Tensor (dtype=torch.long) — 0=A, 1=B, 2=other

    Sequences shorter than ``sequence_length`` are zero-padded at the end.
    Sequences longer than ``sequence_length`` are truncated.
    """

    def __init__(self, parquet_files: list[Path], sequence_length: int = 720) -> None:
        self._sequences: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

        for path in parquet_files:
            df = pd.read_parquet(path)
            for (_, _), group in df.groupby(["demo_name", "round_num"], sort=False):
                site = str(group["bomb_site"].iloc[0])
                if site not in LABEL_MAP:
                    continue  # skip rounds without a confirmed A/B plant
                self._sequences.append(_build_padded_tensor(group, sequence_length))
                self._labels.append(torch.tensor(LABEL_MAP[site], dtype=torch.long))

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sequences[idx], self._labels[idx]


def _build_padded_tensor(group: pd.DataFrame, sequence_length: int) -> torch.Tensor:
    """Build a ``(sequence_length, FEATURE_DIM)`` float32 tensor from one round group."""
    rows = group.sort_values("step")
    n = min(len(rows), sequence_length)
    mat = np.zeros((sequence_length, FEATURE_DIM), dtype=np.float32)
    for i, (_, row) in enumerate(rows.iloc[:n].iterrows()):
        mat[i] = build_state_vector(row)
    return torch.from_numpy(mat)
