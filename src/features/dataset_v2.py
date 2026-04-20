"""Dataset utilities for the v2 218-dim processed_v2 parquet schema.

Uses lazy loading: only builds an index at init time, loads and normalizes
individual rounds on demand in __getitem__. This keeps memory usage low
even with hundreds of parquet files.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.state_vector_v2 import FEATURE_DIM, FEATURE_NAMES, build_state_matrix

LABEL_MAP: dict[str, int] = {"A": 0, "B": 1}


def split_files(
    parquet_files: list[Path],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split file paths into (train, val, test) at the demo level."""
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


# Module-level LRU cache for parquet DataFrames (keyed by file path string).
# Keeps up to 8 files in memory to avoid re-reading the same file repeatedly
# when sampling multiple rounds from the same demo.
@lru_cache(maxsize=8)
def _read_cached(path_str: str) -> pd.DataFrame:
    return pd.read_parquet(path_str)


class RoundSequenceDatasetV2(Dataset):
    """Lazy-loading dataset of per-round v2 state sequences.

    At init time, only scans parquet files to build a lightweight index of
    (file_path, round_key, label). The heavy work (reading data, normalizing
    features, building state matrices) happens on demand in __getitem__.
    """

    def __init__(
        self,
        parquet_files: list[Path],
        sequence_length: int = 720,
        training: bool = False,
    ) -> None:
        self._sequence_length = sequence_length
        self._training = training

        # Index: list of (parquet_path_str, demo_name, round_num, label_int)
        self._index: list[tuple[str, str, int, int]] = []

        for path in parquet_files:
            df = pd.read_parquet(path, columns=["demo_name", "round_num", "bomb_site"])
            for (demo_name, round_num), group in df.groupby(
                ["demo_name", "round_num"], sort=False
            ):
                site = str(group["bomb_site"].iloc[0])
                if site not in LABEL_MAP:
                    continue
                self._index.append(
                    (str(path), str(demo_name), int(round_num), LABEL_MAP[site])
                )

        # Expose labels for train.py's _label_counts()
        self._labels = [torch.tensor(entry[3], dtype=torch.long) for entry in self._index]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path_str, demo_name, round_num, label = self._index[idx]

        df = _read_cached(path_str)
        round_df = df[(df["demo_name"] == demo_name) & (df["round_num"] == round_num)]
        round_df = round_df.sort_values("step")

        n = min(len(round_df), self._sequence_length)
        mat = build_state_matrix(round_df.loc[:, FEATURE_NAMES].iloc[:n])
        real_len = len(mat)

        if self._training and real_len > 1:
            frac = np.random.uniform(0.0, 1.0)
            keep = max(1, int(real_len * frac))
        else:
            keep = real_len

        padded = np.zeros((self._sequence_length, FEATURE_DIM), dtype=np.float32)
        padded[:keep] = mat[:keep]
        return torch.from_numpy(padded), torch.tensor(label, dtype=torch.long)
