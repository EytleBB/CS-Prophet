"""Tests for dataset — RoundSequenceDataset and split_files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.dataset import LABEL_MAP, RoundSequenceDataset, split_files
from src.features.state_vector import FEATURE_DIM


def _make_parquet(
    tmp_path: Path,
    filename: str = "demo.parquet",
    n_rounds: int = 2,
    n_steps: int = 240,
    bomb_site: str = "A",
) -> Path:
    rows = []
    for rnd in range(1, n_rounds + 1):
        for step in range(n_steps):
            row: dict = {
                "demo_name": filename.replace(".parquet", ""),
                "round_num": rnd,
                "step": step,
                "tick": step * 8,
                "bomb_site": bomb_site,
            }
            for side in ("t", "ct"):
                for i in range(5):
                    row[f"{side}{i}_x"] = 0.5
                    row[f"{side}{i}_y"] = 0.5
                    row[f"{side}{i}_z"] = 0.5
                    row[f"{side}{i}_hp"] = 100
                    row[f"{side}{i}_armor"] = 100
                    row[f"{side}{i}_helmet"] = True
                    row[f"{side}{i}_alive"] = True
            rows.append(row)
    path = tmp_path / filename
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


class TestSplitFiles:
    def test_total_files_preserved(self):
        files = [Path(f"demo_{i}.parquet") for i in range(10)]
        train, val, test = split_files(files)
        assert len(train) + len(val) + len(test) == 10

    def test_sets_non_overlapping(self):
        files = [Path(f"demo_{i}.parquet") for i in range(20)]
        train, val, test = split_files(files)
        assert not set(train) & set(val)
        assert not set(train) & set(test)
        assert not set(val) & set(test)

    def test_reproducible_with_seed(self):
        files = [Path(f"demo_{i}.parquet") for i in range(10)]
        a = split_files(files, seed=42)
        b = split_files(files, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        files = [Path(f"demo_{i}.parquet") for i in range(20)]
        a_train, _, _ = split_files(files, seed=0)
        b_train, _, _ = split_files(files, seed=99)
        assert a_train != b_train

    def test_val_test_sizes_approximate(self):
        files = [Path(f"demo_{i}.parquet") for i in range(100)]
        train, val, test = split_files(files, val_frac=0.1, test_frac=0.1)
        assert abs(len(val) - 10) <= 1
        assert abs(len(test) - 10) <= 1

    def test_too_few_files_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            split_files([Path("a.parquet"), Path("b.parquet")])


class TestRoundSequenceDataset:
    def test_len_equals_round_count(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=3, n_steps=10)
        ds = RoundSequenceDataset([path], sequence_length=720)
        assert len(ds) == 3

    def test_sequence_shape(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, n_steps=120)
        ds = RoundSequenceDataset([path], sequence_length=720)
        seq, label = ds[0]
        assert seq.shape == (720, FEATURE_DIM)
        assert seq.dtype == torch.float32
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long

    def test_short_sequence_padded_with_zeros(self, tmp_path):
        n_steps = 50
        path = _make_parquet(tmp_path, n_rounds=1, n_steps=n_steps)
        ds = RoundSequenceDataset([path], sequence_length=720)
        seq, _ = ds[0]
        assert seq[n_steps - 1].abs().sum().item() > 0
        assert seq[n_steps:].abs().sum().item() == 0.0

    def test_long_sequence_truncated(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, n_steps=300)
        ds = RoundSequenceDataset([path], sequence_length=720)
        seq, _ = ds[0]
        assert seq.shape[0] == 720

    def test_label_A_is_0(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, bomb_site="A")
        ds = RoundSequenceDataset([path], sequence_length=720)
        _, label = ds[0]
        assert label == LABEL_MAP["A"]

    def test_label_B_is_1(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, bomb_site="B")
        ds = RoundSequenceDataset([path], sequence_length=720)
        _, label = ds[0]
        assert label == LABEL_MAP["B"]

    def test_other_rounds_skipped(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, bomb_site="other")
        ds = RoundSequenceDataset([path], sequence_length=720)
        assert len(ds) == 0

    def test_multiple_files_concatenated(self, tmp_path):
        p1 = _make_parquet(tmp_path, "demo1.parquet", n_rounds=2)
        p2 = _make_parquet(tmp_path, "demo2.parquet", n_rounds=3)
        ds = RoundSequenceDataset([p1, p2], sequence_length=720)
        assert len(ds) == 5

    def test_values_in_unit_range(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, n_steps=10)
        ds = RoundSequenceDataset([path], sequence_length=720)
        seq, _ = ds[0]
        assert seq.min().item() >= 0.0
        assert seq.max().item() <= 1.0
