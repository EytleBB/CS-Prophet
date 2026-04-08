# CS-Prophet Phase 2: Feature Engineering + Model Training

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Phase 1 parquet output into trained `BombSiteTransformer` checkpoints that predict bomb plant site (A / B / other) from 30-second pre-plant sequences.

**Architecture:** A 74-dim state vector (5T + 5CT player features + map zone) feeds a cross-attention Transformer: T-side features query CT-side features to model adversarial interaction, then a stacked self-attention encoder produces a 3-class prediction from the last token. Training uses Focal Loss (γ=2), mixed-precision AMP, and gradient accumulation.

**Tech Stack:** PyTorch ≥ 2.2, pandas, numpy, pyarrow, omegaconf, tqdm. All modules are unit-tested with pytest before integration.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/features/state_vector.py` | `build_state_vector(row) → np.ndarray` (74-dim); `FEATURE_DIM` constant |
| Create | `src/features/dataset.py` | `RoundSequenceDataset`, `split_files`, `_build_padded_tensor` |
| Modify | `src/model/attention.py` | `PositionalEncoding`, `CrossAttentionLayer` |
| Modify | `src/model/transformer.py` | Replace `RoundTransformer` with `BombSiteTransformer` (3-class, cross-attn) |
| Modify | `src/model/train.py` | `FocalLoss`, `train()`, `evaluate()`, `_run_epoch()` |
| Modify | `configs/train_config.yaml` | `input_dim: 74` (was 84) |
| Create | `tests/test_state_vector.py` | Unit tests for `build_state_vector` |
| Create | `tests/test_dataset.py` | Unit tests for `RoundSequenceDataset` and `split_files` |
| Create | `tests/test_attention.py` | Unit tests for `PositionalEncoding` and `CrossAttentionLayer` |
| Modify | `tests/test_model.py` | Replace `RoundTransformer` tests with `BombSiteTransformer` tests |
| Create | `tests/test_train.py` | Unit tests for `FocalLoss` and `evaluate` |

### Feature vector layout (74 dims)

```
[0:35]  T players 0–4: (x, y, z, hp/100, armor/100, helmet, alive) × 5
[35:70] CT players 0–4: same × 5
[70:74] map_zone one-hot: [A=0, B=1, mid=2, other=3]
```

Coordinates are already `[0, 1]` from Phase 1 parser. `hp` and `armor` are raw `0–100` ints → divide by 100.

---

## Task 1: `src/features/state_vector.py` (TDD)

**Files:**
- Modify: `src/features/state_vector.py`
- Create: `tests/test_state_vector.py`

### 1A — Write failing tests

- [ ] **Step 1: Create `tests/test_state_vector.py`**

```python
"""Tests for state_vector — single-row feature vector builder."""

import numpy as np
import pandas as pd
import pytest
from src.features.state_vector import FEATURE_DIM, build_state_vector


def _make_row(map_zone: str = "A", **overrides) -> pd.Series:
    """Build a minimal parquet row with sensible defaults."""
    data: dict = {"map_zone": map_zone}
    for side in ("t", "ct"):
        for i in range(5):
            data[f"{side}{i}_x"] = 0.5
            data[f"{side}{i}_y"] = 0.5
            data[f"{side}{i}_z"] = 0.5
            data[f"{side}{i}_hp"] = 100
            data[f"{side}{i}_armor"] = 100
            data[f"{side}{i}_helmet"] = True
            data[f"{side}{i}_alive"] = True
    data.update(overrides)
    return pd.Series(data)


class TestFeatureDim:
    def test_constant_is_74(self):
        assert FEATURE_DIM == 74


class TestBuildStateVector:
    def test_output_shape(self):
        vec = build_state_vector(_make_row())
        assert vec.shape == (74,)

    def test_dtype_is_float32(self):
        vec = build_state_vector(_make_row())
        assert vec.dtype == np.float32

    def test_hp_normalised_to_1(self):
        vec = build_state_vector(_make_row(t0_hp=100))
        # t0 hp is at index 3 (x=0, y=1, z=2, hp=3)
        assert vec[3] == pytest.approx(1.0)

    def test_hp_normalised_to_half(self):
        vec = build_state_vector(_make_row(t0_hp=50))
        assert vec[3] == pytest.approx(0.5)

    def test_armor_normalised(self):
        vec = build_state_vector(_make_row(t0_armor=60))
        # t0 armor is at index 4
        assert vec[4] == pytest.approx(0.6)

    def test_ct_player_offset(self):
        # ct0 hp at index 35 + 3 = 38
        vec = build_state_vector(_make_row(ct0_hp=80))
        assert vec[38] == pytest.approx(0.8)

    def test_zone_one_hot_A(self):
        vec = build_state_vector(_make_row(map_zone="A"))
        assert vec[70] == 1.0
        assert vec[71] == 0.0
        assert vec[72] == 0.0
        assert vec[73] == 0.0

    def test_zone_one_hot_B(self):
        vec = build_state_vector(_make_row(map_zone="B"))
        assert vec[70] == 0.0
        assert vec[71] == 1.0

    def test_zone_one_hot_mid(self):
        vec = build_state_vector(_make_row(map_zone="mid"))
        assert vec[72] == 1.0

    def test_zone_one_hot_other(self):
        vec = build_state_vector(_make_row(map_zone="other"))
        assert vec[73] == 1.0

    def test_all_values_in_unit_range(self):
        vec = build_state_vector(_make_row())
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_dead_player_alive_false(self):
        vec = build_state_vector(_make_row(t0_alive=False))
        # t0 alive at index 6
        assert vec[6] == 0.0

    def test_alive_player_alive_true(self):
        vec = build_state_vector(_make_row(t0_alive=True))
        assert vec[6] == 1.0
```

- [ ] **Step 2: Run to verify failures**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/test_state_vector.py -v 2>&1 | head -20
```

Expected: `ImportError` or `AssertionError` — `build_state_vector` not yet implemented.

### 1B — Implement

- [ ] **Step 3: Replace `src/features/state_vector.py`**

```python
"""Build fixed-size float32 feature vectors from parquet rows."""

from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_DIM: int = 74
# Layout:
#   [0:35]  T players 0–4: (x, y, z, hp/100, armor/100, helmet, alive) × 5
#   [35:70] CT players 0–4: same layout × 5
#   [70:74] map_zone one-hot: A=70, B=71, mid=72, other=73

_ZONE_IDX: dict[str, int] = {"A": 0, "B": 1, "mid": 2, "other": 3}
_PLAYER_FIELDS: tuple[str, ...] = ("x", "y", "z", "hp", "armor", "helmet", "alive")
_NORMALISE: frozenset[str] = frozenset({"hp", "armor"})  # raw 0–100 → /100


def build_state_vector(row: pd.Series) -> np.ndarray:
    """Convert one parquet row to a float32 feature vector of shape (FEATURE_DIM,).

    hp and armor are normalised by dividing by 100.
    Coordinates and boolean flags are used as-is (already in [0, 1]).
    Missing columns default to 0.0.

    Args:
        row: One row from a parsed demo parquet (pd.Series with column-name index).

    Returns:
        np.ndarray of shape (74,) and dtype float32.
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    for side, base in (("t", 0), ("ct", 35)):
        for i in range(5):
            for j, field in enumerate(_PLAYER_FIELDS):
                col = f"{side}{i}_{field}"
                val = float(row[col]) if col in row.index else 0.0
                if field in _NORMALISE:
                    val /= 100.0
                vec[base + i * 7 + j] = val

    zone_idx = _ZONE_IDX.get(str(row["map_zone"]) if "map_zone" in row.index else "other", 3)
    vec[70 + zone_idx] = 1.0

    return vec
```

- [ ] **Step 4: Run tests**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/test_state_vector.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /d/CSAI/CS_Prophet && git add src/features/state_vector.py tests/test_state_vector.py && git commit -m "feat: state vector builder — 74-dim float32 feature encoding"
```

---

## Task 2: `src/features/dataset.py` (TDD)

**Files:**
- Create: `src/features/dataset.py`
- Create: `tests/test_dataset.py`

### 2A — Write failing tests

- [ ] **Step 1: Create `tests/test_dataset.py`**

```python
"""Tests for dataset — RoundSequenceDataset and split_files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.dataset import LABEL_MAP, RoundSequenceDataset, split_files
from src.features.state_vector import FEATURE_DIM


# ── Fixture helpers ───────────────────────────────────────────────────────

def _make_parquet(
    tmp_path: Path,
    filename: str = "demo.parquet",
    n_rounds: int = 2,
    n_steps: int = 240,
    bomb_site: str = "A",
) -> Path:
    """Write a synthetic parquet with n_rounds rounds of n_steps steps each."""
    rows = []
    for rnd in range(1, n_rounds + 1):
        for step in range(n_steps):
            row: dict = {
                "demo_name": filename.replace(".parquet", ""),
                "round_num": rnd,
                "step": step,
                "tick": step * 8,
                "bomb_site": bomb_site,
                "map_zone": "A",
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


# ── split_files ───────────────────────────────────────────────────────────

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


# ── RoundSequenceDataset ──────────────────────────────────────────────────

class TestRoundSequenceDataset:
    def test_len_equals_round_count(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=3, n_steps=10)
        ds = RoundSequenceDataset([path], sequence_length=240)
        assert len(ds) == 3

    def test_sequence_shape(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, n_steps=120)
        ds = RoundSequenceDataset([path], sequence_length=240)
        seq, label = ds[0]
        assert seq.shape == (240, FEATURE_DIM)
        assert seq.dtype == torch.float32

    def test_short_sequence_padded_with_zeros(self, tmp_path):
        n_steps = 50
        path = _make_parquet(tmp_path, n_rounds=1, n_steps=n_steps)
        ds = RoundSequenceDataset([path], sequence_length=240)
        seq, _ = ds[0]
        # Last step has data, step after should be zero
        assert seq[n_steps - 1].abs().sum().item() > 0
        assert seq[n_steps:].abs().sum().item() == 0.0

    def test_long_sequence_truncated(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, n_steps=300)
        ds = RoundSequenceDataset([path], sequence_length=240)
        seq, _ = ds[0]
        assert seq.shape[0] == 240

    def test_label_A_is_0(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, bomb_site="A")
        ds = RoundSequenceDataset([path], sequence_length=240)
        _, label = ds[0]
        assert label == LABEL_MAP["A"]  # 0

    def test_label_B_is_1(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, bomb_site="B")
        ds = RoundSequenceDataset([path], sequence_length=240)
        _, label = ds[0]
        assert label == LABEL_MAP["B"]  # 1

    def test_label_other_is_2(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, bomb_site="other")
        ds = RoundSequenceDataset([path], sequence_length=240)
        _, label = ds[0]
        assert label == LABEL_MAP["other"]  # 2

    def test_multiple_files_concatenated(self, tmp_path):
        p1 = _make_parquet(tmp_path, "demo1.parquet", n_rounds=2)
        p2 = _make_parquet(tmp_path, "demo2.parquet", n_rounds=3)
        ds = RoundSequenceDataset([p1, p2], sequence_length=240)
        assert len(ds) == 5

    def test_values_in_unit_range(self, tmp_path):
        path = _make_parquet(tmp_path, n_rounds=1, n_steps=10)
        ds = RoundSequenceDataset([path], sequence_length=240)
        seq, _ = ds[0]
        assert seq.min().item() >= 0.0
        assert seq.max().item() <= 1.0
```

- [ ] **Step 2: Run to verify failures**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/test_dataset.py -v 2>&1 | head -20
```

Expected: `ImportError` — `dataset.py` does not exist.

### 2B — Implement

- [ ] **Step 3: Create `src/features/dataset.py`**

```python
"""Dataset classes and file-split utilities for the bomb-site prediction task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.state_vector import FEATURE_DIM, _ZONE_IDX, _NORMALISE, _PLAYER_FIELDS

LABEL_MAP: dict[str, int] = {"A": 0, "B": 1, "other": 2}


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
    - ``label``: int — 0=A, 1=B, 2=other

    Sequences shorter than ``sequence_length`` are zero-padded at the end.
    Sequences longer than ``sequence_length`` are truncated.
    """

    def __init__(self, parquet_files: list[Path], sequence_length: int = 240) -> None:
        self._sequences: list[torch.Tensor] = []
        self._labels: list[int] = []

        for path in parquet_files:
            df = pd.read_parquet(path)
            for (_, _), group in df.groupby(["demo_name", "round_num"], sort=False):
                label = LABEL_MAP.get(str(group["bomb_site"].iloc[0]), 2)
                self._sequences.append(_build_padded_tensor(group, sequence_length))
                self._labels.append(label)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self._sequences[idx], self._labels[idx]


def _build_padded_tensor(group: pd.DataFrame, sequence_length: int) -> torch.Tensor:
    """Build a ``(sequence_length, FEATURE_DIM)`` float32 tensor from one round group.

    Uses vectorised column reads for speed. Missing columns default to 0.
    """
    rows = group.sort_values("step")
    n = min(len(rows), sequence_length)
    mat = np.zeros((sequence_length, FEATURE_DIM), dtype=np.float32)

    for side, base in (("t", 0), ("ct", 35)):
        for i in range(5):
            for j, field in enumerate(_PLAYER_FIELDS):
                col = f"{side}{i}_{field}"
                if col in rows.columns:
                    vals = rows[col].values[:n].astype(np.float32)
                    if field in _NORMALISE:
                        vals = vals / 100.0
                    mat[:n, base + i * 7 + j] = vals

    if "map_zone" in rows.columns:
        zones = rows["map_zone"].values[:n]
        zone_indices = np.array([_ZONE_IDX.get(str(z), 3) for z in zones])
        mat[np.arange(n), 70 + zone_indices] = 1.0

    return torch.from_numpy(mat)
```

- [ ] **Step 4: Run tests**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/test_dataset.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /d/CSAI/CS_Prophet && git add src/features/dataset.py tests/test_dataset.py && git commit -m "feat: RoundSequenceDataset — padded tensor loader with file-level splits"
```

---

## Task 3: `src/model/attention.py` (TDD)

**Files:**
- Modify: `src/model/attention.py`
- Create: `tests/test_attention.py`

### 3A — Write failing tests

- [ ] **Step 1: Create `tests/test_attention.py`**

```python
"""Tests for PositionalEncoding and CrossAttentionLayer."""

import torch
import pytest
from src.model.attention import CrossAttentionLayer, PositionalEncoding


class TestPositionalEncoding:
    def test_output_shape_preserved(self):
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        x = torch.randn(4, 30, 64)
        out = pe(x)
        assert out.shape == x.shape

    def test_no_nan(self):
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        x = torch.zeros(2, 50, 64)
        out = pe(x)
        assert not torch.isnan(out).any()

    def test_adds_different_values_per_position(self):
        pe = PositionalEncoding(d_model=64, dropout=0.0)
        x = torch.zeros(1, 5, 64)
        out = pe(x)
        # Each position gets a different encoding
        for i in range(4):
            assert not torch.allclose(out[0, i], out[0, i + 1])

    def test_long_sequence_within_max_len(self):
        pe = PositionalEncoding(d_model=32, dropout=0.0, max_len=300)
        x = torch.randn(1, 240, 32)
        out = pe(x)
        assert out.shape == (1, 240, 32)


class TestCrossAttentionLayer:
    def test_output_shape_matches_query(self):
        layer = CrossAttentionLayer(d_model=64, nhead=4, dropout=0.0)
        q = torch.randn(2, 30, 64)
        kv = torch.randn(2, 30, 64)
        out = layer(q, kv)
        assert out.shape == q.shape

    def test_no_nan(self):
        layer = CrossAttentionLayer(d_model=64, nhead=4, dropout=0.0)
        q = torch.randn(2, 20, 64)
        kv = torch.randn(2, 20, 64)
        out = layer(q, kv)
        assert not torch.isnan(out).any()

    def test_different_kv_produces_different_output(self):
        torch.manual_seed(0)
        layer = CrossAttentionLayer(d_model=32, nhead=4, dropout=0.0)
        layer.eval()
        q = torch.randn(1, 10, 32)
        kv1 = torch.randn(1, 10, 32)
        kv2 = torch.randn(1, 10, 32)
        out1 = layer(q, kv1)
        out2 = layer(q, kv2)
        assert not torch.allclose(out1, out2)

    def test_batch_size_1(self):
        layer = CrossAttentionLayer(d_model=64, nhead=8, dropout=0.0)
        q = torch.randn(1, 240, 64)
        kv = torch.randn(1, 240, 64)
        out = layer(q, kv)
        assert out.shape == (1, 240, 64)
```

- [ ] **Step 2: Run to verify failures**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/test_attention.py -v 2>&1 | head -20
```

Expected: `ImportError` — `attention.py` is a stub.

### 3B — Implement

- [ ] **Step 3: Replace `src/model/attention.py`**

```python
"""Attention modules for BombSiteTransformer."""

from __future__ import annotations
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017).

    Adds a fixed, position-dependent signal to the input embeddings so
    the model can distinguish earlier from later ticks in the sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 300) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding and apply dropout.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]  # type: ignore[index]
        return self.dropout(x)


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer: query attends to key_value with residual + LayerNorm.

    Used to model T-side (query) attending to CT-side (key/value),
    capturing adversarial spatial interaction at each timestep.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Compute cross-attention with residual connection.

        Args:
            query: (batch, seq_len, d_model) — T-side representation.
            key_value: (batch, seq_len, d_model) — CT-side representation.

        Returns:
            (batch, seq_len, d_model) — query enriched with CT context.
        """
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_out))
```

- [ ] **Step 4: Run tests**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/test_attention.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /d/CSAI/CS_Prophet && git add src/model/attention.py tests/test_attention.py && git commit -m "feat: PositionalEncoding and CrossAttentionLayer"
```

---

## Task 4: `src/model/transformer.py` — Replace with `BombSiteTransformer` (TDD)

**Files:**
- Modify: `src/model/transformer.py`
- Modify: `tests/test_model.py`
- Modify: `configs/train_config.yaml`

### 4A — Update tests first

- [ ] **Step 1: Replace `tests/test_model.py`**

```python
"""Tests for BombSiteTransformer."""

import torch
import pytest
from src.model.transformer import BombSiteTransformer


class TestBombSiteTransformer:
    def _model(self, **kwargs) -> BombSiteTransformer:
        defaults = dict(input_dim=74, d_model=64, nhead=4,
                        num_layers=2, dropout=0.0, num_classes=3)
        defaults.update(kwargs)
        return BombSiteTransformer(**defaults)

    def test_output_shape(self):
        model = self._model()
        x = torch.randn(4, 240, 74)
        out = model(x)
        assert out.shape == (4, 3)

    def test_output_shape_short_sequence(self):
        model = self._model()
        x = torch.randn(2, 10, 74)
        out = model(x)
        assert out.shape == (2, 3)

    def test_no_nan(self):
        model = self._model()
        x = torch.randn(2, 50, 74)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_batch_size_1(self):
        model = self._model()
        x = torch.randn(1, 240, 74)
        out = model(x)
        assert out.shape == (1, 3)

    def test_num_classes_respected(self):
        model = self._model(num_classes=3)
        x = torch.randn(2, 20, 74)
        out = model(x)
        assert out.shape[-1] == 3

    def test_gradients_flow(self):
        model = self._model()
        x = torch.randn(2, 30, 74, requires_grad=False)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_eval_mode_no_dropout_effect(self):
        model = self._model(dropout=0.5)
        model.eval()
        x = torch.randn(1, 20, 74)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)
```

- [ ] **Step 2: Run to verify failures**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/test_model.py -v 2>&1 | head -20
```

Expected: `ImportError` for `BombSiteTransformer` — it doesn't exist yet.

### 4B — Implement

- [ ] **Step 3: Replace `src/model/transformer.py`**

```python
"""BombSiteTransformer — predicts bomb plant site (A / B / other) from round sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.attention import CrossAttentionLayer, PositionalEncoding

# Indices into the 74-dim feature vector (see state_vector.py):
#   [0:35]   T-player features (5 × 7)
#   [35:70]  CT-player features (5 × 7)
#   [70:74]  map_zone one-hot
_T_SLICE = (slice(None), slice(None), slice(0, 35))       # T-player features
_ZONE_SLICE = (slice(None), slice(None), slice(70, 74))   # zone one-hot
_CT_SLICE = (slice(None), slice(None), slice(35, 70))     # CT-player features

_T_IN = 39   # 35 T-player features + 4 zone features → projected to d_model
_CT_IN = 35  # 35 CT-player features → projected to d_model


class BombSiteTransformer(nn.Module):
    """Sequence-to-label Transformer for bomb-site prediction.

    Architecture:
        1. Split input into T-side (player + zone, 39-dim) and CT-side (35-dim)
        2. Project each to d_model with learned linear layers
        3. Add sinusoidal positional encoding to both
        4. Cross-attention: T queries CT to model adversarial interaction
        5. Self-attention encoder stack on T-side representation
        6. Linear classifier on last-token output → (num_classes,) logits
    """

    def __init__(
        self,
        input_dim: int = 74,   # kept for config compatibility; actual split is fixed
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.t_proj = nn.Linear(_T_IN, d_model)
        self.ct_proj = nn.Linear(_CT_IN, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict bomb plant site from a batch of round sequences.

        Args:
            x: float32 of shape (batch, seq_len, 74).

        Returns:
            (batch, num_classes) logits — pass through softmax for probabilities.
        """
        # Split feature vector
        t_feats = torch.cat([x[_T_SLICE], x[_ZONE_SLICE]], dim=-1)  # (B, T, 39)
        ct_feats = x[_CT_SLICE]                                       # (B, T, 35)

        # Project + positional encoding
        t_emb = self.pos_enc(self.t_proj(t_feats))
        ct_emb = self.pos_enc(self.ct_proj(ct_feats))

        # Cross-attention: T side enriched with CT context
        t_emb = self.cross_attn(t_emb, ct_emb)

        # Self-attention encoder
        out = self.encoder(t_emb)

        # Classify from last timestep
        return self.classifier(out[:, -1, :])
```

- [ ] **Step 4: Update `configs/train_config.yaml` — fix `input_dim`**

Read the file, then change `input_dim: 84` to `input_dim: 74`:

```yaml
model:
  input_dim: 74        # 10 players × 7 features = 70 + map_zone_onehot(4)
  d_model: 256
  nhead: 8
  num_layers: 4
  dropout: 0.1
  num_classes: 3       # A / B / other
```

(Leave all other fields unchanged.)

- [ ] **Step 5: Run tests**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/test_model.py tests/test_attention.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 6: Run full suite to check no regressions**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/ -v
```

Expected: all tests PASS (52 from Phase 1 + 12 new model/attention tests).

- [ ] **Step 7: Commit**

```bash
cd /d/CSAI/CS_Prophet && git add src/model/transformer.py tests/test_model.py configs/train_config.yaml && git commit -m "feat: BombSiteTransformer — 3-class model with T×CT cross-attention"
```

---

## Task 5: `src/model/train.py` — FocalLoss + training loop (TDD)

**Files:**
- Modify: `src/model/train.py`
- Create: `tests/test_train.py`

### 5A — Write failing tests

- [ ] **Step 1: Create `tests/test_train.py`**

```python
"""Tests for FocalLoss and evaluate."""

import torch
import torch.nn.functional as F
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.model.train import FocalLoss, evaluate
from src.model.transformer import BombSiteTransformer


def _tiny_model() -> BombSiteTransformer:
    return BombSiteTransformer(
        input_dim=74, d_model=32, nhead=4, num_layers=1, dropout=0.0, num_classes=3
    )


def _tiny_loader(n: int = 8, seq_len: int = 10) -> DataLoader:
    x = torch.randn(n, seq_len, 74)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


# ── FocalLoss ─────────────────────────────────────────────────────────────

class TestFocalLoss:
    def test_returns_scalar(self):
        criterion = FocalLoss(gamma=2.0)
        loss = criterion(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))
        assert loss.shape == ()

    def test_positive_loss(self):
        criterion = FocalLoss(gamma=2.0)
        loss = criterion(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))
        assert loss.item() > 0

    def test_no_nan(self):
        criterion = FocalLoss(gamma=2.0)
        loss = criterion(torch.randn(8, 3), torch.randint(0, 3, (8,)))
        assert not torch.isnan(loss)

    def test_gamma_0_equals_cross_entropy(self):
        """With γ=0, focal loss reduces to standard cross-entropy."""
        torch.manual_seed(42)
        criterion = FocalLoss(gamma=0.0)
        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))
        focal = criterion(logits, targets)
        ce = F.cross_entropy(logits, targets)
        assert torch.allclose(focal, ce, atol=1e-5)

    def test_gamma_2_leq_gamma_0_for_easy_examples(self):
        """Higher γ down-weights easy (high-confidence correct) examples."""
        # Confident correct predictions: high logit for true class
        logits = torch.zeros(4, 3)
        logits[0, 0] = 10.0; logits[1, 1] = 10.0
        logits[2, 2] = 10.0; logits[3, 0] = 10.0
        targets = torch.tensor([0, 1, 2, 0])
        loss_g0 = FocalLoss(gamma=0.0)(logits, targets)
        loss_g2 = FocalLoss(gamma=2.0)(logits, targets)
        assert loss_g2.item() < loss_g0.item()

    def test_differentiable(self):
        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3, requires_grad=True)
        loss = criterion(logits, torch.tensor([0, 1, 2, 1]))
        loss.backward()
        assert logits.grad is not None


# ── evaluate ──────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_float_tuple(self):
        model = _tiny_model()
        criterion = FocalLoss(gamma=2.0)
        loss, acc = evaluate(model, _tiny_loader(), criterion,
                             torch.device("cpu"), use_amp=False)
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_accuracy_in_unit_range(self):
        model = _tiny_model()
        criterion = FocalLoss(gamma=2.0)
        _, acc = evaluate(model, _tiny_loader(), criterion,
                          torch.device("cpu"), use_amp=False)
        assert 0.0 <= acc <= 1.0

    def test_loss_positive(self):
        model = _tiny_model()
        criterion = FocalLoss(gamma=2.0)
        loss, _ = evaluate(model, _tiny_loader(), criterion,
                           torch.device("cpu"), use_amp=False)
        assert loss > 0.0

    def test_perfect_model_accuracy_1(self):
        """A model that always outputs the correct class should get 1.0 accuracy."""
        class PerfectModel(torch.nn.Module):
            def forward(self, x):
                # Always predict class 0 for a batch of all-0 labels
                return torch.tensor([[10.0, 0.0, 0.0]]).expand(x.size(0), -1)

        x = torch.randn(8, 10, 74)
        y = torch.zeros(8, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        criterion = FocalLoss(gamma=2.0)
        _, acc = evaluate(PerfectModel(), loader, criterion,
                          torch.device("cpu"), use_amp=False)
        assert acc == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify failures**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/test_train.py -v 2>&1 | head -20
```

Expected: `ImportError` — `train.py` is a stub.

### 5B — Implement

- [ ] **Step 3: Replace `src/model/train.py`**

```python
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
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long)
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
        test_frac=cfg.training.val_split,
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
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
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    accum_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        y = y.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(x), y) / accum_steps

        scaler.scale(loss).backward()

        if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps * len(y)

    return total_loss / max(len(loader.dataset), 1)
```

- [ ] **Step 4: Run tests**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/test_train.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
cd /d/CSAI/CS_Prophet && python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /d/CSAI/CS_Prophet && git add src/model/train.py tests/test_train.py && git commit -m "feat: Phase 2 training — FocalLoss, AMP, gradient accumulation, evaluate()"
```

---

## Self-Review

### Spec Coverage

| Spec requirement | Task |
|---|---|
| state_vector.py — 74-dim feature vector | Task 1 |
| HP / armor normalised to [0,1] | Task 1 |
| Map zone one-hot encoding | Task 1 |
| Dataset — pad/truncate to 240 steps | Task 2 |
| File-level train/val/test split | Task 2 |
| Label encoding A=0, B=1, other=2 | Task 2 |
| PositionalEncoding (sinusoidal) | Task 3 |
| CrossAttentionLayer (T Query × CT KV) | Task 3 |
| BombSiteTransformer — 3-class output | Task 4 |
| input_dim config updated to 74 | Task 4 |
| Focal Loss (γ=2, reduces to CE at γ=0) | Task 5 |
| Mixed-precision training (AMP) | Task 5 |
| Gradient accumulation (steps=4) | Task 5 |
| Best-checkpoint saving | Task 5 |
| train() function wired to config YAML | Task 5 |

### Notes for Phase 3
- `src/inference/onnx_export.py` — export `BombSiteTransformer` to ONNX after Phase 2 training completes
- `src/inference/realtime_engine.py` — sliding-window GSI consumer
- Player / team embedding layers (spec item) deferred: requires player/team ID columns not yet in the parquet; add to parser in a future phase
