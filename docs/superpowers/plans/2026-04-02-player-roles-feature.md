# Player Role Feature (74→124 dim) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 5-dim role one-hot per player to the feature vector, expanding from 74 to 124 dims.

**Architecture:** Each player gains 5 new dims at offsets [7:12] within their 12-dim block (IGL/AWPer/Entry fragger/Support/Lurker; all-zero = unknown). Parser writes `t{i}_role` / `ct{i}_role` string columns to parquet; state_vector and dataset read them. Transformer slices updated to match new layout.

**Tech Stack:** numpy, pandas, torch, pytest

---

## New Layout (124 dims)

```
[0:60]    T players 0–4: (x, y, z, hp/100, armor/100, helmet, alive, r0..r4) × 5
[60:120]  CT players 0–4: same × 5
[120:124] map_zone one-hot: A=120, B=121, mid=122, other=123
```

Per-player block (12 dims):
- `[0]` x, `[1]` y, `[2]` z, `[3]` hp/100, `[4]` armor/100, `[5]` helmet, `[6]` alive
- `[7]` IGL, `[8]` AWPer, `[9]` Entry fragger, `[10]` Support, `[11]` Lurker

---

## File Map

| File | Action |
|------|--------|
| `src/features/state_vector.py` | Add PLAYER_DIM, ROLE_IDX, CT_BASE, ZONE_BASE; FEATURE_DIM 74→124; update build_state_vector |
| `src/features/dataset.py` | Import new constants; update player stride 7→PLAYER_DIM, base 35→CT_BASE, zone offset 70→ZONE_BASE; add role encoding |
| `src/parser/demo_parser.py` | Add optional `player_roles` param to parse_demo/_extract_sequence/_build_state_row; write role columns |
| `src/model/transformer.py` | Update _T_SLICE/_CT_SLICE/_ZONE_SLICE, _T_IN 39→64, _CT_IN 35→60, input_dim guard 74→124 |
| `configs/train_config.yaml` | input_dim: 74 → 124 |
| `src/inference/predictor.py` | Update docstring (seq_len, 74) → (seq_len, 124) |
| `tests/test_state_vector.py` | Update FEATURE_DIM, shape, offset checks; add role tests |
| `tests/test_dataset.py` | Add role columns to _make_parquet fixture |
| `tests/test_parser.py` | Add player_roles param to _make_tick_df / _build_state_row tests; add role column assertions |
| `tests/test_model.py` | input_dim=74→124 everywhere, randn(...,74)→(...,124) |

---

## Task 1: Update state_vector.py (74→124, add role encoding)

**Files:**
- Modify: `src/features/state_vector.py`
- Test: `tests/test_state_vector.py`

- [ ] **Step 1: Update tests to expect new constants and layout**

Replace the entire content of `tests/test_state_vector.py`:

```python
"""Tests for state_vector — single-row feature vector builder."""

import numpy as np
import pandas as pd
import pytest
from src.features.state_vector import (
    CT_BASE, FEATURE_DIM, PLAYER_DIM, ROLE_IDX, ZONE_BASE,
    build_state_vector,
)


def _make_row(map_zone: str = "A", **overrides) -> pd.Series:
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
            data[f"{side}{i}_role"] = ""  # unknown → all-zero role
    data.update(overrides)
    return pd.Series(data)


class TestConstants:
    def test_feature_dim_is_124(self):
        assert FEATURE_DIM == 124

    def test_player_dim_is_12(self):
        assert PLAYER_DIM == 12

    def test_ct_base_is_60(self):
        assert CT_BASE == 60

    def test_zone_base_is_120(self):
        assert ZONE_BASE == 120

    def test_role_idx_has_5_roles(self):
        assert len(ROLE_IDX) == 5

    def test_role_idx_values(self):
        assert ROLE_IDX["IGL"] == 0
        assert ROLE_IDX["AWPer"] == 1
        assert ROLE_IDX["Entry fragger"] == 2
        assert ROLE_IDX["Support"] == 3
        assert ROLE_IDX["Lurker"] == 4


class TestBuildStateVector:
    def test_output_shape(self):
        vec = build_state_vector(_make_row())
        assert vec.shape == (124,)

    def test_dtype_is_float32(self):
        vec = build_state_vector(_make_row())
        assert vec.dtype == np.float32

    def test_hp_normalised_to_1(self):
        vec = build_state_vector(_make_row(t0_hp=100))
        assert vec[3] == pytest.approx(1.0)

    def test_hp_normalised_to_half(self):
        vec = build_state_vector(_make_row(t0_hp=50))
        assert vec[3] == pytest.approx(0.5)

    def test_armor_normalised(self):
        vec = build_state_vector(_make_row(t0_armor=60))
        assert vec[4] == pytest.approx(0.6)

    def test_ct_player_offset(self):
        # CT base=60, player 0, hp field=3 → index 63
        vec = build_state_vector(_make_row(ct0_hp=80))
        assert vec[63] == pytest.approx(0.8)

    def test_zone_one_hot_A(self):
        vec = build_state_vector(_make_row(map_zone="A"))
        assert vec[120] == 1.0
        assert vec[121] == 0.0
        assert vec[122] == 0.0
        assert vec[123] == 0.0

    def test_zone_one_hot_B(self):
        vec = build_state_vector(_make_row(map_zone="B"))
        assert vec[120] == 0.0
        assert vec[121] == 1.0

    def test_zone_one_hot_mid(self):
        vec = build_state_vector(_make_row(map_zone="mid"))
        assert vec[122] == 1.0

    def test_zone_one_hot_other(self):
        vec = build_state_vector(_make_row(map_zone="other"))
        assert vec[123] == 1.0

    def test_all_values_in_unit_range(self):
        vec = build_state_vector(_make_row())
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_dead_player_alive_false(self):
        vec = build_state_vector(_make_row(t0_alive=False))
        assert vec[6] == 0.0

    def test_alive_player_alive_true(self):
        vec = build_state_vector(_make_row(t0_alive=True))
        assert vec[6] == 1.0


class TestRoleEncoding:
    def test_unknown_role_all_zeros(self):
        vec = build_state_vector(_make_row(t0_role=""))
        assert np.all(vec[7:12] == 0.0)

    def test_rifler_role_all_zeros(self):
        vec = build_state_vector(_make_row(t0_role="Rifler"))
        assert np.all(vec[7:12] == 0.0)

    def test_missing_role_column_all_zeros(self):
        # Row without any role columns
        data: dict = {"map_zone": "A"}
        for side in ("t", "ct"):
            for i in range(5):
                data[f"{side}{i}_x"] = 0.5
                data[f"{side}{i}_y"] = 0.5
                data[f"{side}{i}_z"] = 0.5
                data[f"{side}{i}_hp"] = 100
                data[f"{side}{i}_armor"] = 100
                data[f"{side}{i}_helmet"] = True
                data[f"{side}{i}_alive"] = True
        vec = build_state_vector(pd.Series(data))
        # All role bits should be 0
        for i in range(5):
            assert np.all(vec[i * 12 + 7 : i * 12 + 12] == 0.0)

    def test_t0_role_IGL(self):
        # T player 0 IGL → offset 0*12+7+0 = 7
        vec = build_state_vector(_make_row(t0_role="IGL"))
        assert vec[7] == 1.0
        assert np.all(vec[8:12] == 0.0)

    def test_t0_role_AWPer(self):
        # T player 0 AWPer → offset 0*12+7+1 = 8
        vec = build_state_vector(_make_row(t0_role="AWPer"))
        assert vec[8] == 1.0
        assert vec[7] == 0.0

    def test_t0_role_entry_fragger(self):
        vec = build_state_vector(_make_row(t0_role="Entry fragger"))
        assert vec[9] == 1.0

    def test_t0_role_support(self):
        vec = build_state_vector(_make_row(t0_role="Support"))
        assert vec[10] == 1.0

    def test_t0_role_lurker(self):
        vec = build_state_vector(_make_row(t0_role="Lurker"))
        assert vec[11] == 1.0

    def test_t2_role_offset(self):
        # T player 2 IGL → offset 2*12+7 = 31
        vec = build_state_vector(_make_row(t2_role="IGL"))
        assert vec[31] == 1.0

    def test_ct0_role_offset(self):
        # CT player 0 IGL → offset CT_BASE + 0*12 + 7 = 60+7 = 67
        vec = build_state_vector(_make_row(ct0_role="IGL"))
        assert vec[67] == 1.0

    def test_ct4_role_offset(self):
        # CT player 4 Lurker → CT_BASE + 4*12 + 7 + 4 = 60+48+11 = 119
        vec = build_state_vector(_make_row(ct4_role="Lurker"))
        assert vec[119] == 1.0

    def test_role_one_hot_only_one_bit_set(self):
        for role in ("IGL", "AWPer", "Entry fragger", "Support", "Lurker"):
            vec = build_state_vector(_make_row(t0_role=role))
            assert vec[7:12].sum() == pytest.approx(1.0), f"role={role}"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_state_vector.py -v
```

Expected: failures on `test_feature_dim_is_124`, shape test, offset tests, all role tests.

- [ ] **Step 3: Implement new state_vector.py**

Replace the entire content of `src/features/state_vector.py`:

```python
"""Build fixed-size float32 feature vectors from parquet rows."""

from __future__ import annotations
import numpy as np
import pandas as pd

PLAYER_DIM: int = 12          # 7 core fields + 5 role one-hot
CT_BASE: int = PLAYER_DIM * 5      # 60
ZONE_BASE: int = PLAYER_DIM * 5 * 2  # 120
FEATURE_DIM: int = ZONE_BASE + 4     # 124
# Layout:
#   [0:60]    T players 0–4: (x, y, z, hp/100, armor/100, helmet, alive, r0..r4) × 5
#   [60:120]  CT players 0–4: same layout × 5
#   [120:124] map_zone one-hot: A=120, B=121, mid=122, other=123

ZONE_IDX: dict[str, int] = {"A": 0, "B": 1, "mid": 2, "other": 3}
ROLE_NAMES: tuple[str, ...] = ("IGL", "AWPer", "Entry fragger", "Support", "Lurker")
ROLE_IDX: dict[str, int] = {name: i for i, name in enumerate(ROLE_NAMES)}
PLAYER_FIELDS: tuple[str, ...] = ("x", "y", "z", "hp", "armor", "helmet", "alive")
NORMALISE: frozenset[str] = frozenset({"hp", "armor"})


def build_state_vector(row: pd.Series) -> np.ndarray:
    """Convert one parquet row to a float32 feature vector of shape (FEATURE_DIM,).

    hp and armor are normalised by dividing by 100.
    Coordinates and boolean flags are used as-is (already in [0, 1]).
    Missing columns default to 0.0. Unknown or missing roles are all-zero.

    Args:
        row: One row from a parsed demo parquet (pd.Series with column-name index).

    Returns:
        np.ndarray of shape (124,) and dtype float32.
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    for side, base in (("t", 0), ("ct", CT_BASE)):
        for i in range(5):
            offset = base + i * PLAYER_DIM
            for j, field in enumerate(PLAYER_FIELDS):
                col = f"{side}{i}_{field}"
                val = float(row[col]) if col in row.index else 0.0
                if field in NORMALISE:
                    val /= 100.0
                vec[offset + j] = val
            # role one-hot at offsets [7:12] within this player block
            role_col = f"{side}{i}_role"
            role_str = str(row[role_col]) if role_col in row.index else ""
            role_i = ROLE_IDX.get(role_str, -1)
            if role_i >= 0:
                vec[offset + 7 + role_i] = 1.0

    zone_idx = ZONE_IDX.get(str(row["map_zone"]) if "map_zone" in row.index else "other", 3)
    vec[ZONE_BASE + zone_idx] = 1.0

    return vec
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_state_vector.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/features/state_vector.py tests/test_state_vector.py
git commit -m "feat: expand feature vector 74→124 dims with 5-dim player role one-hot"
```

---

## Task 2: Update dataset.py (use new constants, add role encoding)

**Files:**
- Modify: `src/features/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Update _make_parquet fixture in test_dataset.py to include role columns**

In `tests/test_dataset.py`, update the inner loop of `_make_parquet` to add role columns. Find the block:

```python
            for side in ("t", "ct"):
                for i in range(5):
                    row[f"{side}{i}_x"] = 0.5
                    row[f"{side}{i}_y"] = 0.5
                    row[f"{side}{i}_z"] = 0.5
                    row[f"{side}{i}_hp"] = 100
                    row[f"{side}{i}_armor"] = 100
                    row[f"{side}{i}_helmet"] = True
                    row[f"{side}{i}_alive"] = True
```

Replace with:

```python
            for side in ("t", "ct"):
                for i in range(5):
                    row[f"{side}{i}_x"] = 0.5
                    row[f"{side}{i}_y"] = 0.5
                    row[f"{side}{i}_z"] = 0.5
                    row[f"{side}{i}_hp"] = 100
                    row[f"{side}{i}_armor"] = 100
                    row[f"{side}{i}_helmet"] = True
                    row[f"{side}{i}_alive"] = True
                    row[f"{side}{i}_role"] = ""
```

- [ ] **Step 2: Run dataset tests to confirm they still pass (FEATURE_DIM is imported)**

```
pytest tests/test_dataset.py -v
```

Expected: all pass (dataset imports FEATURE_DIM from state_vector which is now 124, so `seq.shape == (240, FEATURE_DIM)` is `(240, 124)` — the test checks shape against FEATURE_DIM so it auto-adjusts).

If any test fails, investigate before continuing.

- [ ] **Step 3: Update dataset.py to use new constants and encode roles**

Replace the entire content of `src/features/dataset.py`:

```python
"""Dataset classes and file-split utilities for the bomb-site prediction task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.state_vector import (
    CT_BASE, FEATURE_DIM, NORMALISE, PLAYER_DIM, PLAYER_FIELDS,
    ROLE_IDX, ZONE_BASE, ZONE_IDX,
)

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

    def __init__(self, parquet_files: list[Path], sequence_length: int = 240) -> None:
        self._sequences: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

        for path in parquet_files:
            df = pd.read_parquet(path)
            for (_, _), group in df.groupby(["demo_name", "round_num"], sort=False):
                label = LABEL_MAP.get(str(group["bomb_site"].iloc[0]), 2)
                self._sequences.append(_build_padded_tensor(group, sequence_length))
                self._labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sequences[idx], self._labels[idx]


def _build_padded_tensor(group: pd.DataFrame, sequence_length: int) -> torch.Tensor:
    """Build a ``(sequence_length, FEATURE_DIM)`` float32 tensor from one round group."""
    rows = group.sort_values("step")
    n = min(len(rows), sequence_length)
    mat = np.zeros((sequence_length, FEATURE_DIM), dtype=np.float32)

    for side, base in (("t", 0), ("ct", CT_BASE)):
        for i in range(5):
            for j, field in enumerate(PLAYER_FIELDS):
                col = f"{side}{i}_{field}"
                if col in rows.columns:
                    vals = rows[col].values[:n].astype(np.float32)
                    if field in NORMALISE:
                        vals = vals / 100.0
                    mat[:n, base + i * PLAYER_DIM + j] = vals
            # role one-hot — role is constant for a player within a sequence
            role_col = f"{side}{i}_role"
            if role_col in rows.columns and n > 0:
                role_str = str(rows[role_col].iloc[0])
                ridx = ROLE_IDX.get(role_str, -1)
                if ridx >= 0:
                    mat[:n, base + i * PLAYER_DIM + 7 + ridx] = 1.0

    if "map_zone" in rows.columns:
        zones = rows["map_zone"].values[:n]
        zone_indices = np.array([ZONE_IDX.get(str(z), 3) for z in zones])
        mat[np.arange(n), ZONE_BASE + zone_indices] = 1.0

    return torch.from_numpy(mat)
```

- [ ] **Step 4: Run all dataset tests**

```
pytest tests/test_dataset.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/features/dataset.py tests/test_dataset.py
git commit -m "feat: update dataset tensor builder for 124-dim layout with role encoding"
```

---

## Task 3: Update demo_parser.py (add player_roles parameter)

**Files:**
- Modify: `src/parser/demo_parser.py`
- Test: `tests/test_parser.py`

- [ ] **Step 1: Add role tests to test_parser.py**

In `tests/test_parser.py`, add a new `TestBuildStateRowRoles` class at the end of the file:

```python
class TestBuildStateRowRoles:
    def test_role_columns_present_without_roles(self):
        """Role columns exist and are empty string when no player_roles given."""
        row = _build_state_row(
            _make_tick_df(100), step=0, tick=100,
            round_num=1, bomb_site="A", map_name="de_mirage",
        )
        for side in ("t", "ct"):
            for i in range(5):
                assert f"{side}{i}_role" in row
                assert row[f"{side}{i}_role"] == ""

    def test_role_stored_when_provided(self):
        """Known player name maps to its role."""
        roles = {"t_p0": "IGL", "ct_p0": "AWPer"}
        row = _build_state_row(
            _make_tick_df(100), step=0, tick=100,
            round_num=1, bomb_site="A", map_name="de_mirage",
            player_roles=roles,
        )
        assert row["t0_role"] == "IGL"
        assert row["ct0_role"] == "AWPer"

    def test_unknown_player_role_is_empty(self):
        """Player not in roles dict gets empty string."""
        roles = {"unknown_player": "IGL"}
        row = _build_state_row(
            _make_tick_df(100), step=0, tick=100,
            round_num=1, bomb_site="A", map_name="de_mirage",
            player_roles=roles,
        )
        assert row["t0_role"] == ""

    def test_padded_slot_role_is_empty(self):
        """Zero-padded player slots have empty role."""
        slim = _make_tick_df(100, n_t=2, n_ct=5)
        row = _build_state_row(
            slim, step=0, tick=100,
            round_num=1, bomb_site="A", map_name="de_mirage",
            player_roles={"t_p0": "IGL"},
        )
        assert row["t2_role"] == ""
        assert row["t3_role"] == ""
        assert row["t4_role"] == ""

    def test_parse_demo_writes_role_columns(self, tmp_path: Path):
        """End-to-end: role columns appear in the written parquet."""
        from unittest.mock import patch
        player_roles = {f"t_p{i}": "IGL" for i in range(5)}
        player_roles.update({f"ct_p{i}": "AWPer" for i in range(5)})
        mock = _make_mock_parser()
        dem = tmp_path / "roles_test.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out", player_roles=player_roles)

        df = pd.read_parquet(out)
        for side in ("t", "ct"):
            for i in range(5):
                assert f"{side}{i}_role" in df.columns
        assert (df["t0_role"] == "IGL").all()
        assert (df["ct0_role"] == "AWPer").all()
```

Also update the existing `test_all_player_keys_present` test in `TestBuildStateRow` to include `_role`:

```python
    def test_all_player_keys_present(self):
        row = self._row()
        for side in ("t", "ct"):
            for i in range(5):
                for suffix in ("_x", "_y", "_z", "_hp", "_armor", "_helmet", "_alive", "_role"):
                    assert f"{side}{i}{suffix}" in row
```

- [ ] **Step 2: Run parser tests to confirm new tests fail**

```
pytest tests/test_parser.py::TestBuildStateRowRoles -v
pytest tests/test_parser.py::TestBuildStateRow::test_all_player_keys_present -v
```

Expected: both fail (no `_role` key, no `player_roles` param).

- [ ] **Step 3: Update demo_parser.py — add player_roles param**

In `src/parser/demo_parser.py`, update `parse_demo` signature and call:

```python
def parse_demo(
    dem_path: Path | str,
    output_dir: Path | str,
    player_roles: dict[str, str] | None = None,
) -> Optional[Path]:
    """Parse one CS2 demo and write a per-round state-sequence parquet.

    Only rounds with a confirmed bomb plant ('A' or 'B') are included.

    Args:
        dem_path: Path to the .dem file.
        output_dir: Directory for the output parquet.
        player_roles: Optional mapping of player name → role string
            (e.g. {"s1mple": "AWPer"}). Written as t{i}_role / ct{i}_role
            columns in the parquet. Missing names get empty string.

    Returns:
        Path to the written parquet, or None if no valid rounds found.
    """
```

In the same function, update the `_extract_sequence` call to pass `player_roles`:

```python
        seq = _extract_sequence(parser, round_num, int(plant_tick), bomb_site, map_name, player_roles)
```

Update `_extract_sequence` signature and `_build_state_row` call:

```python
def _extract_sequence(
    parser,  # noqa: ANN001
    round_num: int,
    plant_tick: int,
    bomb_site: str,
    map_name: str,
    player_roles: dict[str, str] | None = None,
) -> Optional[pd.DataFrame]:
    """Build a downsampled state DataFrame for the 30 s window before a plant.

    Args:
        parser: demoparser2.DemoParser instance.
        round_num: 1-based round index (stored in output).
        plant_tick: Tick of the bomb_planted event.
        bomb_site: 'A' or 'B'.
        map_name: e.g. 'de_mirage'.
        player_roles: Optional mapping of player name → role string.

    Returns:
        DataFrame (one row per step) or None on failure.
    """
```

In `_extract_sequence`, update the `_build_state_row` call:

```python
        rows.append(
            _build_state_row(tick_slice, step, tick, round_num, bomb_site, map_name, player_roles)
        )
```

Update `_build_state_row` signature and add role logic:

```python
def _build_state_row(
    tick_slice: pd.DataFrame,
    step: int,
    tick: int,
    round_num: int,
    bomb_site: str,
    map_name: str,
    player_roles: dict[str, str] | None = None,
) -> dict:
    """Flatten one tick's player data into a single state dict.

    Players on each side are sorted by name for reproducibility.
    Missing players (< 5 per side) are zero-padded.

    Returns:
        Flat dict with keys round_num, step, tick, bomb_site, map_zone,
        and t{i}/ct{i} suffixed position/status/role columns.
    """
```

In the player loop inside `_build_state_row`, after writing `alive`, add role:

```python
                row[f"{prefix}_x"] = x_n
                row[f"{prefix}_y"] = y_n
                row[f"{prefix}_z"] = z_n
                row[f"{prefix}_hp"] = int(p.get("health", 0))
                row[f"{prefix}_armor"] = int(p.get("armor_value", 0))
                row[f"{prefix}_helmet"] = bool(p.get("has_helmet", False))
                row[f"{prefix}_alive"] = bool(p.get("is_alive", False))
                row[f"{prefix}_role"] = (
                    player_roles.get(str(p.get("name", "")), "")
                    if player_roles else ""
                )
```

In the zero-pad branch, add role:

```python
                row[f"{prefix}_x"] = 0.0
                row[f"{prefix}_y"] = 0.0
                row[f"{prefix}_z"] = 0.0
                row[f"{prefix}_hp"] = 0
                row[f"{prefix}_armor"] = 0
                row[f"{prefix}_helmet"] = False
                row[f"{prefix}_alive"] = False
                row[f"{prefix}_role"] = ""
```

- [ ] **Step 4: Run all parser tests**

```
pytest tests/test_parser.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/parser/demo_parser.py tests/test_parser.py
git commit -m "feat: add optional player_roles param to demo parser, write role columns to parquet"
```

---

## Task 4: Update transformer.py, config, and remaining tests

**Files:**
- Modify: `src/model/transformer.py`
- Modify: `configs/train_config.yaml`
- Modify: `src/inference/predictor.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Update test_model.py (input_dim and tensor dims)**

Replace the entire content of `tests/test_model.py`:

```python
"""Tests for BombSiteTransformer."""

import torch
import pytest
from src.model.transformer import BombSiteTransformer


class TestBombSiteTransformer:
    def _model(self, **kwargs) -> BombSiteTransformer:
        defaults = dict(input_dim=124, d_model=64, nhead=4,
                        num_layers=2, dropout=0.0, num_classes=3)
        defaults.update(kwargs)
        return BombSiteTransformer(**defaults)

    def test_output_shape(self):
        model = self._model()
        x = torch.randn(4, 240, 124)
        out = model(x)
        assert out.shape == (4, 3)

    def test_output_shape_short_sequence(self):
        model = self._model()
        x = torch.randn(2, 10, 124)
        out = model(x)
        assert out.shape == (2, 3)

    def test_no_nan(self):
        model = self._model()
        x = torch.randn(2, 50, 124)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_batch_size_1(self):
        model = self._model()
        x = torch.randn(1, 240, 124)
        out = model(x)
        assert out.shape == (1, 3)

    def test_num_classes_respected(self):
        model = self._model(num_classes=5)
        x = torch.randn(2, 20, 124)
        out = model(x)
        assert out.shape[-1] == 5

    def test_gradients_flow(self):
        model = self._model()
        x = torch.randn(2, 30, 124, requires_grad=False)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_eval_mode_no_dropout_effect(self):
        model = self._model(dropout=0.5)
        model.eval()
        x = torch.randn(1, 20, 124)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_padding_mask_accepted(self):
        model = self._model()
        x = torch.randn(2, 30, 124)
        mask = torch.zeros(2, 30, dtype=torch.bool)
        out = model(x, src_key_padding_mask=mask)
        assert out.shape == (2, 3)
```

- [ ] **Step 2: Run model tests to confirm they fail**

```
pytest tests/test_model.py -v
```

Expected: failures due to `input_dim != 74` guard and wrong tensor dims.

- [ ] **Step 3: Update transformer.py (slices, dims, guard)**

Replace the entire content of `src/model/transformer.py`:

```python
"""BombSiteTransformer — predicts bomb plant site (A / B / other) from round sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.attention import CrossAttentionLayer, PositionalEncoding

# Indices into the 124-dim feature vector (see state_vector.py):
#   [0:60]    T-player features (5 × 12)
#   [60:120]  CT-player features (5 × 12)
#   [120:124] map_zone one-hot
_T_SLICE = (slice(None), slice(None), slice(0, 60))        # T-player features
_ZONE_SLICE = (slice(None), slice(None), slice(120, 124))  # zone one-hot
_CT_SLICE = (slice(None), slice(None), slice(60, 120))     # CT-player features

_T_IN = 64   # 60 T-player features + 4 zone features → projected to d_model
_CT_IN = 60  # 60 CT-player features → projected to d_model


class BombSiteTransformer(nn.Module):
    """Sequence-to-label Transformer for bomb-site prediction.

    Architecture:
        1. Split input into T-side (player + zone, 64-dim) and CT-side (60-dim)
        2. Project each to d_model with learned linear layers
        3. Add sinusoidal positional encoding to both
        4. Cross-attention: T queries CT to model adversarial interaction
        5. Self-attention encoder stack on T-side representation
        6. Linear classifier on last-token output → (num_classes,) logits
    """

    def __init__(
        self,
        input_dim: int = 124,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if input_dim != 124:
            raise ValueError(
                f"BombSiteTransformer requires input_dim=124 (60 T + 60 CT + 4 zone); got {input_dim}"
            )
        self.t_proj = nn.Linear(_T_IN, d_model)
        self.ct_proj = nn.Linear(_CT_IN, d_model)
        self.t_pos_enc = PositionalEncoding(d_model, dropout)
        self.ct_pos_enc = PositionalEncoding(d_model, dropout)
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict bomb plant site from a batch of round sequences.

        Args:
            x: float32 of shape (batch, seq_len, 124).
            src_key_padding_mask: optional bool tensor of shape (batch, seq_len),
                True at positions that should be ignored (padding). Forwarded to
                cross-attention and the self-attention encoder.

        Returns:
            (batch, num_classes) logits — pass through softmax for probabilities.
        """
        # Split feature vector
        t_feats = torch.cat([x[_T_SLICE], x[_ZONE_SLICE]], dim=-1)  # (B, T, 64)
        ct_feats = x[_CT_SLICE]                                       # (B, T, 60)

        # Project + positional encoding
        t_emb = self.t_pos_enc(self.t_proj(t_feats))
        ct_emb = self.ct_pos_enc(self.ct_proj(ct_feats))

        # Cross-attention: T side enriched with CT context
        t_emb = self.cross_attn(t_emb, ct_emb, key_padding_mask=src_key_padding_mask)

        # Self-attention encoder
        out = self.encoder(t_emb, src_key_padding_mask=src_key_padding_mask)

        # Classify from last real (non-padded) timestep
        if src_key_padding_mask is not None:
            seq_lens = (~src_key_padding_mask).sum(dim=1) - 1  # (B,) last valid index
            last_real = out[torch.arange(out.size(0), device=out.device), seq_lens, :]
        else:
            last_real = out[:, -1, :]
        return self.classifier(last_real)
```

- [ ] **Step 4: Run model tests**

```
pytest tests/test_model.py -v
```

Expected: all pass.

- [ ] **Step 5: Update configs/train_config.yaml**

Change line 2 from `input_dim: 74` to `input_dim: 124` and update the comment:

```yaml
model:
  input_dim: 124       # 10 players × 12 features (x,y,z,hp,armor,helmet,alive,role×5) = 120 + map_zone_onehot(4)
```

- [ ] **Step 6: Update predictor.py docstring**

In `src/inference/predictor.py`, update the docstring line:

```python
            features: float32 array of shape (seq_len, 124) — one round's state sequence.
```

- [ ] **Step 7: Run full test suite**

```
pytest tests/ -v
```

Expected: all 111+ tests pass. If any fail, investigate before committing.

- [ ] **Step 8: Commit**

```bash
git add src/model/transformer.py configs/train_config.yaml src/inference/predictor.py tests/test_model.py
git commit -m "feat: update transformer and config for 124-dim feature vector"
```

---

## Self-Review

**Spec coverage check:**
- ✅ FEATURE_DIM 74→124
- ✅ 5-dim role one-hot per player (IGL/AWPer/Entry fragger/Support/Lurker; all-zero = unknown)
- ✅ Parser writes `t{i}_role` / `ct{i}_role` columns from `player_roles` dict
- ✅ `state_vector.py` encodes role one-hot from parquet role columns
- ✅ `dataset.py` encodes roles from parquet when building tensors
- ✅ `transformer.py` slices updated: T=[0:60], CT=[60:120], zone=[120:124]
- ✅ `configs/train_config.yaml` input_dim updated
- ✅ All existing tests updated; new role tests added

**Placeholder scan:** No TBDs, all code blocks are complete.

**Type consistency:** `player_roles: dict[str, str] | None` used consistently in parser. `ROLE_IDX` dict used consistently in state_vector and dataset.
