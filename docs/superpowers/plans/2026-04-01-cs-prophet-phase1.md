# CS-Prophet Phase 1: Demo Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully tested demo-parsing pipeline that reads CS2 .dem files and outputs per-round state sequences (parquet) labelled by bomb plant site (A / B / other).

**Architecture:** `demoparser2` parses the raw .dem → `label_extractor` maps bomb-site events to string labels → `demo_parser` downsamples each round's ticks (8/sec, 30 s pre-plant window = 240 steps) and flattens 10 player state vectors into a flat parquet row per step. `map_utils` provides zone classification and coordinate normalisation used by the parser.

**Tech Stack:** Python 3.11, demoparser2 ≥ 1.5, pandas, pyarrow, pytest (with `unittest.mock` — no extra mock library needed), PyTorch (model, Phase 2+).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `CLAUDE.md` | Project context for Claude Code |
| Modify | `requirements.txt` | Add `tqdm`, `pytest-cov`; pin `awpy` |
| Modify | `configs/train_config.yaml` | Fix to match spec (3 classes, 240 steps, batch 32) |
| Create | `data/raw/demos/.gitkeep` | Placeholder so directory is tracked |
| Create | `data/processed/.gitkeep` | Placeholder |
| Create | `data/splits/.gitkeep` | Placeholder |
| Create | `dashboard/app.py` | Streamlit stub (Phase 3 placeholder) |
| Create | `src/utils/__init__.py` | Package marker |
| Create | `src/utils/map_utils.py` | Zone classification + coordinate normalisation |
| Create | `src/features/label_extractor.py` | site int/str → 'A'/'B'/'other'; get plant ticks |
| Create | `src/features/state_vector.py` | Stub (Phase 2: build model-ready tensors) |
| Create | `src/model/attention.py` | Stub (Phase 2: cross-attention module) |
| Create | `src/model/train.py` | Stub (Phase 2: training loop) |
| Create | `src/inference/onnx_export.py` | Stub (Phase 3) |
| Create | `src/inference/realtime_engine.py` | Stub (Phase 3) |
| Modify | `src/parser/demo_parser.py` | Full implementation (currently a NotImplementedError stub) |
| Create | `tests/test_map_utils.py` | Unit tests for map_utils |
| Create | `tests/test_features.py` | Unit tests for label_extractor |
| Create | `tests/test_parser.py` | Unit + integration tests for demo_parser (mocked demoparser2) |

---

## Task 1: Project Scaffold

**Files:**
- Create: `CLAUDE.md`
- Create: `data/raw/demos/.gitkeep`, `data/processed/.gitkeep`, `data/splits/.gitkeep`
- Create: `dashboard/app.py`
- Create: `src/utils/__init__.py`
- Create: `src/features/state_vector.py`
- Create: `src/model/attention.py`, `src/model/train.py`
- Create: `src/inference/onnx_export.py`, `src/inference/realtime_engine.py`

- [ ] **Step 1: Create data directory placeholders**

```bash
touch data/raw/demos/.gitkeep data/processed/.gitkeep data/splits/.gitkeep
```

- [ ] **Step 2: Create `src/utils/__init__.py`**

```python
# src/utils/__init__.py
```

- [ ] **Step 3: Create stub files**

`src/features/state_vector.py`:
```python
"""State vector builder — Phase 2 placeholder.

Will transform a parsed tick DataFrame into a fixed-size float32 feature vector
suitable for the Transformer encoder.
"""
```

`src/model/attention.py`:
```python
"""Cross-attention module — Phase 2 placeholder.

Will implement T-side Query × CT-side Key/Value cross-attention
to model adversarial interaction.
"""
```

`src/model/train.py`:
```python
"""Training loop — Phase 2 placeholder.

Will implement mixed-precision training with Focal Loss,
gradient accumulation (steps=4), and TensorBoard logging.
"""
```

`src/inference/onnx_export.py`:
```python
"""ONNX export — Phase 3 placeholder."""
```

`src/inference/realtime_engine.py`:
```python
"""Real-time inference engine — Phase 3 placeholder.

Will consume CS2 GSI events, maintain a sliding-window buffer,
run ONNX model inference (<20 ms target), and push predictions
to the Streamlit dashboard.
"""
```

`dashboard/app.py`:
```python
"""Streamlit dashboard — Phase 3 placeholder."""
import streamlit as st

st.title("CS-Prophet — Real-time Bomb Site Predictor")
st.write("Dashboard coming in Phase 3.")
```

- [ ] **Step 4: Create `CLAUDE.md`**

```markdown
# CS-Prophet — CLAUDE.md

## Project Purpose
Transformer-based CS2 professional-match prediction system.
Core task: given a round's game-state sequence up to the current tick,
predict bomb plant site → P(A) / P(B) / P(other).

## Repository Layout
```
CS_Prophet/
├── data/
│   ├── raw/demos/        ← .dem files (not committed)
│   ├── processed/        ← per-demo parquet files
│   └── splits/           ← train / val / test splits
├── src/
│   ├── parser/           ← demo_parser.py (Phase 1 ✓)
│   ├── features/         ← label_extractor.py (Phase 1 ✓), state_vector.py (Phase 2)
│   ├── model/            ← transformer.py (Phase 2), attention.py (Phase 2), train.py (Phase 2)
│   ├── inference/        ← onnx_export.py, realtime_engine.py (Phase 3)
│   └── utils/            ← map_utils.py (Phase 1 ✓)
├── dashboard/            ← app.py (Phase 3)
├── configs/              ← train_config.yaml
├── notebooks/            ← 01_eda.ipynb
└── tests/
```

## Phase Status
- **Phase 1 (current):** demo parser + map utilities — outputs labelled parquet sequences
- **Phase 2:** feature engineering + Transformer training
- **Phase 3:** ONNX export + real-time GSI inference + Streamlit dashboard

## Key Design Decisions
- Tick rate: 64 Hz raw → downsampled to **8 ticks/sec** (every 8th tick)
- Sequence length: **30 s pre-plant** = 240 steps max
- Labels: **A / B / other** (3-class), derived from `bomb_planted` event `site` field
- Players padded to 5T + 5CT; missing players zero-padded
- Coordinates normalised to [0, 1] via per-map bounding boxes in `map_utils.py`

## Parser Output Schema (parquet)
Flat table — one row per (demo, round, step):

| Column | Type | Notes |
|--------|------|-------|
| `demo_name` | str | .dem file stem |
| `round_num` | int | 1-based |
| `step` | int | 0-based within round (max 240) |
| `tick` | int | original demo tick |
| `bomb_site` | str | 'A', 'B', or 'other' |
| `map_zone` | str | mean T position zone: 'A','B','mid','other' |
| `t{i}_{x,y,z}` | float | normalised position, i=0..4 |
| `t{i}_{hp,armor}` | int | health, armour |
| `t{i}_{helmet,alive}` | bool | helmet / alive flags |
| `ct{i}_{...}` | same | CT side, i=0..4 |

## Running Tests
```bash
pytest tests/ -v
```

## Critical Dependencies
- `demoparser2 >= 1.5` — fast Rust-backed CS2 demo parser
- `pyarrow` — parquet I/O
- `torch >= 2.2` — model (Phase 2+)
```

- [ ] **Step 5: Commit scaffold**

```bash
git add data/raw/demos/.gitkeep data/processed/.gitkeep data/splits/.gitkeep \
        dashboard/app.py src/utils/__init__.py \
        src/features/state_vector.py \
        src/model/attention.py src/model/train.py \
        src/inference/onnx_export.py src/inference/realtime_engine.py \
        CLAUDE.md
git commit -m "chore: project scaffold — stubs, data dirs, CLAUDE.md"
```

---

## Task 2: requirements.txt + train_config.yaml

**Files:**
- Modify: `requirements.txt`
- Modify: `configs/train_config.yaml`

- [ ] **Step 1: Update `requirements.txt`**

Replace the entire file with:
```text
# ── Demo parsing ──────────────────────────────────────────────────────────
demoparser2>=1.5.0
awpy>=2.0.0

# ── Core ML ───────────────────────────────────────────────────────────────
torch>=2.2.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0

# ── Data storage ──────────────────────────────────────────────────────────
pyarrow>=15.0.0
fastparquet>=2024.2.0

# ── Config ────────────────────────────────────────────────────────────────
pyyaml>=6.0.1
omegaconf>=2.3.0

# ── Experiment tracking ───────────────────────────────────────────────────
tensorboard>=2.16.0

# ── Real-time / dashboard ─────────────────────────────────────────────────
streamlit>=1.33.0
onnxruntime>=1.17.0

# ── Dev / notebooks ───────────────────────────────────────────────────────
jupyter>=1.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
ipykernel>=6.29.0
tqdm>=4.66.0

# ── Testing ───────────────────────────────────────────────────────────────
pytest>=8.0.0
pytest-cov>=5.0.0
```

- [ ] **Step 2: Update `configs/train_config.yaml`**

Replace the entire file with:
```yaml
model:
  input_dim: 84        # 10 players × (x,y,z,hp,armor,helmet,alive)=7 + map_zone_onehot(4) = 74; rounded up with padding
  d_model: 256
  nhead: 8
  num_layers: 4
  dropout: 0.1
  num_classes: 3       # A / B / other

training:
  epochs: 100
  batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  val_split: 0.1
  seed: 42
  use_amp: true        # mixed-precision (torch.cuda.amp)
  focal_loss_gamma: 2.0

data:
  raw_dir: data/raw/demos
  processed_dir: data/processed
  splits_dir: data/splits
  sequence_length: 240  # 30 s × 8 ticks/s
  tick_rate: 64
  target_tick_rate: 8

logging:
  log_dir: runs/
  save_dir: checkpoints/
  log_interval: 10
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt configs/train_config.yaml
git commit -m "chore: update requirements and train config for Phase 1 spec"
```

---

## Task 3: `src/utils/map_utils.py` (TDD)

**Files:**
- Create: `tests/test_map_utils.py`
- Create: `src/utils/map_utils.py`

### 3A — Write failing tests

- [ ] **Step 1: Create `tests/test_map_utils.py`**

```python
"""Tests for map zone classification and coordinate normalisation."""

import pytest
from src.utils.map_utils import classify_zone, normalize_coords


class TestClassifyZone:
    def test_a_site_mirage(self):
        # Centre of Mirage A site box
        assert classify_zone(1080.0, -400.0, "de_mirage") == "A"

    def test_b_site_mirage(self):
        assert classify_zone(-670.0, 60.0, "de_mirage") == "B"

    def test_mid_mirage(self):
        assert classify_zone(100.0, -200.0, "de_mirage") == "mid"

    def test_outside_all_zones_returns_other(self):
        assert classify_zone(9999.0, 9999.0, "de_mirage") == "other"

    def test_unknown_map_returns_other(self):
        assert classify_zone(0.0, 0.0, "de_unknown_map") == "other"

    def test_a_site_inferno(self):
        assert classify_zone(2000.0, 800.0, "de_inferno") == "A"

    def test_b_site_inferno(self):
        assert classify_zone(500.0, -900.0, "de_inferno") == "B"

    def test_a_site_dust2(self):
        assert classify_zone(1300.0, 2200.0, "de_dust2") == "A"


class TestNormalizeCoords:
    def test_output_in_unit_cube_mirage(self):
        x_n, y_n, z_n = normalize_coords(1080.0, -400.0, 0.0, "de_mirage")
        assert 0.0 <= x_n <= 1.0
        assert 0.0 <= y_n <= 1.0
        assert 0.0 <= z_n <= 1.0

    def test_unknown_map_returns_midpoint(self):
        assert normalize_coords(0.0, 0.0, 0.0, "de_unknown") == (0.5, 0.5, 0.5)

    def test_z_clamped_below_floor(self):
        _, _, z_n = normalize_coords(0.0, 0.0, -9999.0, "de_mirage")
        assert z_n == 0.0

    def test_z_clamped_above_ceiling(self):
        _, _, z_n = normalize_coords(0.0, 0.0, 9999.0, "de_mirage")
        assert z_n == 1.0

    def test_returns_three_floats(self):
        result = normalize_coords(500.0, 200.0, 100.0, "de_inferno")
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_map_utils.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `map_utils` does not exist yet.

### 3B — Implement

- [ ] **Step 3: Create `src/utils/map_utils.py`**

```python
"""CS2 map zone utilities — zone classification and coordinate normalisation."""

from __future__ import annotations
from typing import Final

# Approximate bounding boxes per map zone: (x_min, x_max, y_min, y_max).
# Coordinates are CS2 world units as reported by demoparser2.
# Source: empirically measured from demo tick data; calibrate per map update.
_ZONE_BOXES: Final[dict[str, dict[str, tuple[float, float, float, float]]]] = {
    "de_mirage": {
        "A":   ( 630.0,  1530.0, -880.0,   70.0),
        "B":   (-1000.0,  -350.0, -360.0,  490.0),
        "mid": ( -350.0,   630.0, -880.0,  490.0),
    },
    "de_inferno": {
        "A":   (1490.0, 2870.0,  450.0, 1100.0),
        "B":   ( 180.0,  810.0, -1250.0, -550.0),
        "mid": ( 810.0, 1490.0,  -550.0,  450.0),
    },
    "de_dust2": {
        "A":   ( 660.0, 2020.0, 1650.0, 2820.0),
        "B":   (-2220.0, -1100.0,  180.0, 1080.0),
        "mid": (-1100.0,  660.0,  180.0, 1650.0),
    },
    "de_nuke": {
        "A":   (-900.0, -280.0, 1130.0, 1830.0),
        "B":   (-900.0, -280.0,  230.0, 1130.0),
        "mid": (-280.0,  900.0,  230.0, 1830.0),
    },
    "de_ancient": {
        "A":   ( 930.0, 2070.0, -520.0,  370.0),
        "B":   (-1150.0, -180.0, -690.0,  170.0),
        "mid": ( -180.0,  930.0, -690.0, -520.0),
    },
}

# z world-unit range that covers all walkable surfaces in CS2 maps
_Z_MIN: Final[float] = -500.0
_Z_MAX: Final[float] =  500.0


def classify_zone(x: float, y: float, map_name: str) -> str:
    """Return zone label ('A', 'B', 'mid', or 'other') for a world position.

    Args:
        x: World x-coordinate (demoparser2 units).
        y: World y-coordinate.
        map_name: Map name string, e.g. 'de_mirage'.

    Returns:
        Zone name string.
    """
    for zone, (x_min, x_max, y_min, y_max) in _ZONE_BOXES.get(map_name, {}).items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone
    return "other"


def normalize_coords(x: float, y: float, z: float, map_name: str) -> tuple[float, float, float]:
    """Normalise (x, y, z) to [0, 1] using per-map bounding extents.

    x/y extents are derived from the union of all zone bounding boxes.
    z is clamped to [-500, 500] (covers all walkable CS2 geometry).

    Returns (0.5, 0.5, 0.5) for unknown maps.
    """
    zones = _ZONE_BOXES.get(map_name)
    if not zones:
        return 0.5, 0.5, 0.5

    all_x = [v for box in zones.values() for v in (box[0], box[1])]
    all_y = [v for box in zones.values() for v in (box[2], box[3])]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0
    z_range = _Z_MAX - _Z_MIN

    return (
        max(0.0, min(1.0, (x - x_min) / x_range)),
        max(0.0, min(1.0, (y - y_min) / y_range)),
        max(0.0, min(1.0, (z - _Z_MIN) / z_range)),
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_map_utils.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/map_utils.py tests/test_map_utils.py
git commit -m "feat: map zone classification and coordinate normalisation (map_utils)"
```

---

## Task 4: `src/features/label_extractor.py` (TDD)

**Files:**
- Create: `tests/test_features.py`
- Create: `src/features/label_extractor.py`

### 4A — Write failing tests

- [ ] **Step 1: Create `tests/test_features.py`**

```python
"""Tests for label_extractor — bomb site label extraction."""

import pandas as pd
import pytest
from src.features.label_extractor import extract_bomb_site, get_plant_ticks


class TestExtractBombSite:
    def test_integer_0_maps_to_A(self):
        df = pd.DataFrame({"site": [0], "tick": [1000]})
        assert extract_bomb_site(df).iloc[0] == "A"

    def test_integer_1_maps_to_B(self):
        df = pd.DataFrame({"site": [1], "tick": [2000]})
        assert extract_bomb_site(df).iloc[0] == "B"

    def test_string_A_passthrough(self):
        df = pd.DataFrame({"site": ["A"], "tick": [1000]})
        assert extract_bomb_site(df).iloc[0] == "A"

    def test_string_B_passthrough(self):
        df = pd.DataFrame({"site": ["B"], "tick": [1000]})
        assert extract_bomb_site(df).iloc[0] == "B"

    def test_unknown_integer_returns_other(self):
        df = pd.DataFrame({"site": [99], "tick": [1000]})
        assert extract_bomb_site(df).iloc[0] == "other"

    def test_none_value_returns_other(self):
        df = pd.DataFrame({"site": [None], "tick": [1000]})
        assert extract_bomb_site(df).iloc[0] == "other"

    def test_missing_site_column_raises_value_error(self):
        df = pd.DataFrame({"tick": [1000]})
        with pytest.raises(ValueError, match="site"):
            extract_bomb_site(df)

    def test_multiple_rows_all_mapped(self):
        df = pd.DataFrame({"site": [0, 1, "A", 99], "tick": [100, 200, 300, 400]})
        result = extract_bomb_site(df).tolist()
        assert result == ["A", "B", "A", "other"]

    def test_returns_series(self):
        df = pd.DataFrame({"site": [0], "tick": [1000]})
        result = extract_bomb_site(df)
        assert isinstance(result, pd.Series)


class TestGetPlantTicks:
    def test_returns_int_value(self):
        df = pd.DataFrame({"site": [0], "tick": [1234]})
        assert get_plant_ticks(df).iloc[0] == 1234

    def test_missing_tick_column_raises_value_error(self):
        df = pd.DataFrame({"site": [0]})
        with pytest.raises(ValueError, match="tick"):
            get_plant_ticks(df)

    def test_coerces_float_tick_to_int(self):
        df = pd.DataFrame({"site": [0], "tick": [1234.0]})
        result = get_plant_ticks(df)
        assert result.dtype.kind == "i"  # integer kind

    def test_multiple_ticks(self):
        df = pd.DataFrame({"site": [0, 1], "tick": [100, 200]})
        result = get_plant_ticks(df).tolist()
        assert result == [100, 200]
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_features.py -v
```

Expected: `ImportError` — `label_extractor` does not exist yet.

### 4B — Implement

- [ ] **Step 3: Create `src/features/label_extractor.py`**

```python
"""Extract bomb-site labels from demoparser2 bomb_planted events."""

from __future__ import annotations
import pandas as pd

# demoparser2 returns site as int (0=A, 1=B) or sometimes string ('A'/'B')
_SITE_MAP: dict = {0: "A", 1: "B", "A": "A", "B": "B"}


def extract_bomb_site(bomb_planted_df: pd.DataFrame) -> pd.Series:
    """Map raw site values to 'A', 'B', or 'other'.

    Args:
        bomb_planted_df: DataFrame from ``DemoParser.parse_event('bomb_planted')``.
                         Must contain a 'site' column.

    Returns:
        pd.Series of str labels with the same index as the input.

    Raises:
        ValueError: If 'site' column is missing.
    """
    if "site" not in bomb_planted_df.columns:
        raise ValueError("bomb_planted_df must contain a 'site' column")
    return bomb_planted_df["site"].map(lambda v: _SITE_MAP.get(v, "other"))


def get_plant_ticks(bomb_planted_df: pd.DataFrame) -> pd.Series:
    """Return the tick series from a bomb_planted event DataFrame.

    Args:
        bomb_planted_df: DataFrame from ``DemoParser.parse_event('bomb_planted')``.
                         Must contain a 'tick' column.

    Returns:
        pd.Series of int tick values.

    Raises:
        ValueError: If 'tick' column is missing.
    """
    if "tick" not in bomb_planted_df.columns:
        raise ValueError("bomb_planted_df must contain a 'tick' column")
    return bomb_planted_df["tick"].astype(int)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_features.py -v
```

Expected: all 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features/label_extractor.py tests/test_features.py
git commit -m "feat: bomb-site label extractor with full test coverage"
```

---

## Task 5: `tests/test_parser.py` — Write failing tests first

**Files:**
- Create: `tests/test_parser.py`

- [ ] **Step 1: Create `tests/test_parser.py`**

```python
"""Unit and integration tests for demo_parser — demoparser2 is fully mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.parser.demo_parser import (
    DOWNSAMPLE,
    MAX_STEPS,
    PRE_PLANT_SECS,
    TARGET_RATE,
    TICK_RATE,
    _build_state_row,
    _extract_sequence,
    parse_demo,
)


# ── Test fixtures ─────────────────────────────────────────────────────────

def _make_tick_df(tick: int, n_t: int = 5, n_ct: int = 5) -> pd.DataFrame:
    """Minimal one-tick DataFrame matching demoparser2 column names."""
    rows = []
    for i in range(n_t):
        rows.append({
            "tick": tick, "name": f"t_p{i}", "team_name": "TERRORIST",
            "X": float(1000 + i * 10), "Y": float(500 + i * 5), "Z": 0.0,
            "health": 100, "armor_value": 100,
            "has_helmet": True, "is_alive": True,
        })
    for i in range(n_ct):
        rows.append({
            "tick": tick, "name": f"ct_p{i}", "team_name": "CT",
            "X": float(-500 + i * 10), "Y": float(200 + i * 5), "Z": 0.0,
            "health": 100, "armor_value": 100,
            "has_helmet": True, "is_alive": True,
        })
    return pd.DataFrame(rows)


def _make_multi_tick_df(ticks: list[int]) -> pd.DataFrame:
    return pd.concat([_make_tick_df(t) for t in ticks], ignore_index=True)


def _make_mock_parser(plant_tick: int = 5120, site: int = 0) -> MagicMock:
    """DemoParser mock for a single-round demo with one bomb plant."""
    start = max(0, plant_tick - PRE_PLANT_SECS * TICK_RATE)
    ticks = list(range(start, plant_tick + 1, DOWNSAMPLE))

    mock = MagicMock()
    mock.parse_event.return_value = pd.DataFrame({"site": [site], "tick": [plant_tick]})
    mock.parse_ticks.return_value = _make_multi_tick_df(ticks)
    mock.parse_header.return_value = {"map_name": "de_mirage"}
    return mock


# ── Constants ─────────────────────────────────────────────────────────────

def test_constants_match_spec():
    assert TICK_RATE == 64
    assert TARGET_RATE == 8
    assert DOWNSAMPLE == 8          # 64 // 8
    assert PRE_PLANT_SECS == 30
    assert MAX_STEPS == 240         # 30 * 8


# ── _build_state_row ──────────────────────────────────────────────────────

class TestBuildStateRow:
    def _row(self, tick: int = 100, bomb_site: str = "A"):
        return _build_state_row(
            _make_tick_df(tick), step=0, tick=tick,
            round_num=1, bomb_site=bomb_site, map_name="de_mirage",
        )

    def test_all_player_keys_present(self):
        row = self._row()
        for side in ("t", "ct"):
            for i in range(5):
                for suffix in ("_x", "_y", "_z", "_hp", "_armor", "_helmet", "_alive"):
                    assert f"{side}{i}{suffix}" in row

    def test_bomb_site_stored(self):
        assert self._row(bomb_site="B")["bomb_site"] == "B"

    def test_step_and_tick_stored(self):
        row = _build_state_row(
            _make_tick_df(500), step=7, tick=500,
            round_num=3, bomb_site="A", map_name="de_mirage",
        )
        assert row["step"] == 7
        assert row["tick"] == 500
        assert row["round_num"] == 3

    def test_normalized_coords_in_unit_cube(self):
        row = self._row()
        for side in ("t", "ct"):
            for i in range(5):
                for axis in ("_x", "_y", "_z"):
                    val = row[f"{side}{i}{axis}"]
                    assert 0.0 <= val <= 1.0, f"{side}{i}{axis} = {val}"

    def test_missing_players_zero_padded(self):
        slim = _make_tick_df(100, n_t=3, n_ct=5)
        row = _build_state_row(slim, step=0, tick=100,
                               round_num=1, bomb_site="A", map_name="de_mirage")
        assert row["t3_hp"] == 0
        assert row["t4_x"] == 0.0
        assert row["t4_alive"] is False

    def test_map_zone_key_present(self):
        assert "map_zone" in self._row()


# ── _extract_sequence ─────────────────────────────────────────────────────

class TestExtractSequence:
    def _mock_parser(self, plant_tick: int = 1280) -> MagicMock:
        start = max(0, plant_tick - PRE_PLANT_SECS * TICK_RATE)
        ticks = list(range(start, plant_tick + 1, DOWNSAMPLE))
        mock = MagicMock()
        mock.parse_ticks.return_value = _make_multi_tick_df(ticks)
        return mock

    def test_returns_dataframe(self):
        mock = self._mock_parser()
        result = _extract_sequence(mock, round_num=1, plant_tick=1280,
                                   bomb_site="A", map_name="de_mirage")
        assert isinstance(result, pd.DataFrame)

    def test_step_count_at_most_max_steps_plus_one(self):
        mock = self._mock_parser(plant_tick=5120)
        result = _extract_sequence(mock, round_num=1, plant_tick=5120,
                                   bomb_site="A", map_name="de_mirage")
        assert len(result) <= MAX_STEPS + 1

    def test_steps_are_sequential_from_zero(self):
        mock = self._mock_parser()
        result = _extract_sequence(mock, round_num=1, plant_tick=1280,
                                   bomb_site="A", map_name="de_mirage")
        steps = result["step"].tolist()
        assert steps == list(range(len(steps)))

    def test_bomb_site_column_is_consistent(self):
        mock = self._mock_parser()
        result = _extract_sequence(mock, round_num=1, plant_tick=1280,
                                   bomb_site="B", map_name="de_mirage")
        assert (result["bomb_site"] == "B").all()

    def test_returns_none_on_empty_tick_df(self):
        mock = MagicMock()
        mock.parse_ticks.return_value = pd.DataFrame()
        result = _extract_sequence(mock, round_num=1, plant_tick=1280,
                                   bomb_site="A", map_name="de_mirage")
        assert result is None

    def test_parse_ticks_called_with_correct_ticks(self):
        plant_tick = 640
        start = max(0, plant_tick - PRE_PLANT_SECS * TICK_RATE)
        expected = list(range(start, plant_tick + 1, DOWNSAMPLE))

        mock = self._mock_parser(plant_tick)
        _extract_sequence(mock, round_num=1, plant_tick=plant_tick,
                          bomb_site="A", map_name="de_mirage")

        called_ticks = mock.parse_ticks.call_args[0][1]
        assert called_ticks == expected

    def test_partial_round_when_near_start(self):
        # plant at tick 200 → only a few ticks available
        plant_tick = 200
        start = max(0, plant_tick - PRE_PLANT_SECS * TICK_RATE)
        ticks = list(range(start, plant_tick + 1, DOWNSAMPLE))
        mock = MagicMock()
        mock.parse_ticks.return_value = _make_multi_tick_df(ticks)
        result = _extract_sequence(mock, round_num=1, plant_tick=plant_tick,
                                   bomb_site="A", map_name="de_mirage")
        assert result is not None
        assert len(result) < MAX_STEPS


# ── parse_demo integration ─────────────────────────────────────────────────

class TestParseDemoIntegration:
    def test_writes_parquet_file(self, tmp_path: Path):
        mock = _make_mock_parser()
        dem = tmp_path / "match.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        assert out is not None
        assert out.exists()
        assert out.suffix == ".parquet"

    def test_output_contains_demo_name_column(self, tmp_path: Path):
        mock = _make_mock_parser()
        dem = tmp_path / "match_xyz.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        df = pd.read_parquet(out)
        assert "demo_name" in df.columns
        assert (df["demo_name"] == "match_xyz").all()

    def test_site_0_produces_A_labels(self, tmp_path: Path):
        mock = _make_mock_parser(site=0)
        dem = tmp_path / "a_site.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        df = pd.read_parquet(out)
        assert (df["bomb_site"] == "A").all()

    def test_site_1_produces_B_labels(self, tmp_path: Path):
        plant_tick = 5120
        start = max(0, plant_tick - PRE_PLANT_SECS * TICK_RATE)
        ticks = list(range(start, plant_tick + 1, DOWNSAMPLE))
        mock = MagicMock()
        mock.parse_event.return_value = pd.DataFrame({"site": [1], "tick": [plant_tick]})
        mock.parse_ticks.return_value = _make_multi_tick_df(ticks)
        mock.parse_header.return_value = {"map_name": "de_inferno"}
        dem = tmp_path / "b_site.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        df = pd.read_parquet(out)
        assert (df["bomb_site"] == "B").all()

    def test_returns_none_when_no_plant_events(self, tmp_path: Path):
        mock = MagicMock()
        mock.parse_event.return_value = pd.DataFrame()
        mock.parse_header.return_value = {"map_name": "de_mirage"}
        dem = tmp_path / "no_plants.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        assert out is None

    def test_output_dir_created_if_missing(self, tmp_path: Path):
        mock = _make_mock_parser()
        dem = tmp_path / "match.dem"
        dem.touch()
        new_dir = tmp_path / "brand_new_subdir"

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, new_dir)

        assert new_dir.exists()
        assert out is not None

    def test_output_has_expected_columns(self, tmp_path: Path):
        mock = _make_mock_parser()
        dem = tmp_path / "match.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        df = pd.read_parquet(out)
        for col in ("demo_name", "round_num", "step", "tick", "bomb_site"):
            assert col in df.columns, f"Missing column: {col}"
        for side in ("t", "ct"):
            for i in range(5):
                assert f"{side}{i}_x" in df.columns
                assert f"{side}{i}_hp" in df.columns

    def test_multi_round_demo(self, tmp_path: Path):
        plant_tick_1, plant_tick_2 = 3200, 7680
        ticks_1 = list(range(max(0, plant_tick_1 - PRE_PLANT_SECS * TICK_RATE),
                              plant_tick_1 + 1, DOWNSAMPLE))
        ticks_2 = list(range(max(0, plant_tick_2 - PRE_PLANT_SECS * TICK_RATE),
                              plant_tick_2 + 1, DOWNSAMPLE))

        mock = MagicMock()
        mock.parse_event.return_value = pd.DataFrame({
            "site": [0, 1],
            "tick": [plant_tick_1, plant_tick_2],
        })
        mock.parse_ticks.side_effect = [
            _make_multi_tick_df(ticks_1),
            _make_multi_tick_df(ticks_2),
        ]
        mock.parse_header.return_value = {"map_name": "de_mirage"}
        dem = tmp_path / "two_rounds.dem"
        dem.touch()

        with patch("src.parser.demo_parser.DemoParser", return_value=mock):
            out = parse_demo(dem, tmp_path / "out")

        df = pd.read_parquet(out)
        assert df["round_num"].nunique() == 2
        assert set(df["bomb_site"].unique()) == {"A", "B"}
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/test_parser.py -v
```

Expected: import failures and `NotImplementedError` from the stub `demo_parser.py`.

---

## Task 6: `src/parser/demo_parser.py` — Full Implementation

**Files:**
- Modify: `src/parser/demo_parser.py`

- [ ] **Step 1: Replace `demo_parser.py` with full implementation**

```python
"""Demo parser — reads CS2 .dem files and writes per-round state sequences (parquet).

Output schema (one row per step):
    demo_name, round_num, step, tick, bomb_site, map_zone,
    t{0..4}_{x,y,z,hp,armor,helmet,alive},
    ct{0..4}_{x,y,z,hp,armor,helmet,alive}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.label_extractor import extract_bomb_site, get_plant_ticks
from src.utils.map_utils import classify_zone, normalize_coords

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
TICK_RATE: int = 64
TARGET_RATE: int = 8
DOWNSAMPLE: int = TICK_RATE // TARGET_RATE   # 8
PRE_PLANT_SECS: int = 30
MAX_STEPS: int = PRE_PLANT_SECS * TARGET_RATE  # 240

PLAYER_PROPS: list[str] = [
    "X", "Y", "Z",
    "health",
    "armor_value",
    "has_helmet",
    "is_alive",
    "team_name",
    "name",
]

_TEAM_T = "TERRORIST"
_TEAM_CT = "CT"


# ── Public API ─────────────────────────────────────────────────────────────

def parse_demo(dem_path: Path | str, output_dir: Path | str) -> Optional[Path]:
    """Parse one CS2 demo and write a per-round state-sequence parquet.

    Only rounds with a confirmed bomb plant ('A' or 'B') are included.

    Args:
        dem_path: Path to the .dem file.
        output_dir: Directory for the output parquet.

    Returns:
        Path to the written parquet, or None if no valid rounds found.
    """
    dem_path = Path(dem_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Parsing demo: %s", dem_path.name)

    try:
        from demoparser2 import DemoParser  # local import keeps mock.patch simple
    except ImportError as exc:
        raise ImportError(
            "demoparser2 is not installed — run: pip install demoparser2"
        ) from exc

    parser = DemoParser(str(dem_path))

    try:
        plant_df = parser.parse_event("bomb_planted", other_props=["site"])
    except Exception:
        logger.exception("parse_event('bomb_planted') failed in %s", dem_path.name)
        return None

    if plant_df is None or plant_df.empty:
        logger.warning("No bomb_planted events in %s — skipping", dem_path.name)
        return None

    sites = extract_bomb_site(plant_df)
    plant_ticks = get_plant_ticks(plant_df)
    map_name = _get_map_name(parser)

    sequences: list[pd.DataFrame] = []
    for round_num, (plant_tick, bomb_site) in enumerate(
        zip(plant_ticks, sites), start=1
    ):
        if bomb_site not in ("A", "B"):
            logger.debug("Round %d: site %r not A/B — skipping", round_num, bomb_site)
            continue
        seq = _extract_sequence(parser, round_num, int(plant_tick), bomb_site, map_name)
        if seq is not None:
            sequences.append(seq)
            logger.debug("Round %d → site %s (%d steps)", round_num, bomb_site, len(seq))

    if not sequences:
        logger.warning("No valid sequences extracted from %s", dem_path.name)
        return None

    result = pd.concat(sequences, ignore_index=True)
    result.insert(0, "demo_name", dem_path.stem)

    out_path = output_dir / f"{dem_path.stem}.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Saved %d rows → %s", len(result), out_path)
    return out_path


def parse_demos_batch(
    dem_dir: Path | str,
    output_dir: Path | str,
    glob: str = "*.dem",
) -> list[Path]:
    """Parse all .dem files in dem_dir, writing one parquet per demo.

    Errors on individual demos are logged and skipped.

    Args:
        dem_dir: Directory containing .dem files.
        output_dir: Output directory for parquet files.
        glob: File glob (default '*.dem').

    Returns:
        List of successfully written parquet paths.
    """
    from tqdm import tqdm  # optional progress bar

    dem_dir = Path(dem_dir)
    dem_files = sorted(dem_dir.glob(glob))
    if not dem_files:
        logger.warning("No files matching '%s' in %s", glob, dem_dir)
        return []

    results: list[Path] = []
    for dem in tqdm(dem_files, desc="Parsing demos"):
        try:
            out = parse_demo(dem, output_dir)
            if out:
                results.append(out)
        except Exception:
            logger.exception("Error parsing %s — skipping", dem.name)
    return results


# ── Private helpers ────────────────────────────────────────────────────────

def _get_map_name(parser) -> str:  # noqa: ANN001
    try:
        return parser.parse_header().get("map_name", "")
    except Exception:
        return ""


def _extract_sequence(
    parser,  # noqa: ANN001
    round_num: int,
    plant_tick: int,
    bomb_site: str,
    map_name: str,
) -> Optional[pd.DataFrame]:
    """Build a downsampled state DataFrame for the 30 s window before a plant.

    Args:
        parser: demoparser2.DemoParser instance.
        round_num: 1-based round index (stored in output).
        plant_tick: Tick of the bomb_planted event.
        bomb_site: 'A' or 'B'.
        map_name: e.g. 'de_mirage'.

    Returns:
        DataFrame (one row per step) or None on failure.
    """
    start_tick = max(0, plant_tick - PRE_PLANT_SECS * TICK_RATE)
    wanted_ticks = list(range(start_tick, plant_tick + 1, DOWNSAMPLE))

    try:
        tick_df = parser.parse_ticks(PLAYER_PROPS, wanted_ticks)
    except Exception:
        logger.exception("parse_ticks failed for round %d", round_num)
        return None

    if tick_df is None or tick_df.empty:
        logger.warning("Empty tick data for round %d", round_num)
        return None

    rows: list[dict] = []
    for step, tick in enumerate(wanted_ticks):
        tick_slice = tick_df[tick_df["tick"] == tick]
        if tick_slice.empty:
            continue
        rows.append(
            _build_state_row(tick_slice, step, tick, round_num, bomb_site, map_name)
        )

    return pd.DataFrame(rows) if rows else None


def _build_state_row(
    tick_slice: pd.DataFrame,
    step: int,
    tick: int,
    round_num: int,
    bomb_site: str,
    map_name: str,
) -> dict:
    """Flatten one tick's player data into a single state dict.

    Players on each side are sorted by name for reproducibility.
    Missing players (< 5 per side) are zero-padded.

    Returns:
        Flat dict with keys round_num, step, tick, bomb_site, map_zone,
        and t{i}/ct{i} suffixed position/status columns.
    """
    t_rows = tick_slice[tick_slice["team_name"] == _TEAM_T].sort_values("name")
    ct_rows = tick_slice[tick_slice["team_name"] == _TEAM_CT].sort_values("name")

    # map_zone tracks where the T side is concentrated
    t_x_mean = float(t_rows["X"].mean()) if not t_rows.empty else 0.0
    t_y_mean = float(t_rows["Y"].mean()) if not t_rows.empty else 0.0

    row: dict = {
        "round_num": round_num,
        "step": step,
        "tick": tick,
        "bomb_site": bomb_site,
        "map_zone": classify_zone(t_x_mean, t_y_mean, map_name),
    }

    for side_prefix, side_rows in (("t", t_rows), ("ct", ct_rows)):
        for i in range(5):
            prefix = f"{side_prefix}{i}"
            if i < len(side_rows):
                p = side_rows.iloc[i]
                x_n, y_n, z_n = normalize_coords(
                    float(p.get("X", 0.0)),
                    float(p.get("Y", 0.0)),
                    float(p.get("Z", 0.0)),
                    map_name,
                )
                row[f"{prefix}_x"] = x_n
                row[f"{prefix}_y"] = y_n
                row[f"{prefix}_z"] = z_n
                row[f"{prefix}_hp"] = int(p.get("health", 0))
                row[f"{prefix}_armor"] = int(p.get("armor_value", 0))
                row[f"{prefix}_helmet"] = bool(p.get("has_helmet", False))
                row[f"{prefix}_alive"] = bool(p.get("is_alive", False))
            else:
                row[f"{prefix}_x"] = 0.0
                row[f"{prefix}_y"] = 0.0
                row[f"{prefix}_z"] = 0.0
                row[f"{prefix}_hp"] = 0
                row[f"{prefix}_armor"] = 0
                row[f"{prefix}_helmet"] = False
                row[f"{prefix}_alive"] = False

    return row
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests in `test_map_utils.py`, `test_features.py`, `test_parser.py`, and `test_model.py` PASS.

- [ ] **Step 3: Check coverage**

```bash
pytest tests/test_parser.py tests/test_features.py tests/test_map_utils.py \
       --cov=src/parser --cov=src/features --cov=src/utils \
       --cov-report=term-missing
```

Aim: ≥ 85 % coverage on all three modules.

- [ ] **Step 4: Commit**

```bash
git add src/parser/demo_parser.py tests/test_parser.py
git commit -m "feat: Phase 1 demo parser — full implementation with 240-step parquet output"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec requirement | Task |
|---|---|
| Directory structure created | Task 1 |
| requirements.txt with CUDA 12.x / Python 3.11 deps | Task 2 |
| CLAUDE.md project context | Task 1 |
| Parse single .dem, split by round | Task 6 |
| Extract bomb_planted site label | Task 4 + 6 |
| Downsample to 8 tick/sec | Task 6 (DOWNSAMPLE=8) |
| Output parquet to data/processed/ | Task 6 |
| Progress logging | Task 6 (logger calls) |
| Error handling | Task 6 (try/except, None returns) |
| tests/test_parser.py | Task 5 + 6 |

### Notes for Phase 2
- `src/model/transformer.py` currently has `num_classes=2` (CT/T win). Must be updated to `3` (A/B/other) before training.
- `src/features/state_vector.py` is a stub — needs implementation to build the 84-dim feature vector from parquet rows.
- `src/features/feature_engineering.py` (existing) has an incompatible interface; it will be superseded by `state_vector.py` in Phase 2.
