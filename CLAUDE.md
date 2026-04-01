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
