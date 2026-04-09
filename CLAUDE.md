# CS-Prophet — CLAUDE.md

## Project Purpose
Transformer-based CS2 professional-match prediction system.
Core task: given a round's game-state sequence up to the current tick,
predict bomb plant site → P(A) / P(B) (2-class; rounds without confirmed A/B plant are excluded).

## Repository Layout
```
CS_Prophet/
├── data/
│   ├── raw/demos/        ← .dem files (not committed)
│   ├── processed/        ← per-demo parquet files
│   └── splits/           ← train / val / test splits
├── src/
│   ├── parser/           ← demo_parser.py (Phase 1 ✓)
│   ├── features/         ← label_extractor.py (Phase 1 ✓)
│   │                        state_vector.py (Phase 2 ✓)
│   │                        dataset.py (Phase 2 ✓)
│   ├── model/            ← attention.py (Phase 2 ✓)
│   │                        transformer.py (Phase 2 ✓)
│   │                        train.py (Phase 2 ✓)
│   ├── inference/        ← predictor.py (Phase 2 ✓, stub)
│   │                        onnx_export.py (Phase 3)
│   │                        realtime_engine.py (Phase 3)
│   └── utils/            ← map_utils.py (Phase 1 ✓)
├── dashboard/            ← app.py (Phase 3)
├── configs/              ← train_config.yaml
├── notebooks/            ← 01_eda.ipynb
└── tests/                ← 155 tests, all passing
```

## Phase Status

### Phase 1 — Demo Parser (COMPLETE)
- `src/utils/map_utils.py` — zone classification + coord normalisation for 7 maps
  - Maps: de_mirage, de_inferno, de_dust2, de_nuke, de_ancient, de_overpass, de_anubis
- `src/features/label_extractor.py` — maps site int/str to 'A'/'B'/'other'
- `src/parser/demo_parser.py` — full demo parsing pipeline → parquet output

### Phase 2 — Feature Engineering + Model Training (COMPLETE)
- `src/features/state_vector.py` — 74-dim float32 feature vector builder
- `src/features/dataset.py` — `RoundSequenceDataset`, `split_files` (file-level splits)
- `src/model/attention.py` — `PositionalEncoding`, `CrossAttentionLayer`
- `src/model/transformer.py` — `BombSiteTransformer` (3-class, T×CT cross-attention)
- `src/model/train.py` — `FocalLoss`, AMP training loop, checkpoint saving
- `src/inference/predictor.py` — `RoundPredictor` stub (wired to Phase 2 API)

### Phase 3 — Inference + Dashboard (PENDING)
- `src/inference/onnx_export.py` — export trained model to ONNX
- `src/inference/realtime_engine.py` — sliding-window GSI consumer
- `dashboard/app.py` — Streamlit probability display

## Key Design Decisions
- Tick rate: 64 Hz raw → downsampled to **8 ticks/sec** (every 8th tick)
- Sequence length: **90 s** = 720 steps max (at 8 ticks/s)
- Labels: **A / B** (2-class); rounds without confirmed A/B plant are skipped in dataset
- Players padded to 5T + 5CT; missing players zero-padded
- Coordinates normalised to [0, 1] via per-map bounding boxes in `map_utils.py`
- Feature vector: 275 dims — [0:135] T-players, [135:270] CT-players, [270:275] global
- Model reads from last **real** (non-padded) timestep using src_key_padding_mask
- Checkpoint saves `model_config` so `RoundPredictor` can self-describe

## Feature Vector Layout (275 dims)
```
Per-player stride = 27 dims:
  [0:7]   x, y, z, hp/100, armor/100, helmet, alive
  [7:12]  role one-hot (IGL, AWPer, Entry fragger, Support, Lurker)
  [12:19] weapon_category one-hot (pistol, rifle, sniper, smg, heavy, grenade, other)
  [19]    has_smoke
  [20]    has_flash
  [21]    has_he
  [22]    has_molotov
  [23]    flash_duration / 3.0
  [24]    equip_value / 20000.0
  [25]    is_scoped
  [26]    is_defusing

[0:135]    T players 0–4  × 27
[135:270]  CT players 0–4 × 27
[270]      ct_score / 30
[271]      t_score / 30
[272]      round_num / 30
[273]      ct_losing_streak / 5
[274]      t_losing_streak / 5
```

## BombSiteTransformer Architecture
1. Split 275-dim input → T-side (135 player + 5 global = 140-dim) and CT-side (135-dim)
2. Project both to d_model with learned linear layers
3. Add sinusoidal positional encoding independently to each stream
4. Cross-attention: T queries CT to model adversarial interaction
5. Self-attention encoder stack on T-side representation
6. Linear classifier on last real (non-padded) timestep → (3,) logits

## Parser Output Schema (parquet)
Flat table — one row per (demo, round, step):

| Column | Type | Notes |
|--------|------|-------|
| `demo_name` | str | .dem file stem |
| `round_num` | int | 1-based |
| `step` | int | 0-based within round (max 240) |
| `tick` | int | original demo tick |
| `bomb_site` | str | 'A', 'B', or 'other' |
| `ct_score`, `t_score` | int | score at round start |
| `ct_losing_streak`, `t_losing_streak` | int | consecutive losses |
| `t{i}_{x,y,z}` | float | normalised position, i=0..4 |
| `t{i}_{hp,armor}` | int | health, armour |
| `t{i}_{helmet,alive}` | bool | helmet / alive flags |
| `t{i}_weapon` | str | weapon category string |
| `t{i}_{has_smoke,has_flash,has_he,has_molotov}` | bool | grenade inventory |
| `t{i}_flash_duration` | float | seconds still blinded |
| `t{i}_equip_value` | int | equipment value this round |
| `t{i}_{is_scoped,is_defusing}` | bool | action flags |
| `ct{i}_{...}` | same | CT side, i=0..4 |

## Running Tests
```bash
pytest tests/ -v
```
Current: 111 tests, all passing.

## Critical Dependencies
- `demoparser2 >= 1.5` — fast Rust-backed CS2 demo parser
- `pyarrow` — parquet I/O
- `torch >= 2.2` — model
- `omegaconf` — config loading
- `tqdm` — progress display (Phase 3)
