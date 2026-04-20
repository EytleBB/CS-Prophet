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

## Current Experimental Feature Work (2026-04-13)
- There is now a separate experimental feature workflow under `tools/`.
- Relevant files:
  - `src/features/state_vector_v2.py`
  - `src/features/feature_builder_v2.py`
  - `tools/demo_full_extract.py`
  - `tools/demo_feature_preview.py`
  - `tools/validate_feature_preview.py`
- The schema order is now formalized in `src/features/state_vector_v2.py`.
- `src/features/feature_builder_v2.py` now builds canonical raw v2 rows from extracted `*_full.pkl` payloads.
- This workflow is not yet wired into the production dataset or training pipeline.
- The main training path still uses the older `275`-dim schema documented below.

### Experimental 348-dim preview schema
- Per-player: `31` dims each, `10` players total = `310`
- Global: `4` dims
  - `ct_score`
  - `t_score`
  - `round_num`
  - `time_in_round`
- Bomb: `3` dims
  - `bomb_dropped`
  - `bomb_x`
  - `bomb_y`
- Active utility: `24` dims
  - smokes: `5 x (x, y, remain)`
  - molotovs: `3 x (x, y, remain)`
- Map one-hot: `7` dims
- Total: `348`

### Experimental bomb-state handling
- `bomb_dropped`, `bomb_pickup`, and `bomb_planted` are extracted with player `X/Y/Z` from `demoparser2`.
- Extraction normalizes event coordinate columns to stable `X/Y/Z`.
- Preview feature generation uses exact event coordinates for dropped-C4 location whenever available.
- For older pkls without event coordinates, it falls back to the nearest player snapshot.
- Bomb events are filtered by round time window (`freeze_tick -> end_tick`) rather than trusting sparse `round_num` values alone.

### Experimental validation status
- Raw demo used for validation:
  - `data/raw/demos/2389983_de_dust2.dem`
- Re-extracted pkl:
  - `data/viz/2389983_de_dust2_full.pkl`
- Validation command:
  - `python tools/validate_feature_preview.py data/viz/2389983_de_dust2_full.pkl --all`
- Result:
  - validated `14876/14876` unique ticks
  - `4,938,843` checks
  - `0` failures
- Validator coverage:
  - per-tick 5T/5CT structure
  - player features against raw `tick_df`
  - global features against `round_info`
  - bomb state against event timeline
  - active smoke/molotov slots against event-derived expectations
  - one-hot integrity and final dimension count
- Additional v2 code checks completed:
  - targeted unit tests for `label_extractor`, `feature_builder_v2`, and `state_vector_v2` all passed
  - full real-demo sanity run over all `14876` unique ticks produced finite `(348,)` vectors with observed value range `[-0.9999942, 1.0]`

### Experimental data-cleaning note
- Some freeze-end ticks contain `NaN` in `velocity_X/Y/Z`.
- The experimental preview builder now coerces those numeric NaNs to `0.0` before emitting features.
- `src/features/label_extractor.py` now supports both `user_X/user_Y/user_Z` and normalized `X/Y/Z` planted-coordinate columns.

### Experimental normalization policy
- `state_vector_v2.py` no longer just packs raw values; it now normalizes them for model input.
- Positions and bomb / utility coordinates are normalized via `src.utils.map_utils.normalize_coords`.
- Signed velocities and yaw are scaled to `[-1, 1]`.
- Scalar magnitudes such as hp, armor, flash duration, money, score, round number, and time-in-round are clipped to `[0, 1]`.

## External Data Root (2026-04-14)
- The project no longer assumes repo-local `data/` is the primary storage location.
- Shared path logic now lives in `src/utils/paths.py`.
- Active data-root priority:
  - `CS_PROPHET_DATA_ROOT`
  - `H:\CS_Prophet\data` on this Windows machine
  - repo-local `data/` fallback
- Core scripts/configs were updated to use the shared resolver instead of hardcoded `data/...` paths.
- Repo-local `data/` should now be treated as lightweight placeholder/fallback only.

### Current machine state
- External data root has been created at `H:\CS_Prophet\data`.
- Existing local data has already been migrated there.
- Verified migrated content:
  - `218` raw demos in `raw/demos`
  - `693` processed parquet files in `processed`
  - `2` extracted `*_full.pkl` files in `viz`

### Operational guidance
- When adding or updating code, prefer `src/utils/paths.py` helpers:
  - `data_root()`
  - `data_path(...)`
  - `resolve_path_input(...)`
- Prefer data-root-relative config values like `raw/demos` and `processed`.
- Only use repo-local `data/...` paths as compatibility input, not as the primary storage target.

## V2 Data Pipeline (2026-04-14)
- The old `processed/*.parquet` files remain valid only for the legacy `275`-dim schema.
- The new `348`-dim schema now has a parallel pipeline and should not overwrite the old data.

### New modules
- `src/features/processed_v2.py`
  - converts extracted `*_full.pkl` payloads into `processed_v2/*.parquet`
- `src/features/dataset_v2.py`
  - `RoundSequenceDatasetV2`
  - same training-facing interface as the old dataset, but emits `348`-dim sequences
- `tools/build_processed_v2.py`
  - batch export utility for building `processed_v2`

### Training integration
- `src/model/train.py` now switches by `data.schema_version`
  - `v1` -> old dataset / `275` dims
  - `v2` -> new dataset / `348` dims
- `src/model/transformer.py` now supports both `input_dim=275` and `input_dim=348`
- New configs:
  - `configs/train_config_v2.yaml`
  - `configs/train_config_v2.smoke.yaml`

### Current real-data v2 state
- Exported real `processed_v2` parquet files:
  - `H:\CS_Prophet\data\processed_v2\2389983_de_dust2.parquet`
  - `H:\CS_Prophet\data\processed_v2\2389662_de_dust2.parquet`
  - `H:\CS_Prophet\data\processed_v2\2389648_de_dust2.parquet`
- `2389471_de_mirage_full.pkl` currently exports no `processed_v2` parquet because it has no labeled `A/B` rounds under the current label extraction.

### Validation status
- Unit/integration coverage now includes `tests/test_processed_v2.py`
- Focused tests after the v2 pipeline work: `36 passed`
- Real-data smoke training completed successfully with:
  - `python -m src.model.train --config configs/train_config_v2.smoke.yaml`
- This confirms the new path works end-to-end:
  - extracted `*_full.pkl`
  - `processed_v2`
  - `dataset_v2`
  - `BombSiteTransformer(348)`
  - training loop

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
