# Work Log

## 2026-04-13

### Feature-schema progress
- Added experimental demo feature tooling under `tools/`:
  - `tools/demo_full_extract.py`
  - `tools/demo_feature_preview.py`
  - `tools/validate_feature_preview.py`
- Formalized the fixed-order experimental schema in:
  - `src/features/state_vector_v2.py`
- Added canonical raw-row builder in:
  - `src/features/feature_builder_v2.py`
- Re-extracted `data/viz/2389983_de_dust2_full.pkl` from raw demo:
  - source demo: `data/raw/demos/2389983_de_dust2.dem`
- The experimental preview schema is now `348` dims:
  - `10 x 31 = 310` player dims
  - `4` global dims: `ct_score`, `t_score`, `round_num`, `time_in_round`
  - `3` bomb dims: `bomb_dropped`, `bomb_x`, `bomb_y`
  - `24` active utility dims: `smoke 5 x 3`, `molotov 3 x 3`
  - `7` map one-hot dims

### Bomb-state fixes
- `bomb_dropped`, `bomb_pickup`, and `bomb_planted` now request player `X/Y/Z` directly from `demoparser2`.
- Event coordinates are normalized to stable `X/Y/Z` column names during extraction.
- Bomb state in the preview feature builder now:
  - uses round time windows based on `freeze_tick -> end_tick`
  - prefers exact event coordinates for dropped C4
  - falls back to nearest player snapshot only for older extracted pkls that lack event coordinates
- This fixed an earlier issue where some early-round ticks could incorrectly appear as `bomb_dropped=True` because `round_num = NA` bomb events were being treated too loosely.

### Data-cleaning fixes
- Added defensive numeric conversion in the preview feature builder.
- `NaN` values in tick data, especially `velocity_X/Y/Z` near freeze end, are now coerced to `0` before entering the feature vector.
- Added explicit v2 normalization policy in `src/features/state_vector_v2.py`:
  - positions and bomb / utility coordinates -> normalized by per-map bounds
  - signed velocities and yaw -> scaled to `[-1, 1]`
  - hp / armor / flash / money / score / time / round scalars -> clipped to `[0, 1]`
  - booleans and one-hot fields -> `0/1`

### Validation completed
- Wrote `tools/validate_feature_preview.py` to validate the 348-dim schema automatically.
- The validator independently checks:
  - dataset structure (`10` players per tick, `5T + 5CT`, supported map)
  - event schema (`bomb_*` events include `X/Y/Z`)
  - player-slot features against raw `tick_df`
  - global features against `round_info`
  - bomb state against bomb event timeline
  - active smoke/molotov slots against event-derived expected values
  - one-hot integrity and final feature dimension count
- Validation results on `2389983_de_dust2_full.pkl`:
  - sample run: `512/512` unique ticks passed
  - full run: `14876/14876` unique ticks passed
  - total checks: `4,938,843`
  - failures: `0`
- Added unit tests for:
  - `src/features/label_extractor.py` normalized `X/Y/Z` support
  - `src/features/feature_builder_v2.py`
  - `src/features/state_vector_v2.py`
- Current focused test status:
  - `28` targeted tests passed
- Full real-demo v2 build sanity check:
  - built vectors for all `14876` unique ticks in `2389983_de_dust2_full.pkl`
  - no NaN / Inf
  - vector shape always `(348,)`
  - observed value range: `[-0.9999942, 1.0]`

### Important note
- The validated `348`-dim schema is now formally defined in `src/features/state_vector_v2.py`.
- The main training pipeline in `src/features/state_vector.py` still uses the older `275`-dim schema.
- `state_vector_v2.py` is now schema + normalization policy.
- `feature_builder_v2.py` can build canonical raw rows from `*_full.pkl` payloads.
- The v2 path has not yet been wired into dataset loading or training.

## 2026-04-14

### External data-root migration
- Added `src/utils/paths.py` to centralize data-path resolution.
- Active data-root priority is now:
  - `CS_PROPHET_DATA_ROOT`
  - `H:\CS_Prophet\data` on Windows when available
  - repo-local `data/` as fallback
- Updated core entry points to use the shared resolver:
  - `parse_demos.py`
  - `src/parser/demo_parser.py`
  - `src/model/train.py`
  - `tools/download_demos.py`
  - `tools/pipeline.py`
  - `tools/demo_full_extract.py`
  - `tools/demo_feature_preview.py`
  - `tools/demo_visualize.py`
  - `tools/validate_feature_preview.py`
  - `analyze_demo.py`
  - `check_labels.py`
  - `viz_parquet.py`
- Updated configs to be data-root relative instead of hardcoding repo-local `data/...`:
  - `configs/train_config.yaml`
  - `configs/train_config.smoke.yaml`
  - `tools/hltv_config.yaml`

### External storage layout created
- Created `H:\CS_Prophet\data` with:
  - `raw/demos`
  - `processed`
  - `splits`
  - `viz`
  - `viz/assets`
  - `tmp`

### Existing local data migrated
- Moved repo-local dataset content from `D:\CSAI\CS_Prophet\data` to `H:\CS_Prophet\data`.
- Verified migrated counts:
  - raw demos: `218`
  - processed parquets: `693`
  - extracted `*_full.pkl`: `2`
- Repo-local `data/` is now treated as lightweight placeholder/fallback only.
- Added `data/README.md` explaining the new storage contract.

### Sanity checks
- `python -m py_compile` passed for the updated path-aware scripts/modules.
- Verified resolved defaults now point to `H:\CS_Prophet\data`.
- Legacy `data/...` inputs still resolve correctly during transition.

### V2 data pipeline started
- Added `src/features/processed_v2.py`:
  - converts extracted `*_full.pkl` payloads into row-wise `processed_v2` parquet files
  - metadata columns: `demo_name`, `map_name`, `bomb_site`, `step`, `tick`
  - feature payload columns follow `src/features/state_vector_v2.py::FEATURE_NAMES`
- Added `src/features/dataset_v2.py`:
  - `RoundSequenceDatasetV2`
  - same training-facing interface as the old dataset
  - consumes `processed_v2/*.parquet` and emits `(sequence_length, 348)` tensors
- Added `tools/build_processed_v2.py` to batch-export `processed_v2` from `viz/*_full.pkl`.
- Added configs:
  - `configs/train_config_v2.yaml`
  - `configs/train_config_v2.smoke.yaml`
- Updated `src/model/train.py` to switch between `v1` and `v2` via `data.schema_version`.
- Updated `src/model/transformer.py` to support both `275` and `348` input schemas.

### V2 tests and validation
- Added `tests/test_processed_v2.py`.
- Current focused test status after v2 pipeline work:
  - `36` tests passed across model/train/v2 feature/export coverage
- Real-data `processed_v2` exports completed:
  - `H:\CS_Prophet\data\processed_v2\2389983_de_dust2.parquet`
  - `H:\CS_Prophet\data\processed_v2\2389662_de_dust2.parquet`
  - `H:\CS_Prophet\data\processed_v2\2389648_de_dust2.parquet`
- `2389471_de_mirage_full.pkl` was skipped during `processed_v2` export because it had no labeled `A/B` rounds under the current label extraction.
- Verified real-data v2 dataset load on `2389983_de_dust2.parquet`:
  - rows: `3854`
  - labeled rounds: `7`
  - emitted sample tensor shape: `(720, 348)`
  - all values finite

### V2 smoke training
- Ran `python -m src.model.train --config configs/train_config_v2.smoke.yaml`
- Result:
  - training loop completed successfully on the new `348`-dim data path
  - smoke split used `3` processed_v2 parquet files
  - epoch 1 finished with:
    - train loss: `0.4252`
    - val loss: `0.0122`
    - val acc: `1.000`
- This verifies the new path end-to-end:
  - `*_full.pkl`
  - `processed_v2`
  - `dataset_v2`
  - `BombSiteTransformer(input_dim=348)`
  - training loop

## 2026-04-09

### Project understanding
- The repository's actual task is bomb-site prediction, not round-win prediction.
- Ground-truth labels come from `bomb_planted` events.
- If `user_X/user_Y/user_Z` are present, the code classifies site by map bounding boxes in `src/utils/map_utils.py`.
- If coordinates are unavailable, it falls back to the event `site` field.
- Training data keeps only rounds with confirmed `A` or `B` plants.

### Label-quality conclusion
- The map zone boxes are heuristic and empirically calibrated, not formally validated.
- The repo claims they were calibrated from `demoparser2` coordinates across HLTV demos, but it does not include a reproducible calibration dataset or accuracy report.
- Current tests only verify representative sample points, not end-to-end labeling accuracy.

### Training diagnosis
- Running `python -m src.model.train` twice produced the same result because training is deterministic enough for this setup:
  - fixed seed in `src/model/train.py`
  - deterministic file split in `src/features/dataset.py`
- Early stopping at epoch 13 is expected behavior:
  - `early_stop_patience = 10`
  - the saved best checkpoint was from epoch 3 (`epoch: 2` in `checkpoints/best.pt`)
- Current data split observed during analysis:
  - files: 165 total, 133 train, 16 val, 16 test
  - rounds: 1435 train, 127 val, 176 test
  - labels are roughly balanced between A and B
- A major bottleneck is low optimizer update count:
  - `batch_size = 8`
  - `gradient_accumulation_steps = 16`
  - about 180 batches/epoch but only 12 optimizer updates/epoch
- `acc ~ 0.59` is above random, but not strong.
- More data is a reasonable direction, especially if the dataset can be expanded to several thousand or more rounds from similar competitive demos.

### Code changes completed
- Updated `src/model/train.py`:
  - added real `--config` CLI support via `argparse`
  - logs the config path actually used
  - logs file split sizes, round counts, label counts, batches/epoch, and optimizer updates/epoch
  - logs richer per-epoch status: current train/val loss, acc, best val loss, best epoch, and patience counter
- Added `configs/train_config.smoke.yaml` for one-epoch smoke validation without overwriting the main training setup

### Validation completed
- Smoke run with `D:/EDGE/python.exe -m src.model.train --config configs/train_config.smoke.yaml` succeeded.
- Verified that `--config` now works and that the new diagnostic logs are printed.

### Recommended next steps
- First training changes to try before deeper architecture work:
  - lower `gradient_accumulation_steps`
  - consider reducing `dropout`
  - re-evaluate whether `FocalLoss` is necessary for a roughly balanced A/B dataset
- If scaling data:
  - prioritize more HLTV professional demos from similar map pool and similar game-version window
  - keep train/val/test split at file level
  - re-tune training config after data expansion rather than only adding more files
- If validating labels:
  - compare coordinate-based site assignment against `bomb_planted.site`
  - produce per-map agreement statistics before trusting the heuristic boxes as primary truth
