# Work Log

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
