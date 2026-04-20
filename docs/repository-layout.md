# Repository Layout

This note keeps the repository GitHub-friendly without changing runtime paths or
moving source code around.

## What belongs in Git

Keep these directories as the main tracked surface:

- `src/`: parser, features, model, inference, utility code
- `src/memory_reader/`: vendored offset metadata consumed by the realtime
  memory reader
- `tests/`: automated tests and test helpers
- `configs/`: train and inference YAML configs
- `tools/`: reusable scripts and CLIs that are part of the project
- `dashboard/`: browser UI assets for realtime viewing
- `cfg/`: CS2 Game State Integration config files
- `docs/`: design notes, plans, and repository-facing documentation
- `notebooks/`: lightweight notebooks only when they are intentional examples

Top-level scripts such as `parse_demos.py`, `analyze_demo.py`, and
`viz_parquet.py` are still part of the tracked interface and were left in place
to avoid breaking existing commands or docs.

## What stays local

The repo now treats these areas as local-only/generated unless there is a very
specific reason to publish them:

- `data/`: repo-local fallback for logs, manifests, reports, dumps, and small
  temporary assets
- `processed_v2_2hz_preplant/`: generated parquet exports
- `scratch_v2_export/`: scratch exports
- `checkpoints/v2/` and `checkpoints/v2_2hz/`: local model artifacts
- `tools/analysis_output*/`, `tools/compare_output/`, `tools/data/`: generated
  analysis results
- `.claude/`, `.codex_task*.txt`, `.tmp/`, `pytest-basetemp/`, `test-temp/`,
  `tmp*/`: local tooling and temp state

## Data path policy

Data path resolution is centralized in `src/utils/paths.py`.

Current preference order:

1. `CS_PROPHET_DATA_ROOT`
2. `H:\CS_Prophet\data` on Windows if available
3. repo-local `data/` as a fallback

That means keeping large or noisy outputs out of Git is now the safe default,
and repo-local `data/` should mostly carry placeholders or lightweight
documentation.

## Intentional tracked artifacts

A few small artifacts are still tracked on purpose today:

- `checkpoints/best.pt`
- `checkpoints/smoke/best.pt`
- `src/memory_reader/client_dll.json`
- `src/memory_reader/offsets.json`
- `.gitkeep` placeholders under `data/`

If you decide not to publish model weights, remove them deliberately rather than
through a blanket cleanup.

## Upload checklist

Before pushing to GitHub:

1. Review `git status --short` and confirm only source/doc changes remain.
2. Avoid force-adding ignored local outputs.
3. Double-check any new file under `data/`, `checkpoints/`, or `tools/` to
   confirm it is really source, not a generated artifact.
4. Keep `tools/hltv_secrets.yaml` local only.
