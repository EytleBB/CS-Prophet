#!/usr/bin/env python3
"""Build processed_v2 parquet files from extracted *_full.pkl payloads.

Usage:
    python tools/build_processed_v2.py
    python tools/build_processed_v2.py viz --out-dir processed_v2 --resume
    python tools/build_processed_v2.py viz/2389983_de_dust2_full.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.processed_v2 import export_full_pkl_to_processed_v2, infer_demo_name
from src.utils.paths import ensure_data_layout, resolve_path_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 218-dim processed_v2 parquet files.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default="viz",
        help="A *_full.pkl file or a directory containing them (default: viz)",
    )
    parser.add_argument(
        "--out-dir",
        default="processed_v2",
        help="Output directory for processed_v2 parquet files",
    )
    parser.add_argument(
        "--glob",
        default="*_full.pkl",
        help="Glob used when input_path is a directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files whose output parquet already exists",
    )
    return parser.parse_args()


def _iter_inputs(input_path: Path, glob_pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(glob_pattern))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> int:
    args = parse_args()
    ensure_data_layout()

    input_path = resolve_path_input(args.input_path)
    output_dir = resolve_path_input(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_pkls = _iter_inputs(input_path, args.glob)
    if not full_pkls:
        print(f"No *_full.pkl files found in {input_path}")
        return 1

    exported = 0
    skipped = 0
    for full_pkl in full_pkls:
        demo_name = infer_demo_name(full_pkl)
        out_path = output_dir / f"{demo_name}.parquet"
        if args.resume and out_path.exists():
            print(f"[skip] {out_path.name} already exists")
            skipped += 1
            continue

        result = export_full_pkl_to_processed_v2(full_pkl, output_dir)
        if result is None:
            print(f"[skip] {full_pkl.name}: no labeled A/B rounds")
            skipped += 1
            continue

        print(f"[ok] {full_pkl.name} -> {result.name}")
        exported += 1

    print(f"Done. exported={exported} skipped={skipped} total={len(full_pkls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
