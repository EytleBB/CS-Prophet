#!/usr/bin/env python3
"""Batch extract demos to *_full.pkl then convert to processed_v2 parquet.

Usage:
    python tools/batch_extract_v2.py
    python tools/batch_extract_v2.py --per-map 2 --resume
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path

MAP_ORDER = [
    "de_ancient", "de_anubis", "de_dust2", "de_inferno",
    "de_mirage", "de_nuke", "de_overpass",
]


def select_demos(per_map: int) -> list[Path]:
    raw_dir = data_path("raw", "demos")
    by_map: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for dem in raw_dir.glob("*.dem"):
        parts = dem.stem.split("_", 1)
        if len(parts) != 2:
            continue
        try:
            by_map[parts[1]].append((int(parts[0]), dem))
        except ValueError:
            pass

    selected = []
    for m in MAP_ORDER:
        candidates = sorted(by_map.get(m, []), reverse=True)
        for _, dem in candidates[:per_map]:
            selected.append(dem)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-map", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    demos = select_demos(args.per_map)
    viz_dir = data_path("viz")
    pv2_dir = data_path("processed_v2")
    python = sys.executable

    print(f"Selected {len(demos)} demos across {len(MAP_ORDER)} maps\n")

    extracted = 0
    converted = 0
    skipped_extract = 0
    skipped_convert = 0

    for i, dem in enumerate(demos, 1):
        stem = dem.stem
        pkl_path = viz_dir / f"{stem}_full.pkl"
        pq_path = pv2_dir / f"{stem}.parquet"

        print(f"\n[{i}/{len(demos)}] {stem}")

        # Step 1: extract to pkl
        if args.resume and pkl_path.exists():
            print(f"  pkl exists, skipping extract")
            skipped_extract += 1
        else:
            print(f"  extracting -> {pkl_path.name}")
            ret = subprocess.run(
                [python, "tools/demo_full_extract.py", str(dem)],
                cwd=str(REPO_ROOT),
            )
            if ret.returncode != 0:
                print(f"  EXTRACT FAILED (rc={ret.returncode})")
                continue
            extracted += 1

        # Step 2: convert pkl to processed_v2
        if args.resume and pq_path.exists():
            print(f"  parquet exists, skipping convert")
            skipped_convert += 1
        else:
            if not pkl_path.exists():
                print(f"  pkl not found, skipping convert")
                continue
            print(f"  converting -> {pq_path.name}")
            ret = subprocess.run(
                [python, "tools/build_processed_v2.py", str(pkl_path)],
                cwd=str(REPO_ROOT),
            )
            if ret.returncode != 0:
                print(f"  CONVERT FAILED (rc={ret.returncode})")
                continue
            converted += 1

    print(f"\n{'='*50}")
    print(f"Done. extracted={extracted} converted={converted}")
    print(f"     skipped_extract={skipped_extract} skipped_convert={skipped_convert}")

    # Summary
    pqs = sorted(pv2_dir.glob("*.parquet"))
    print(f"\nprocessed_v2 parquets: {len(pqs)}")
    for p in pqs:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
