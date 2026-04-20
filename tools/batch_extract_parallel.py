#!/usr/bin/env python3
"""Parallel batch: .dem -> *_full.pkl -> processed_v2/*.parquet

Uses multiprocessing to saturate available CPU cores.

Usage:
    python tools/batch_extract_parallel.py
    python tools/batch_extract_parallel.py --workers 10 --resume
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path


def process_one(
    dem: Path,
    python: str,
    resume: bool,
    downsample: int,
    viz_subdir: str,
    processed_subdir: str,
) -> str:
    """Extract + convert a single demo. Returns status string."""
    stem = dem.stem
    viz_dir = data_path(viz_subdir)
    pv2_dir = data_path(processed_subdir)
    pkl_path = viz_dir / f"{stem}_full.pkl"
    pq_path = pv2_dir / f"{stem}.parquet"

    # Step 1: extract
    if resume and pkl_path.exists():
        pass
    else:
        ret = subprocess.run(
            [python, "tools/demo_full_extract.py", str(dem),
             "--downsample", str(downsample),
             "--output-dir", viz_subdir],
            cwd=str(REPO_ROOT),
            capture_output=True,
        )
        if ret.returncode != 0:
            return f"[FAIL extract] {stem}: {ret.stderr[-200:] if ret.stderr else ''}"

    # Step 2: convert
    if resume and pq_path.exists():
        return f"[skip] {stem}"

    if not pkl_path.exists():
        return f"[FAIL no pkl] {stem}"

    ret = subprocess.run(
        [python, "tools/build_processed_v2.py", str(pkl_path),
         "--out-dir", processed_subdir],
        cwd=str(REPO_ROOT),
        capture_output=True,
    )
    if ret.returncode != 0:
        return f"[FAIL convert] {stem}: {ret.stderr[-200:] if ret.stderr else ''}"

    return f"[ok] {stem}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--downsample",
        type=int,
        default=8,
        help="Tick stride passed to demo_full_extract (8=8Hz, 32=2Hz).",
    )
    parser.add_argument(
        "--viz-subdir",
        default="viz",
        help="Data-root-relative dir for *_full.pkl outputs.",
    )
    parser.add_argument(
        "--processed-subdir",
        default="processed_v2",
        help="Data-root-relative dir for *.parquet outputs.",
    )
    args = parser.parse_args()

    raw_dir = data_path("raw", "demos")
    demos = sorted(raw_dir.glob("*.dem"))
    print(
        f"Found {len(demos)} demos, workers={args.workers}, "
        f"downsample={args.downsample}, viz={args.viz_subdir}, "
        f"processed={args.processed_subdir}\n"
    )

    python = sys.executable
    ok = 0
    fail = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one,
                dem,
                python,
                args.resume,
                args.downsample,
                args.viz_subdir,
                args.processed_subdir,
            ): dem
            for dem in demos
        }
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            print(f"[{i}/{len(demos)}] {result}")
            if result.startswith("[ok]") or result.startswith("[skip]"):
                ok += 1
            else:
                fail += 1

    print(f"\nDone. ok={ok} fail={fail} total={len(demos)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
