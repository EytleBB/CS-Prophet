#!/usr/bin/env python3
"""Batch-parse all .dem files in raw/demos/ → processed/.

Usage (from project root):
    python parse_demos.py
    python parse_demos.py --demo-dir raw/demos --out-dir processed
    python parse_demos.py --workers 4       # parallel with 4 processes
    python parse_demos.py --resume          # skip already-processed demos
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

from src.utils.paths import resolve_path_input

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_one(args: tuple[Path, Path]) -> tuple[str, bool, str]:
    """Worker function: parse one demo. Returns (stem, success, error_msg)."""
    dem_path, out_dir = args
    try:
        from src.parser.demo_parser import parse_demo
        result = parse_demo(dem_path, out_dir)
        return dem_path.stem, result is not None, ""
    except Exception as exc:
        return dem_path.stem, False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-parse CS2 demos to parquet.")
    parser.add_argument("--demo-dir",  default="raw/demos",  help="Directory with .dem files")
    parser.add_argument("--out-dir",   default="processed",  help="Output directory for parquets")
    parser.add_argument("--workers",   type=int, default=1,        help="Parallel worker processes (default: 1)")
    parser.add_argument("--resume",    action="store_true",        help="Skip demos already parsed")
    args = parser.parse_args()

    demo_dir = resolve_path_input(args.demo_dir)
    out_dir = resolve_path_input(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_demos = sorted(demo_dir.glob("*.dem"))
    if not all_demos:
        logger.error("No .dem files found in %s", demo_dir)
        sys.exit(1)

    # Resume: skip demos whose parquet already exists
    if args.resume:
        already = {p.stem for p in out_dir.glob("*.parquet")}
        todo = [d for d in all_demos if d.stem not in already]
        logger.info("Resume mode: %d / %d demos already processed, %d remaining",
                    len(already), len(all_demos), len(todo))
    else:
        todo = all_demos

    if not todo:
        logger.info("Nothing to do.")
        return

    logger.info("Parsing %d demos  |  workers=%d  |  output → %s",
                len(todo), args.workers, out_dir)

    work = [(d, out_dir) for d in todo]
    ok = 0
    skipped = 0   # no valid rounds (no A/B plants)
    failed = 0
    t0 = time.time()

    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for i, (stem, success, err) in enumerate(pool.imap_unordered(_parse_one, work), 1):
                _log_result(i, len(todo), stem, success, err, t0)
                if success:   ok      += 1
                elif not err: skipped += 1
                else:         failed  += 1
    else:
        from src.parser.demo_parser import parse_demo
        from tqdm import tqdm
        for i, (dem_path, _) in enumerate(tqdm(work, desc="Parsing", unit="demo"), 1):
            stem, success, err = _parse_one((dem_path, out_dir))
            _log_result(i, len(todo), stem, success, err, t0)
            if success:   ok      += 1
            elif not err: skipped += 1
            else:         failed  += 1

    elapsed = time.time() - t0
    logger.info(
        "Done in %.1f min  |  parsed=%d  skipped(no plants)=%d  failed=%d  |  total=%d",
        elapsed / 60, ok, skipped, failed, len(todo),
    )

    parquets = list(out_dir.glob("*.parquet"))
    total_mb = sum(p.stat().st_size for p in parquets) / 1024 / 1024
    logger.info("Output: %d parquet files  |  %.1f MB total", len(parquets), total_mb)


def _log_result(i: int, total: int, stem: str, success: bool, err: str, t0: float) -> None:
    elapsed = time.time() - t0
    rate = i / elapsed if elapsed > 0 else 0
    eta  = (total - i) / rate if rate > 0 else 0
    status = "OK" if success else ("skip" if not err else "FAIL")
    if err:
        logger.warning("[%d/%d] %s  %s  — %s", i, total, status, stem, err[:120])
    else:
        logger.debug("[%d/%d] %s  %s  ETA %.0fs", i, total, status, stem, eta)


if __name__ == "__main__":
    main()
