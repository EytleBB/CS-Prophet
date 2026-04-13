#!/usr/bin/env python3
"""
HLTV Demo Pipeline — download, parse to parquet, delete .dem in one shot.

Usage:
    python tools/pipeline.py
    python tools/pipeline.py --config tools/hltv_config.yaml
    python tools/pipeline.py --dry-run

Same config and CLI as download_demos.py. Each demo is:
  1. Downloaded & extracted to a temp dir
  2. Parsed to parquet → data/processed/
  3. .dem deleted immediately

No .dem files accumulate on disk.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.hltv.manifest import log_failure
from tools.hltv.parser import get_map_from_dem_filename, parse_match_page, parse_results_page
from tools.hltv.scraper import HLTVScraper
from tools.hltv.downloader import extract_archive
from src.parser.demo_parser import parse_demo


def _seen_match_ids_from_parquets(processed_dir: Path) -> set[str]:
    """Extract match_ids from existing parquet filenames (e.g. '2389251_de_mirage.parquet')."""
    seen = set()
    for p in processed_dir.glob("*.parquet"):
        parts = p.stem.split("_", 1)
        if parts[0].isdigit():
            seen.add(parts[0])
    return seen


def load_config(path: str) -> dict:
    path = os.path.abspath(path)
    cfg_dir = os.path.dirname(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = cfg.setdefault("output", {})
    for key in ("demos_dir", "failed_log"):
        if key in out and not os.path.isabs(out[key]):
            out[key] = os.path.normpath(os.path.join(cfg_dir, out[key]))
    return cfg


def build_results_url(offset: int) -> str:
    return f"https://www.hltv.org/results?offset={offset}"


def _event_allowed(event_name: str, allowed_events: list[str]) -> bool:
    event_lower = event_name.lower()
    return any(allowed.lower() in event_lower for allowed in allowed_events)


def _already_processed(match_id: str, map_name: str, processed_dir: Path) -> bool:
    """Check if this match+map combo already has a parquet."""
    return (processed_dir / f"{match_id}_{map_name}.parquet").exists()


def run(cfg: dict, dry_run: bool = False) -> None:
    rate_cfg = cfg["rate_limit"]
    out_cfg = cfg["output"]
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    scraper = HLTVScraper(rate_cfg)
    seen = _seen_match_ids_from_parquets(processed_dir)
    supported_maps = set(cfg["maps"])
    allowed_events = cfg["allowed_events"]

    cutoff_date = datetime.now(tz=timezone.utc) - timedelta(days=cfg["cutoff_days"])
    start_date = cutoff_date.strftime("%Y-%m-%d")

    existing_parquets = len(list(processed_dir.glob("*.parquet")))
    target = cfg["target_demos"]
    downloads_since_pause = 0
    new_downloads = 0
    parsed_count = 0
    skipped_count = 0
    failed_count = 0

    print(f"Pipeline mode: download → parse → delete")
    print(f"Existing parquets: {existing_parquets}  |  New to download: {target}")
    print(f"Already seen matches (from parquets): {len(seen)}")
    print(f"Cutoff date: {start_date}")
    print(f"Allowed events: {'all (no filter)' if not allowed_events else len(allowed_events)}")
    print()

    offset = 0
    consecutive_page_errors = 0
    while new_downloads < target:
        url = build_results_url(offset)
        print(f"Fetching results page offset={offset} ...")
        try:
            html = scraper.get(url)
        except Exception as e:
            consecutive_page_errors += 1
            wait = min(60, 2 ** (consecutive_page_errors + 2))
            print(f"  [error] results page failed ({consecutive_page_errors}): {e}")
            if consecutive_page_errors >= 8:
                print(f"  Giving up after {consecutive_page_errors} consecutive results-page failures.")
                break
            print(f"  Waiting {wait}s then retrying same offset ...")
            time.sleep(wait)
            continue
        consecutive_page_errors = 0

        matches = parse_results_page(html)
        if not matches:
            print("  No more matches on this page.")
            break

        if all(m["date"] and m["date"] < start_date for m in matches if m["date"]):
            print("  All matches on page predate cutoff. Stopping.")
            break

        for match in matches:
            if new_downloads >= target:
                break
            match_id = match["match_id"]
            if match_id in seen:
                continue
            if match.get("date") and match["date"] < start_date:
                continue
            if allowed_events and not _event_allowed(match["event"], allowed_events):
                continue

            print(f"  Match {match_id} ({match['event']}) ...")
            if dry_run:
                print(f"    [dry-run] would download & parse match {match_id}")
                continue

            try:
                match_html = scraper.get(match["url"])
            except Exception as e:
                log_failure(out_cfg["failed_log"], match_id, f"match page: {e}")
                failed_count += 1
                continue

            match_data = parse_match_page(match_html)
            if not match_data:
                print(f"    [skip] no demo link for {match_id}")
                log_failure(out_cfg["failed_log"], match_id, "no demo link")
                skipped_count += 1
                continue

            with tempfile.TemporaryDirectory() as tmp:
                demo_url = match_data["demo_url"]
                archive_ext = ".zip" if demo_url.lower().endswith(".zip") else ".rar"
                archive_path = os.path.join(tmp, f"{match_id}{archive_ext}")
                match_url = scraper.BASE + match["url"]
                try:
                    scraper.download_file(demo_url, archive_path, referer=match_url)
                    dem_paths = extract_archive(archive_path, tmp)
                except Exception as e:
                    print(f"    [fail] download/extract: {e}")
                    log_failure(out_cfg["failed_log"], match_id, f"download/extract: {e}")
                    failed_count += 1
                    continue

                if not dem_paths:
                    print(f"    [fail] no .dem files in archive")
                    log_failure(out_cfg["failed_log"], match_id, "no .dem files in archive")
                    failed_count += 1
                    continue

                # Parse each .dem → parquet, then it's auto-deleted with tempdir
                match_parsed = 0
                for dem_path in dem_paths:
                    dem_filename = os.path.basename(dem_path)
                    map_name = get_map_from_dem_filename(dem_filename)
                    if map_name not in supported_maps:
                        print(f"    [skip] {dem_filename}: unsupported map")
                        continue

                    if _already_processed(match_id, map_name, processed_dir):
                        print(f"    [skip] {match_id}_{map_name}: already parsed")
                        continue

                    # Rename .dem so parquet gets the right name
                    renamed = Path(tmp) / f"{match_id}_{map_name}.dem"
                    shutil.move(dem_path, renamed)

                    try:
                        result = parse_demo(renamed, processed_dir)
                        if result:
                            size_kb = result.stat().st_size / 1024
                            print(f"    + {result.name} ({size_kb:.0f} KB)")
                            match_parsed += 1
                            parsed_count += 1
                        else:
                            print(f"    [skip] {match_id}_{map_name}: no valid rounds")
                            skipped_count += 1
                    except Exception as e:
                        print(f"    [fail] {match_id}_{map_name}: {str(e)[:100]}")
                        failed_count += 1

                seen.add(match_id)
                new_downloads += 1

                if match_parsed > 0:
                    downloads_since_pause += 1
                    if downloads_since_pause >= rate_cfg["pause_every"]:
                        print(f"  Pausing {rate_cfg['pause_duration']}s ...")
                        time.sleep(rate_cfg["pause_duration"])
                        downloads_since_pause = 0

            # tempdir cleaned up — .dem files gone

        offset += 100

    print(f"\nDone. Parsed: {parsed_count}  Skipped: {skipped_count}  Failed: {failed_count}")
    total_parquets = len(list(processed_dir.glob("*.parquet")))
    total_mb = sum(p.stat().st_size for p in processed_dir.glob("*.parquet")) / 1024 / 1024
    print(f"Total parquets: {total_parquets}  ({total_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="HLTV demo pipeline: download → parse → cleanup")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hltv_config.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be downloaded without downloading")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
