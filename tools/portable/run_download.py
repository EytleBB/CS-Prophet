#!/usr/bin/env python3
"""
Portable HLTV Demo Downloader — download .dem files only.

Usage (from USB or any machine):
    python run_download.py
    python run_download.py --config hltv_config.yaml
    python run_download.py --dry-run

Downloads .dem files to data/demos/ with manifest-based resume.
Transfer the whole portable/ folder to your project machine afterwards.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import yaml

from hltv.manifest import append_record, load_seen_demos, log_failure
from hltv.parser import get_map_from_dem_filename, parse_match_page, parse_results_page
from hltv.scraper import HLTVScraper
from hltv.downloader import extract_archive


def load_config(path: str) -> dict:
    path = os.path.abspath(path)
    cfg_dir = os.path.dirname(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = cfg.setdefault("output", {})
    for key in ("demos_dir", "processed_dir", "manifest", "failed_log"):
        if key in out and not os.path.isabs(out[key]):
            out[key] = os.path.normpath(os.path.join(cfg_dir, out[key]))
    return cfg


def build_results_url(offset: int) -> str:
    return f"https://www.hltv.org/results?offset={offset}"


def _event_allowed(event_name: str, allowed_events: list[str]) -> bool:
    event_lower = event_name.lower()
    return any(allowed.lower() in event_lower for allowed in allowed_events)


def run(cfg: dict, dry_run: bool = False) -> None:
    rate_cfg = cfg["rate_limit"]
    out_cfg = cfg["output"]
    os.makedirs(out_cfg["demos_dir"], exist_ok=True)

    scraper = HLTVScraper(rate_cfg)
    # Track at (match_id, map_name) level so partial BO3/BO5 can resume
    seen_demos = load_seen_demos(out_cfg["manifest"])  # set of "matchid_mapname"
    supported_maps = set(cfg["maps"])
    allowed_events = cfg.get("allowed_events") or []
    # visited_matches: only tracks matches visited THIS run to avoid duplicate requests
    visited_matches: set[str] = set()

    # Also scan existing .dem files for resume
    demos_dir = out_cfg["demos_dir"]
    if os.path.isdir(demos_dir):
        for f in os.listdir(demos_dir):
            if f.endswith(".dem"):
                seen_demos.add(f.replace(".dem", ""))

    cutoff_date = datetime.now(tz=timezone.utc) - timedelta(days=cfg["cutoff_days"])
    start_date = cutoff_date.strftime("%Y-%m-%d")

    total_downloaded = len(seen_demos)
    target = cfg["target_demos"]
    downloads_since_pause = 0

    print(f"=== Portable HLTV Downloader (download-only) ===")
    print(f"Resuming from {total_downloaded} existing demos (maps). Target: {target}")
    print(f"Cutoff date: {start_date}")
    print(f"Allowed events: {'all (no filter)' if not allowed_events else len(allowed_events)}")
    print(f"Output: {demos_dir}")
    print()

    offset = 0
    while total_downloaded < target:
        url = build_results_url(offset)
        print(f"Fetching results page offset={offset} ...")
        try:
            html = scraper.get(url)
        except Exception as e:
            print(f"  [error] results page failed: {e}")
            break

        matches = parse_results_page(html)
        if not matches:
            print("  No more matches on this page.")
            break

        if all(m["date"] and m["date"] < start_date for m in matches if m["date"]):
            print("  All matches on page predate cutoff. Stopping.")
            break

        for match in matches:
            if total_downloaded >= target:
                break
            match_id = match["match_id"]
            if match.get("date") and match["date"] < start_date:
                continue
            if allowed_events and not _event_allowed(match["event"], allowed_events):
                continue

            # Skip if already visited this match in the current run
            if match_id in visited_matches:
                continue

            print(f"  Match {match_id} ({match['event']}) ...")
            if dry_run:
                print(f"    [dry-run] would download match {match_id}")
                visited_matches.add(match_id)
                continue

            try:
                match_html = scraper.get(match["url"])
            except Exception as e:
                log_failure(out_cfg["failed_log"], match_id, f"match page: {e}")
                continue

            match_data = parse_match_page(match_html)
            if not match_data:
                print(f"    [skip] no demo link for {match_id}")
                log_failure(out_cfg["failed_log"], match_id, "no demo link")
                visited_matches.add(match_id)
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
                    log_failure(out_cfg["failed_log"], match_id, f"download/extract: {e}")
                    continue

                if not dem_paths:
                    log_failure(out_cfg["failed_log"], match_id, "no .dem files in archive")
                    continue

                for dem_path in dem_paths:
                    dem_filename = os.path.basename(dem_path)
                    map_name = get_map_from_dem_filename(dem_filename)
                    if map_name not in supported_maps:
                        print(f"    [skip] {dem_filename}: unsupported map")
                        continue

                    demo_key = f"{match_id}_{map_name}"
                    if demo_key in seen_demos:
                        print(f"    [skip] {demo_key}: already downloaded")
                        continue

                    dest = os.path.join(out_cfg["demos_dir"], f"{demo_key}.dem")
                    shutil.move(dem_path, dest)

                    record = {
                        "match_id": match_id,
                        "demo_file": os.path.basename(dest),
                        "map": map_name,
                        "date": match["date"],
                        "event": match["event"],
                        "team_ct": match_data["team_ct"],
                        "team_t": match_data["team_t"],
                    }
                    append_record(out_cfg["manifest"], record)
                    seen_demos.add(demo_key)
                    total_downloaded += 1
                    downloads_since_pause += 1
                    print(f"    + {demo_key}.dem  [{total_downloaded}/{target}]")

                    if downloads_since_pause >= rate_cfg["pause_every"]:
                        print(f"  Pausing {rate_cfg['pause_duration']}s ...")
                        time.sleep(rate_cfg["pause_duration"])
                        downloads_since_pause = 0

                visited_matches.add(match_id)

        offset += 100

    print(f"\nDone. Total demos: {total_downloaded}")
    demos_dir = out_cfg["demos_dir"]
    dem_count = len([f for f in os.listdir(demos_dir) if f.endswith(".dem")]) if os.path.isdir(demos_dir) else 0
    print(f"Files in {demos_dir}: {dem_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable HLTV demo downloader")
    parser.add_argument(
        "--config",
        default=os.path.join(SCRIPT_DIR, "hltv_config.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
