#!/usr/bin/env python3
"""
HLTV Demo Downloader
Usage: python tools/download_demos.py [--config tools/hltv_config.yaml] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

# Force UTF-8 output on Windows (default console uses GBK which can't encode all Unicode)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.hltv.manifest import (
    append_record,
    load_seen_match_ids,
    log_failure,
)
from tools.hltv.parser import (
    get_map_from_dem_filename,
    parse_match_page,
    parse_results_page,
)
from tools.hltv.scraper import HLTVScraper
from tools.hltv.downloader import extract_archive


def load_config(path: str) -> dict:
    path = os.path.abspath(path)
    cfg_dir = os.path.dirname(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = cfg.setdefault("output", {})
    for key in ("demos_dir", "manifest", "failed_log"):
        if key in out and not os.path.isabs(out[key]):
            out[key] = os.path.normpath(os.path.join(cfg_dir, out[key]))
    return cfg


def build_results_url(offset: int) -> str:
    # Note: startDate/endDate/eventType params trigger Cloudflare's stricter bot protection.
    # Fetch unfiltered results and filter client-side by event name.
    return f"https://www.hltv.org/results?offset={offset}"


def _event_allowed(event_name: str, allowed_events: list[str]) -> bool:
    event_lower = event_name.lower()
    return any(allowed.lower() in event_lower for allowed in allowed_events)


def run(cfg: dict, dry_run: bool = False) -> None:
    rate_cfg = cfg["rate_limit"]
    out_cfg = cfg["output"]
    os.makedirs(out_cfg["demos_dir"], exist_ok=True)

    scraper = HLTVScraper(rate_cfg)
    seen = load_seen_match_ids(out_cfg["manifest"])
    supported_maps = set(cfg["maps"])
    allowed_events = cfg["allowed_events"]

    cutoff_date = datetime.now(tz=timezone.utc) - timedelta(days=cfg["cutoff_days"])
    start_date = cutoff_date.strftime("%Y-%m-%d")

    total_downloaded = len(seen)
    target = cfg["target_demos"]
    downloads_since_pause = 0

    print(f"Resuming from {total_downloaded} existing demos. Target: {target}")
    print(f"Cutoff date: {start_date}")
    print(f"Allowed events: {'all (no filter)' if not allowed_events else len(allowed_events)}")

    offset = 0
    while total_downloaded < target:
        url = build_results_url(offset)
        print(f"\nFetching results page offset={offset} ...")
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
            if match_id in seen:
                continue
            if match.get("date") and match["date"] < start_date:
                continue
            if allowed_events and not _event_allowed(match["event"], allowed_events):
                continue

            print(f"  Processing match {match_id} ({match['event']}) ...")
            if dry_run:
                print(f"    [dry-run] would download match {match_id}")
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
                continue

            with tempfile.TemporaryDirectory() as tmp:
                demo_url = match_data["demo_url"]
                archive_ext = ".zip" if demo_url.lower().endswith(".zip") else ".rar"
                archive_path = os.path.join(tmp, f"{match_id}{archive_ext}")
                match_url = scraper.BASE + match["url"]
                try:
                    scraper.download_file(demo_url, archive_path, referer=match_url)
                    arc_size = os.path.getsize(archive_path) if os.path.exists(archive_path) else -1
                    print(f"    [debug] archive size: {arc_size // 1024 // 1024} MB")
                    dem_paths = extract_archive(archive_path, tmp)
                    print(f"    [debug] dem_paths: {dem_paths}")
                except Exception as e:
                    print(f"    [debug] exception: {e}")
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

                    dest = os.path.join(out_cfg["demos_dir"], f"{match_id}_{map_name}.dem")
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
                    seen.add(match_id)
                    total_downloaded += 1
                    downloads_since_pause += 1
                    print(f"    + {match_id}_{map_name}.dem  [{total_downloaded}/{target}]")

                    if downloads_since_pause >= rate_cfg["pause_every"]:
                        print(f"  Pausing {rate_cfg['pause_duration']}s ...")
                        time.sleep(rate_cfg["pause_duration"])
                        downloads_since_pause = 0

        offset += 100

    print(f"\nDone. Total demos: {total_downloaded}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CS2 demos from HLTV")
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
