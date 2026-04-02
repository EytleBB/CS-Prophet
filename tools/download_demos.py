#!/usr/bin/env python3
"""
HLTV Demo Downloader
Usage: python tools/download_demos.py [--config tools/hltv_config.yaml] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.hltv.manifest import (
    append_record,
    load_player_cache,
    load_seen_match_ids,
    log_failure,
    save_player_cache,
)
from tools.hltv.parser import (
    get_map_from_dem_filename,
    parse_match_page,
    parse_player_page,
    parse_results_page,
)
from tools.hltv.scraper import HLTVScraper
from tools.hltv.downloader import extract_archive


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_results_url(event_type_param: str, start_date: str, end_date: str, offset: int) -> str:
    return (
        f"https://www.hltv.org/results"
        f"?startDate={start_date}&endDate={end_date}"
        f"&eventType={event_type_param}&offset={offset}"
    )


def resolve_player_roles(
    players: list[dict],
    scraper: HLTVScraper,
    player_cache: dict,
    cache_path: str,
) -> list[dict]:
    """Fetch role for each player (using cache). Adds 'role' key to each player dict."""
    dirty = False
    for p in players:
        pid = p["player_id"]
        if pid not in player_cache:
            try:
                slug = p["name"].lower().replace(" ", "-")
                html = scraper.get(f"/player/{pid}/{slug}")
                player_cache[pid] = parse_player_page(html)
                dirty = True
            except Exception as e:
                print(f"    [warn] player {pid} fetch failed: {e}")
                player_cache[pid] = None
        p["role"] = player_cache[pid]
    if dirty:
        save_player_cache(cache_path, player_cache)
    return players


def run(cfg: dict, dry_run: bool = False) -> None:
    rate_cfg = cfg["rate_limit"]
    out_cfg = cfg["output"]
    os.makedirs(out_cfg["demos_dir"], exist_ok=True)

    scraper = HLTVScraper(rate_cfg)
    seen = load_seen_match_ids(out_cfg["manifest"])
    player_cache = load_player_cache(out_cfg["player_cache"])
    supported_maps = set(cfg["maps"])

    cutoff_date = datetime.now(tz=timezone.utc) - timedelta(days=cfg["cutoff_days"])
    start_date = cutoff_date.strftime("%Y-%m-%d")
    end_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    total_downloaded = len(seen)
    target = cfg["target_demos"]
    downloads_since_pause = 0

    print(f"Resuming from {total_downloaded} existing demos. Target: {target}")
    print(f"Date range: {start_date} to {end_date}")

    for tier in cfg["event_tiers"]:
        if total_downloaded >= target:
            break
        event_type_param = cfg["event_type_params"][tier]
        print(f"\n=== Tier: {tier} ({event_type_param}) ===")
        offset = 0

        while total_downloaded < target:
            url = build_results_url(event_type_param, start_date, end_date, offset)
            print(f"  Fetching results page offset={offset} ...")
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
                print("  All matches on page predate cutoff. Stopping tier.")
                break

            for match in matches:
                if total_downloaded >= target:
                    break
                match_id = match["match_id"]
                if match_id in seen:
                    print(f"  [skip] {match_id} already downloaded")
                    continue
                if match.get("date") and match["date"] < start_date:
                    print(f"  [skip] {match_id} too old ({match['date']})")
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
                    archive_ext = ".zip" if demo_url.endswith(".zip") else ".rar"
                    archive_path = os.path.join(tmp, f"{match_id}{archive_ext}")
                    try:
                        scraper.download_file(demo_url, archive_path)
                        dem_paths = extract_archive(archive_path, tmp)
                    except Exception as e:
                        log_failure(out_cfg["failed_log"], match_id, f"download/extract: {e}")
                        continue

                    if not dem_paths:
                        log_failure(out_cfg["failed_log"], match_id, "no .dem files in archive")
                        continue

                    players = resolve_player_roles(
                        match_data["players"], scraper, player_cache, out_cfg["player_cache"]
                    )

                    for dem_path in dem_paths:
                        dem_filename = os.path.basename(dem_path)
                        map_name = get_map_from_dem_filename(dem_filename)
                        if map_name not in supported_maps:
                            print(f"    [skip] {dem_filename}: unsupported map")
                            continue

                        dest = os.path.join(out_cfg["demos_dir"], f"{match_id}_{map_name}.dem")
                        os.rename(dem_path, dest)

                        record = {
                            "match_id": match_id,
                            "demo_file": os.path.basename(dest),
                            "map": map_name,
                            "date": match["date"],
                            "event": match["event"],
                            "event_tier": tier,
                            "team_ct": match_data["team_ct"],
                            "team_t": match_data["team_t"],
                            "players": [
                                {k: v for k, v in p.items() if k != "player_id"}
                                for p in players
                            ],
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
    parser.add_argument("--config", default="tools/hltv_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be downloaded without downloading")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
