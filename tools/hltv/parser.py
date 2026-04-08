from __future__ import annotations

from datetime import datetime, timezone

from bs4 import BeautifulSoup

KNOWN_MAPS = {
    "de_mirage", "de_inferno", "de_dust2",
    "de_nuke", "de_ancient", "de_overpass", "de_anubis",
}

_MAP_DISPLAY_TO_KEY = {m.replace("de_", ""): m for m in KNOWN_MAPS}


def parse_results_page(html: str) -> list[dict]:
    """
    Parse HLTV /results page HTML.
    Returns list of {"match_id", "url", "event", "date" (YYYY-MM-DD)}.
    Update CSS selectors here if HLTV changes their HTML.
    """
    soup = BeautifulSoup(html, "lxml")
    matches = []
    for con in soup.select("div.result-con"):
        link = con.select_one("a[href*='/matches/']")
        if not link:
            continue
        href = link["href"]
        parts = href.strip("/").split("/")
        if len(parts) < 2:
            continue
        match_id = parts[1]

        event_tag = con.select_one(".event-name")
        event = event_tag.get_text(strip=True) if event_tag else ""

        # HLTV stores the timestamp in data-zonedgrouping-entry-unix on the result-con
        # div itself, or in data-unix on a child element (older layout).
        unix_ms = con.get("data-zonedgrouping-entry-unix") or (
            con.find(attrs={"data-unix": True}) or {}
        ).get("data-unix")
        date_str = ""
        if unix_ms:
            dt = datetime.fromtimestamp(int(unix_ms) / 1000, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")

        matches.append({"match_id": match_id, "url": href, "event": event, "date": date_str})
    return matches


def parse_match_page(html: str) -> dict | None:
    """
    Parse HLTV match page HTML.
    Returns {"demo_url", "team_ct", "team_t"} or None if no demo link.
    First lineup box = CT side.
    """
    soup = BeautifulSoup(html, "lxml")

    demo_link = soup.select_one("a[href*='/download/demo/']")
    if not demo_link:
        return None
    demo_url = demo_link["href"]

    lineups = soup.select("div.lineup")
    team_ct, team_t = "", ""

    for i, lineup in enumerate(lineups[:2]):
        team_name_tag = lineup.select_one(".teamName, .team-name")
        team_name = team_name_tag.get_text(strip=True) if team_name_tag else ""
        if i == 0:
            team_ct = team_name
        else:
            team_t = team_name

    return {"demo_url": demo_url, "team_ct": team_ct, "team_t": team_t}


def get_map_from_dem_filename(filename: str) -> str | None:
    """
    Infer CS2 map name from a .dem filename.
    "navi-vs-faze-m1-de_mirage.dem" → "de_mirage"
    "match-inferno.dem" → "de_inferno"
    "match-vertigo.dem" → None
    """
    stem = filename.lower().replace(".dem", "")
    for part in stem.split("-"):
        if part in KNOWN_MAPS:
            return part
    for part in stem.split("-"):
        if part in _MAP_DISPLAY_TO_KEY:
            return _MAP_DISPLAY_TO_KEY[part]
    return None
