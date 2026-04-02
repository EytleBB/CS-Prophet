# HLTV Demo Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a scraper that downloads 600 CS2 professional match demos from HLTV with team/player metadata, prioritising Major → S-tier → A-tier → B-tier events from the last 365 days.

**Architecture:** Four focused modules (`manifest`, `parser`, `scraper`, `downloader`) under `tools/hltv/`, wired together by a CLI entry point `tools/download_demos.py`. Each module has one responsibility and is independently testable. The parser module depends on HTML fixture strings in tests, not live HTTP.

**Tech Stack:** `cloudscraper` (Cloudflare bypass), `beautifulsoup4` + `lxml` (HTML parsing), `rarfile` (demo archive extraction), `pyyaml` (config), `pytest` (tests).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `requirements.txt` | Add new scraping deps |
| Create | `tools/hltv_config.yaml` | All tunable parameters |
| Create | `tools/__init__.py` | Package marker |
| Create | `tools/hltv/__init__.py` | Package marker |
| Create | `tools/hltv/manifest.py` | Read/write manifest.json, player_cache.json, failed.json |
| Create | `tools/hltv/parser.py` | BeautifulSoup HTML parsing — results page, match page, player page |
| Create | `tools/hltv/scraper.py` | Rate-limited cloudscraper wrapper with retry |
| Create | `tools/hltv/downloader.py` | Download archive, extract .dem, detect map from filename |
| Create | `tools/download_demos.py` | Orchestrator / CLI entry point |
| Create | `tests/tools/__init__.py` | Package marker |
| Create | `tests/tools/test_manifest.py` | Manifest I/O tests |
| Create | `tests/tools/test_parser.py` | Parser tests with HTML fixtures |
| Create | `tests/tools/test_downloader.py` | Extraction + map-detection tests |

> **Note on HLTV selectors:** CSS selectors in `parser.py` are derived from HLTV's known page structure. Run `python -c "from tools.hltv.scraper import HLTVScraper; s=HLTVScraper({}); print(s.get('https://www.hltv.org/results')[:3000])"` after Task 4 to verify selectors against live HTML; adjust class names in `parser.py` if needed. No other file needs to change.

---

## Task 1: Dependencies + Directory Structure + Config

**Files:**
- Modify: `requirements.txt`
- Create: `tools/__init__.py`, `tools/hltv/__init__.py`, `tests/tools/__init__.py`
- Create: `tools/hltv_config.yaml`

- [ ] **Step 1: Add scraping dependencies to requirements.txt**

Append to the end of `requirements.txt`:

```
# ── Demo downloading ───────────────────────────────────────────────────────
cloudscraper>=1.2.71
beautifulsoup4>=4.12.0
lxml>=5.2.0
rarfile>=4.1
```

- [ ] **Step 2: Install new dependencies**

```bash
pip install cloudscraper beautifulsoup4 lxml rarfile
```

Expected: installs without errors.

- [ ] **Step 3: Create package markers**

Create `tools/__init__.py` (empty):
```python
```

Create `tools/hltv/__init__.py` (empty):
```python
```

Create `tests/tools/__init__.py` (empty):
```python
```

- [ ] **Step 4: Create hltv_config.yaml**

Create `tools/hltv_config.yaml`:

```yaml
target_demos: 600
cutoff_days: 365

event_tiers:
  - Major
  - S-tier
  - A-tier
  - B-tier

# HLTV eventType query param values for each tier
event_type_params:
  Major: MAJOR
  S-tier: S_TIER
  A-tier: A_TIER
  B-tier: B_TIER

maps:
  - de_mirage
  - de_inferno
  - de_dust2
  - de_nuke
  - de_ancient
  - de_overpass
  - de_anubis

rate_limit:
  min_delay: 2      # seconds between requests
  max_delay: 5
  pause_every: 10   # pause after every N demo downloads
  pause_duration: 30
  max_retries: 3

output:
  demos_dir: data/raw/demos
  manifest: data/raw/manifest.json
  player_cache: data/raw/player_cache.json
  failed_log: data/raw/failed.json
```

- [ ] **Step 5: Verify structure**

```bash
find tools tests/tools -type f | sort
```

Expected output (order may vary):
```
tests/tools/__init__.py
tools/__init__.py
tools/hltv/__init__.py
tools/hltv_config.yaml
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tools/ tests/tools/__init__.py
git commit -m "chore: add hltv downloader scaffold and config"
```

---

## Task 2: Manifest Module

**Files:**
- Create: `tools/hltv/manifest.py`
- Create: `tests/tools/test_manifest.py`

- [ ] **Step 1: Write failing tests**

Create `tests/tools/test_manifest.py`:

```python
import json
import os
import pytest
from tools.hltv.manifest import (
    load_seen_match_ids,
    append_record,
    load_player_cache,
    save_player_cache,
    log_failure,
)


@pytest.fixture
def tmp_manifest(tmp_path):
    return str(tmp_path / "manifest.json")


@pytest.fixture
def tmp_cache(tmp_path):
    return str(tmp_path / "player_cache.json")


@pytest.fixture
def tmp_failed(tmp_path):
    return str(tmp_path / "failed.json")


def test_load_seen_match_ids_empty(tmp_manifest):
    assert load_seen_match_ids(tmp_manifest) == set()


def test_append_record_and_reload(tmp_manifest):
    record = {
        "match_id": "123",
        "demo_file": "123_de_mirage.dem",
        "map": "de_mirage",
        "date": "2026-01-01",
        "event": "Test Event",
        "event_tier": "Major",
        "team_ct": "Team A",
        "team_t": "Team B",
        "players": [],
    }
    append_record(tmp_manifest, record)
    seen = load_seen_match_ids(tmp_manifest)
    assert "123" in seen


def test_append_multiple_records(tmp_manifest):
    for i in range(3):
        append_record(tmp_manifest, {"match_id": str(i), "demo_file": f"{i}.dem",
                                     "map": "de_mirage", "date": "2026-01-01",
                                     "event": "E", "event_tier": "Major",
                                     "team_ct": "A", "team_t": "B", "players": []})
    seen = load_seen_match_ids(tmp_manifest)
    assert seen == {"0", "1", "2"}


def test_player_cache_roundtrip(tmp_cache):
    assert load_player_cache(tmp_cache) == {}
    cache = {"7998": "AWPer", "10394": "IGL"}
    save_player_cache(tmp_cache, cache)
    loaded = load_player_cache(tmp_cache)
    assert loaded == cache


def test_log_failure(tmp_failed):
    log_failure(tmp_failed, "999", "HTTP 429")
    with open(tmp_failed) as f:
        entry = json.loads(f.readline())
    assert entry["match_id"] == "999"
    assert entry["reason"] == "HTTP 429"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/tools/test_manifest.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` for `tools.hltv.manifest`.

- [ ] **Step 3: Implement manifest.py**

Create `tools/hltv/manifest.py`:

```python
import json
import os


def load_seen_match_ids(manifest_path: str) -> set:
    """Return set of match_ids already recorded in the manifest."""
    if not os.path.exists(manifest_path):
        return set()
    seen = set()
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["match_id"])
    return seen


def append_record(manifest_path: str, record: dict) -> None:
    """Append one record (NDJSON line) to the manifest."""
    with open(manifest_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_player_cache(cache_path: str) -> dict:
    """Return {player_id: role} dict from cache file."""
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path) as f:
        return json.load(f)


def save_player_cache(cache_path: str, cache: dict) -> None:
    """Overwrite player cache file with current dict."""
    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def log_failure(failed_path: str, match_id: str, reason: str) -> None:
    """Append one failure entry to the failed log."""
    with open(failed_path, "a") as f:
        f.write(json.dumps({"match_id": match_id, "reason": reason}) + "\n")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/tools/test_manifest.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/hltv/manifest.py tests/tools/test_manifest.py
git commit -m "feat: add manifest I/O module"
```

---

## Task 3: HTML Parser Module

**Files:**
- Create: `tools/hltv/parser.py`
- Create: `tests/tools/test_parser.py`

- [ ] **Step 1: Write failing tests**

Create `tests/tools/test_parser.py`:

```python
from tools.hltv.parser import (
    parse_results_page,
    parse_match_page,
    parse_player_page,
    get_map_from_dem_filename,
)

# ── Fixtures ──────────────────────────────────────────────────────────────

RESULTS_HTML = """
<html><body>
<div class="results-sublist">
  <div class="result-con">
    <a class="a-reset plain" href="/matches/2380145/navi-vs-faze-starladder-budapest-major-2025">
      <div class="result-teamnames text-ellipsis">
        <span class="team-name">Natus Vincere</span>
        <span class="team-name">FaZe Clan</span>
      </div>
      <div class="event-name">StarLadder Budapest Major 2025</div>
      <div class="date-cell"><div data-unix="1733500000000"></div></div>
    </a>
  </div>
  <div class="result-con">
    <a class="a-reset plain" href="/matches/2380200/spirit-vs-vitality-iem-cologne-2025">
      <div class="result-teamnames text-ellipsis">
        <span class="team-name">Team Spirit</span>
        <span class="team-name">Team Vitality</span>
      </div>
      <div class="event-name">IEM Cologne 2025</div>
      <div class="date-unix"><div data-unix="1720000000000"></div></div>
    </a>
  </div>
</div>
</body></html>
"""

MATCH_HTML = """
<html><body>
<div class="standard-box veto-box">
  <div class="mapholder">
    <div class="mapname">Mirage</div>
    <div class="results-center-half-score">
      <span class="ct">CT</span>
      <span class="t">T</span>
    </div>
    <div class="results-left">
      <div class="team">Natus Vincere</div>
    </div>
    <div class="results-right">
      <div class="team">FaZe Clan</div>
    </div>
  </div>
</div>
<div class="streams-and-vods">
  <div class="stream-box">
    <a href="/download/demo/654321">Download Demo</a>
  </div>
</div>
<div class="lineups">
  <div class="lineup standard-box">
    <div class="teamName">Natus Vincere</div>
    <div class="players">
      <div class="player">
        <a href="/player/7998/s1mple">s1mple</a>
      </div>
      <div class="player">
        <a href="/player/8918/electronic">electronic</a>
      </div>
    </div>
  </div>
  <div class="lineup standard-box">
    <div class="teamName">FaZe Clan</div>
    <div class="players">
      <div class="player">
        <a href="/player/10394/karrigan">karrigan</a>
      </div>
      <div class="player">
        <a href="/player/11816/ropz">ropz</a>
      </div>
    </div>
  </div>
</div>
</body></html>
"""

PLAYER_HTML_WITH_ROLE = """
<html><body>
<div class="profile-team-nav">
  <div class="playerRealname">Oleksandr Kostyliev</div>
</div>
<div class="playerpage-leftside">
  <div class="infobox-columns">
    <div class="infobox-columns-cell">
      <div class="cell-title">Role</div>
      <div class="cell-value">AWPer</div>
    </div>
  </div>
</div>
</body></html>
"""

PLAYER_HTML_NO_ROLE = """
<html><body>
<div class="playerpage-leftside">
  <div class="infobox-columns">
    <div class="infobox-columns-cell">
      <div class="cell-title">Age</div>
      <div class="cell-value">27</div>
    </div>
  </div>
</div>
</body></html>
"""

# ── Tests ─────────────────────────────────────────────────────────────────

def test_parse_results_page_returns_matches():
    matches = parse_results_page(RESULTS_HTML)
    assert len(matches) == 2


def test_parse_results_page_match_fields():
    matches = parse_results_page(RESULTS_HTML)
    m = matches[0]
    assert m["match_id"] == "2380145"
    assert "navi-vs-faze" in m["url"]
    assert m["event"] == "StarLadder Budapest Major 2025"
    assert m["date"] == "2024-12-06"  # unix 1733500000 → 2024-12-06


def test_parse_results_page_empty():
    assert parse_results_page("<html><body></body></html>") == []


def test_parse_match_page_demo_url():
    result = parse_match_page(MATCH_HTML)
    assert result is not None
    assert result["demo_url"] == "/download/demo/654321"


def test_parse_match_page_teams():
    result = parse_match_page(MATCH_HTML)
    assert "Natus Vincere" in (result["team_ct"], result["team_t"])
    assert "FaZe Clan" in (result["team_ct"], result["team_t"])


def test_parse_match_page_players():
    result = parse_match_page(MATCH_HTML)
    ids = [p["player_id"] for p in result["players"]]
    assert "7998" in ids
    assert "10394" in ids


def test_parse_match_page_no_demo_returns_none():
    result = parse_match_page("<html><body><p>No demo here</p></body></html>")
    assert result is None


def test_parse_player_page_with_role():
    role = parse_player_page(PLAYER_HTML_WITH_ROLE)
    assert role == "AWPer"


def test_parse_player_page_no_role():
    role = parse_player_page(PLAYER_HTML_NO_ROLE)
    assert role is None


def test_get_map_from_dem_filename_standard():
    assert get_map_from_dem_filename("navi-vs-faze-m1-de_mirage.dem") == "de_mirage"


def test_get_map_from_dem_filename_no_prefix():
    assert get_map_from_dem_filename("match-inferno.dem") == "de_inferno"


def test_get_map_from_dem_filename_unknown():
    assert get_map_from_dem_filename("match-vertigo.dem") is None
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/tools/test_parser.py -v
```

Expected: `ImportError` for `tools.hltv.parser`.

- [ ] **Step 3: Implement parser.py**

Create `tools/hltv/parser.py`:

```python
from __future__ import annotations

import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

KNOWN_MAPS = {
    "de_mirage", "de_inferno", "de_dust2",
    "de_nuke", "de_ancient", "de_overpass", "de_anubis",
}

# HLTV displays map names without "de_" prefix
_MAP_DISPLAY_TO_KEY = {m.replace("de_", ""): m for m in KNOWN_MAPS}
# e.g. {"mirage": "de_mirage", "inferno": "de_inferno", ...}


def parse_results_page(html: str) -> list[dict]:
    """
    Parse HLTV /results page HTML.

    Returns list of dicts:
        {"match_id": str, "url": str, "event": str, "date": str (YYYY-MM-DD)}

    Note: if HLTV changes class names, update selectors here only.
    """
    soup = BeautifulSoup(html, "lxml")
    matches = []
    for con in soup.select("div.result-con"):
        link = con.select_one("a[href*='/matches/']")
        if not link:
            continue
        href = link["href"]
        # href format: /matches/<match_id>/<slug>
        parts = href.strip("/").split("/")
        if len(parts) < 2:
            continue
        match_id = parts[1]

        event_tag = con.select_one(".event-name")
        event = event_tag.get_text(strip=True) if event_tag else ""

        # date: look for data-unix attribute anywhere in the container
        unix_tag = con.find(attrs={"data-unix": True})
        date_str = ""
        if unix_tag:
            unix_ms = int(unix_tag["data-unix"])
            dt = datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")

        matches.append({
            "match_id": match_id,
            "url": href,
            "event": event,
            "date": date_str,
        })
    return matches


def parse_match_page(html: str) -> dict | None:
    """
    Parse HLTV match page HTML.

    Returns dict or None if no demo link found:
        {
            "demo_url": str,           # relative URL e.g. /download/demo/12345
            "team_ct": str,
            "team_t": str,
            "players": [{"player_id": str, "name": str, "team": str}],
        }

    team_ct/team_t: first team listed is treated as CT (left side).
    players include all found lineup entries (up to 10 total).
    """
    soup = BeautifulSoup(html, "lxml")

    # Demo download link
    demo_link = soup.select_one("a[href*='/download/demo/']")
    if not demo_link:
        return None
    demo_url = demo_link["href"]

    # Teams — from lineup boxes (first box = CT side)
    lineups = soup.select("div.lineup")
    team_ct, team_t = "", ""
    players = []

    for i, lineup in enumerate(lineups[:2]):
        # team name may be in a heading or team-name span
        team_name_tag = lineup.select_one(".teamName, .team-name")
        team_name = team_name_tag.get_text(strip=True) if team_name_tag else ""
        side = "CT" if i == 0 else "T"
        if i == 0:
            team_ct = team_name
        else:
            team_t = team_name

        for player_link in lineup.select("a[href*='/player/']"):
            href = player_link["href"]
            # href format: /player/<player_id>/<name>
            parts = href.strip("/").split("/")
            if len(parts) >= 2:
                player_id = parts[1]
                name = player_link.get_text(strip=True)
                players.append({
                    "player_id": player_id,
                    "name": name,
                    "team": team_name,
                    "side": side,
                })

    return {
        "demo_url": demo_url,
        "team_ct": team_ct,
        "team_t": team_t,
        "players": players,
    }


def parse_player_page(html: str) -> str | None:
    """
    Parse HLTV player profile page HTML.

    Returns role string (e.g. "AWPer", "IGL") or None if not found.
    """
    soup = BeautifulSoup(html, "lxml")
    for cell in soup.select("div.infobox-columns-cell, div.playerInfoRow"):
        title = cell.select_one(".cell-title, .playerInfoTitle")
        value = cell.select_one(".cell-value, .playerInfoValue")
        if title and value and "role" in title.get_text(strip=True).lower():
            return value.get_text(strip=True)
    return None


def get_map_from_dem_filename(filename: str) -> str | None:
    """
    Infer CS2 map name from a .dem filename.

    Handles:
    - "navi-vs-faze-m1-de_mirage.dem"  → "de_mirage"
    - "match-inferno.dem"              → "de_inferno"
    - "match-vertigo.dem"              → None (not a supported map)
    """
    stem = filename.lower().replace(".dem", "")
    # Try exact de_<map> token first
    for part in stem.split("-"):
        if part in KNOWN_MAPS:
            return part
    # Try display name (without de_ prefix)
    for part in stem.split("-"):
        if part in _MAP_DISPLAY_TO_KEY:
            return _MAP_DISPLAY_TO_KEY[part]
    return None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/tools/test_parser.py -v
```

Expected: all 11 tests PASS.

If any selector-based test fails (test_parse_results_page_match_fields, test_parse_match_page_*), inspect the fixture HTML in the test and make sure the parser selectors match. The fixture HTML in the test is the ground truth — fix the selector in `parser.py`, not the test.

- [ ] **Step 5: Commit**

```bash
git add tools/hltv/parser.py tests/tools/test_parser.py
git commit -m "feat: add HLTV HTML parser module"
```

---

## Task 4: HTTP Scraper Module

**Files:**
- Create: `tools/hltv/scraper.py`

(No unit tests — this module is a thin wrapper around `cloudscraper`. Integration-tested by running the orchestrator against live HLTV in Task 6.)

- [ ] **Step 1: Implement scraper.py**

Create `tools/hltv/scraper.py`:

```python
from __future__ import annotations

import random
import time
from typing import Any

import cloudscraper


class HLTVScraper:
    """
    Rate-limited HTTP client for HLTV pages.

    Usage:
        cfg = {"min_delay": 2, "max_delay": 5, "max_retries": 3}
        s = HLTVScraper(cfg)
        html = s.get("https://www.hltv.org/results")
        s.download_file("https://www.hltv.org/download/demo/123", "/tmp/demo.rar")
    """

    BASE = "https://www.hltv.org"

    def __init__(self, rate_limit_cfg: dict[str, Any]) -> None:
        self._min_delay = rate_limit_cfg.get("min_delay", 2)
        self._max_delay = rate_limit_cfg.get("max_delay", 5)
        self._max_retries = rate_limit_cfg.get("max_retries", 3)
        self._session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )

    def _sleep(self) -> None:
        time.sleep(random.uniform(self._min_delay, self._max_delay))

    def get(self, url: str) -> str:
        """
        Fetch URL and return response text.
        Retries on 429/503 with exponential backoff.
        Raises RuntimeError after max_retries exhausted.
        """
        if not url.startswith("http"):
            url = self.BASE + url
        for attempt in range(self._max_retries):
            self._sleep()
            resp = self._session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 503):
                wait = 2 ** (attempt + 2)  # 4s, 8s, 16s
                print(f"  [{resp.status_code}] backing off {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Failed to fetch {url} after {self._max_retries} retries")

    def download_file(self, url: str, dest_path: str) -> None:
        """
        Stream-download a file (demo archive) to dest_path.
        Retries on transient errors.
        """
        if not url.startswith("http"):
            url = self.BASE + url
        for attempt in range(self._max_retries):
            self._sleep()
            resp = self._session.get(url, stream=True, timeout=60)
            if resp.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return
            if resp.status_code in (429, 503):
                wait = 2 ** (attempt + 2)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"Failed to download {url} after {self._max_retries} retries")
```

- [ ] **Step 2: Smoke test import**

```bash
python -c "from tools.hltv.scraper import HLTVScraper; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add tools/hltv/scraper.py
git commit -m "feat: add rate-limited HLTV scraper"
```

---

## Task 5: Downloader Module

**Files:**
- Create: `tools/hltv/downloader.py`
- Create: `tests/tools/test_downloader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/tools/test_downloader.py`:

```python
import os
import zipfile
import pytest
from tools.hltv.downloader import extract_archive, get_dem_files


@pytest.fixture
def zip_with_two_dems(tmp_path):
    """Create a .zip containing two .dem files."""
    archive = tmp_path / "demo.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("navi-vs-faze-m1-de_mirage.dem", b"FAKE_DEM_CONTENT" * 10)
        zf.writestr("navi-vs-faze-m2-de_inferno.dem", b"FAKE_DEM_CONTENT" * 10)
    return str(archive)


@pytest.fixture
def zip_with_non_dem(tmp_path):
    """Create a .zip with a .dem and a readme."""
    archive = tmp_path / "demo.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("match-de_dust2.dem", b"FAKE")
        zf.writestr("README.txt", b"hello")
    return str(archive)


def test_extract_archive_returns_dem_paths(zip_with_two_dems, tmp_path):
    dest = str(tmp_path / "out")
    os.makedirs(dest)
    paths = extract_archive(zip_with_two_dems, dest)
    assert len(paths) == 2
    assert all(p.endswith(".dem") for p in paths)


def test_extract_archive_filters_non_dem(zip_with_non_dem, tmp_path):
    dest = str(tmp_path / "out")
    os.makedirs(dest)
    paths = extract_archive(zip_with_non_dem, dest)
    assert len(paths) == 1
    assert paths[0].endswith(".dem")


def test_get_dem_files_finds_all(tmp_path):
    (tmp_path / "a.dem").write_bytes(b"x")
    (tmp_path / "b.dem").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    result = get_dem_files(str(tmp_path))
    assert len(result) == 2
    assert all(f.endswith(".dem") for f in result)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/tools/test_downloader.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement downloader.py**

Create `tools/hltv/downloader.py`:

```python
from __future__ import annotations

import os
import zipfile


def extract_archive(archive_path: str, dest_dir: str) -> list[str]:
    """
    Extract .dem files from a .zip or .rar archive into dest_dir.

    Returns list of absolute paths to extracted .dem files.
    Raises ValueError for unsupported archive formats.
    """
    archive_path = os.path.abspath(archive_path)
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    if archive_path.endswith(".zip"):
        return _extract_zip(archive_path, dest_dir)
    elif archive_path.endswith(".rar"):
        return _extract_rar(archive_path, dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def _extract_zip(archive_path: str, dest_dir: str) -> list[str]:
    extracted = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".dem"):
                zf.extract(name, dest_dir)
                extracted.append(os.path.join(dest_dir, name))
    return extracted


def _extract_rar(archive_path: str, dest_dir: str) -> list[str]:
    try:
        import rarfile
    except ImportError:
        raise RuntimeError("rarfile not installed. Run: pip install rarfile")
    extracted = []
    with rarfile.RarFile(archive_path) as rf:
        for name in rf.namelist():
            if name.endswith(".dem"):
                rf.extract(name, dest_dir)
                extracted.append(os.path.join(dest_dir, name))
    return extracted


def get_dem_files(directory: str) -> list[str]:
    """Return list of .dem file paths in directory (non-recursive)."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".dem")
    ]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/tools/test_downloader.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/hltv/downloader.py tests/tools/test_downloader.py
git commit -m "feat: add demo archive extractor"
```

---

## Task 6: Main Orchestrator

**Files:**
- Create: `tools/download_demos.py`

- [ ] **Step 1: Run full test suite to confirm baseline**

```bash
pytest tests/ -v --tb=short
```

Expected: all existing tests + new tests (manifest, parser, downloader) PASS. Note total count.

- [ ] **Step 2: Implement download_demos.py**

Create `tools/download_demos.py`:

```python
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

# Allow running from project root: python tools/download_demos.py
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
    supported_maps: list[str],
) -> list[dict]:
    """Fetch role for each player (using cache). Mutates players in place, returns them."""
    dirty = False
    for p in players:
        pid = p["player_id"]
        if pid not in player_cache:
            try:
                html = scraper.get(f"/player/{pid}/{p['name'].lower().replace(' ', '-')}")
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

            # Stop paging if all matches on page are outside date window
            if all(m["date"] and m["date"] < start_date for m in matches if m["date"]):
                print("  All matches on page predate cutoff. Stopping tier.")
                break

            for match in matches:
                if total_downloaded >= target:
                    break
                match_id = match["match_id"]
                if match_id in seen:
                    print(f"  [skip] {match_id} already in manifest")
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

                # Download archive to temp file
                with tempfile.TemporaryDirectory() as tmp:
                    archive_ext = ".rar"  # HLTV default; fallback handled below
                    demo_url = match_data["demo_url"]
                    if demo_url.endswith(".zip"):
                        archive_ext = ".zip"
                    archive_path = os.path.join(tmp, f"{match_id}{archive_ext}")
                    try:
                        scraper.download_file(demo_url, archive_path)
                        # Re-detect ext from actual content if needed
                        if not os.path.exists(archive_path):
                            raise FileNotFoundError("Archive not saved")
                        dem_paths = extract_archive(archive_path, tmp)
                    except Exception as e:
                        log_failure(out_cfg["failed_log"], match_id, f"download/extract: {e}")
                        continue

                    if not dem_paths:
                        log_failure(out_cfg["failed_log"], match_id, "no .dem files in archive")
                        continue

                    # Resolve player roles
                    players = resolve_player_roles(
                        match_data["players"], scraper, player_cache,
                        out_cfg["player_cache"], list(supported_maps)
                    )

                    # One manifest entry per .dem file
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
                        print(f"    ✓ {match_id}_{map_name}.dem  [{total_downloaded}/{target}]")

                        if downloads_since_pause >= rate_cfg["pause_every"]:
                            print(f"  Pausing {rate_cfg['pause_duration']}s ...")
                            time.sleep(rate_cfg["pause_duration"])
                            downloads_since_pause = 0

            offset += 100  # next results page

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
```

- [ ] **Step 3: Smoke test with dry-run**

```bash
python tools/download_demos.py --dry-run
```

Expected: prints "Resuming from 0 existing demos. Target: 600", starts fetching results page. No files created. Exit with no errors.

If you see an HTTP error, verify `cloudscraper` is installed and HLTV is reachable.

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests pass (count should be 111 + new tests).

- [ ] **Step 5: Commit**

```bash
git add tools/download_demos.py
git commit -m "feat: HLTV demo orchestrator with resume and dry-run support"
```

---

## Task 7: Live Selector Verification

Before a full run, verify the parser works against real HLTV pages.

- [ ] **Step 1: Fetch a live results page and check match count**

```python
# Run in Python REPL from project root
import sys; sys.path.insert(0, ".")
import yaml
from tools.hltv.scraper import HLTVScraper
from tools.hltv.parser import parse_results_page
from datetime import datetime, timedelta, timezone

cfg = yaml.safe_load(open("tools/hltv_config.yaml"))
s = HLTVScraper(cfg["rate_limit"])
cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
url = f"https://www.hltv.org/results?startDate={cutoff}&endDate={today}&eventType=MAJOR&offset=0"
html = s.get(url)
matches = parse_results_page(html)
print(f"Parsed {len(matches)} matches")
for m in matches[:3]:
    print(m)
```

Expected: prints 3+ match dicts with non-empty `match_id`, `event`, `date`.

If `matches` is empty: print `html[:2000]` and inspect the actual class names used in HLTV's HTML. Update `parse_results_page` selectors in `tools/hltv/parser.py` accordingly, then also update the fixture HTML in `tests/tools/test_parser.py` to match. Re-run `pytest tests/tools/test_parser.py -v` to confirm.

- [ ] **Step 2: Fetch a live match page and check demo URL**

```python
# Continuing from Step 1 REPL
match_url = matches[0]["url"]
match_html = s.get(match_url)
from tools.hltv.parser import parse_match_page
data = parse_match_page(match_html)
print(data)
```

Expected: dict with non-None `demo_url`, non-empty `team_ct`/`team_t`, 6–10 players with `player_id`.

If `data is None`: the demo link selector needs updating. Search `match_html` for "download" or "demo" to find the actual link pattern. Update `parse_match_page` in `tools/hltv/parser.py`.

- [ ] **Step 3: Fetch a live player page and check role**

```python
# Continuing from Step 2 REPL
from tools.hltv.parser import parse_player_page
p = data["players"][0]
player_html = s.get(f"/player/{p['player_id']}/{p['name'].lower()}")
role = parse_player_page(player_html)
print(f"Role for {p['name']}: {role}")
```

Expected: a role string like "AWPer", "IGL", "Rifler", or `None` (if not listed on HLTV).

- [ ] **Step 4: Commit any selector fixes**

```bash
git add tools/hltv/parser.py tests/tools/test_parser.py
git commit -m "fix: update parser selectors for live HLTV HTML"
```

(Skip this step if no changes were needed.)

---

## Task 8: Full Download Run

- [ ] **Step 1: Start download (can be interrupted and resumed)**

```bash
python tools/download_demos.py
```

Monitor progress. The script prints each downloaded demo and the running total. It resumes automatically if interrupted — just re-run the same command.

- [ ] **Step 2: Verify output**

```bash
# Count downloaded demos
ls data/raw/demos/*.dem | wc -l

# Inspect first 3 manifest entries
head -3 data/raw/manifest.json | python -m json.tool
```

Expected: `.dem` count growing toward 600; each manifest entry has `match_id`, `map`, `players` with roles.

- [ ] **Step 3: Check failure log**

```bash
cat data/raw/failed.json
```

If there are many failures for the same reason (e.g., "no demo link"), that selector likely needs a fix. Re-run Task 7 Step 2 to diagnose.

- [ ] **Step 4: Final commit**

```bash
git add data/raw/manifest.json data/raw/player_cache.json
git commit -m "data: initial HLTV demo metadata manifest"
```

Note: `.dem` files are not committed (listed in `.gitignore` or too large). Only commit the manifest and player cache.
