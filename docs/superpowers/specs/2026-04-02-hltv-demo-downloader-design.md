# HLTV Demo Downloader — Design Spec
**Date:** 2026-04-02  
**Status:** Approved

---

## Goal

Download 500+ CS2 professional match demo files from HLTV for training the CS-Prophet unified model. Capture structured metadata (teams, players, roles) alongside each demo to support team-style analysis in future phases.

---

## Scope & Constraints

- **Target volume:** 600 demos (with buffer above 500 minimum)
- **Date range:** Last 365 days only (configurable via `cutoff_days`)
- **Maps:** All 7 supported maps (de_mirage, de_inferno, de_dust2, de_nuke, de_ancient, de_overpass, de_anubis) — no per-map quota; download until total target is reached
- **No map version filtering** — all demos within the date window are valid
- **Model target:** Unified single model across all maps (map identity added as 7-dim one-hot in feature vector)

---

## Files

```
tools/
├── download_demos.py      ← main script
└── hltv_config.yaml       ← scraping parameters
data/raw/
├── demos/                 ← .dem files
├── manifest.json          ← per-demo metadata
├── player_cache.json      ← cached player role lookups
└── failed.json            ← failed downloads log
```

---

## Architecture & Data Flow

```
HLTV /results (filtered by event tier + date range)
    ↓ Priority queue: Major → S-tier → A-tier → B-tier
    ↓ Stop when total demos >= target OR page dates < cutoff
    ↓
Per match page /matches/<id>/...
    ↓ Extract: demo download URL, team names, event name, event tier, date, map
    ↓
Download .rar / .zip → extract .dem → save to data/raw/demos/<match_id>_<map>.dem
    ↓
Team page /team/<id>/... → current 5-player roster
    ↓
Player page /player/<id>/... → role (cached in player_cache.json)
    ↓
Append record to data/raw/manifest.json
```

---

## Event Tier Priority

Scrape in this order, stopping early once the total demo target is met:

| Priority | Tier   | Notes                          |
|----------|--------|--------------------------------|
| 1        | Major  | Highest quality, last 365 days |
| 2        | S-tier | Top tournaments outside Majors |
| 3        | A-tier | Regional/mid-tier tournaments  |
| 4        | B-tier | Fill to reach target if needed |

---

## manifest.json Schema

One JSON object per line (newline-delimited JSON):

```json
{
  "match_id": "2380145",
  "demo_file": "2380145_de_mirage.dem",
  "map": "de_mirage",
  "date": "2026-03-15",
  "event": "PGL Major Copenhagen 2026",
  "event_tier": "Major",
  "team_ct": "Natus Vincere",
  "team_t": "FaZe Clan",
  "players": [
    {"name": "s1mple",  "team": "Natus Vincere", "side": "CT", "role": "AWPer"},
    {"name": "karrigan", "team": "FaZe Clan",    "side": "T",  "role": "IGL"}
  ]
}
```

**Field notes:**
- `role` is scraped from HLTV player profile; may be `null` if not listed
- `event_tier` stored for metadata / future use, not used as sample weight
- `side` (CT/T) reflects the team's side in this specific map

---

## Rate Limiting & Politeness

| Mechanism | Value |
|-----------|-------|
| Per-request delay | random 2–5 s |
| Pause every N downloads | 30 s every 10 demos |
| Max retries on 429/503 | 3, with exponential backoff |
| Player profile cache | `player_cache.json` — never re-fetches the same player |
| Resume support | Skip `match_id` already in `manifest.json` |

---

## hltv_config.yaml Parameters

```yaml
target_demos: 600
cutoff_days: 365
event_tiers: [Major, S-tier, A-tier, B-tier]
maps:
  - de_mirage
  - de_inferno
  - de_dust2
  - de_nuke
  - de_ancient
  - de_overpass
  - de_anubis
rate_limit:
  min_delay: 2
  max_delay: 5
  pause_every: 10
  pause_duration: 30
  max_retries: 3
output:
  demos_dir: data/raw/demos
  manifest: data/raw/manifest.json
  player_cache: data/raw/player_cache.json
  failed_log: data/raw/failed.json
```

---

## Future Integration Points

| Data field | Used in |
|------------|---------|
| `map` | 7-dim map identity one-hot added to `state_vector.py` (Phase 3) |
| `players[].role` | Player role embedding in team-style analysis (Phase 4) |
| `team_ct / team_t` | Team identity embedding lookup key (Phase 4) |
| `event_tier` | Metadata only — stored for filtering/analysis, not training weight |

---

## Out of Scope

- Map version / patch date filtering (explicitly excluded)
- Per-map demo quotas (total volume only)
- Sample weighting by event tier
- Player role inference from demo data (Phase 4)
