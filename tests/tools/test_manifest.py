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
