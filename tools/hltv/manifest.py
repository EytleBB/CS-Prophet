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
