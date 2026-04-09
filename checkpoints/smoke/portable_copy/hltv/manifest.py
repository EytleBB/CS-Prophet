import json
import os


def load_seen_match_ids(manifest_path: str) -> set:
    if not os.path.exists(manifest_path):
        return set()
    seen = set()
    with open(manifest_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip().strip('\x00')
            if line:
                seen.add(json.loads(line)["match_id"])
    return seen


def append_record(manifest_path: str, record: dict) -> None:
    with open(manifest_path, "a", encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_failure(failed_path: str, match_id: str, reason: str) -> None:
    with open(failed_path, "a", encoding='utf-8') as f:
        f.write(json.dumps({"match_id": match_id, "reason": reason}) + "\n")
