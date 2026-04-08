import json
import os


def load_seen_match_ids(manifest_path: str) -> set:
    """Return set of match_ids already recorded in the manifest."""
    if not os.path.exists(manifest_path):
        return set()
    seen = set()
    # 【修改点 1】添加 encoding='utf-8'，防止读取到特殊字符时崩溃
    with open(manifest_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip().strip('\x00')
            if line:
                seen.add(json.loads(line)["match_id"])
    return seen


def append_record(manifest_path: str, record: dict) -> None:
    """Append one record (NDJSON line) to the manifest."""
    # 【修改点 2】添加 encoding='utf-8'，解决之前的写入报错
    with open(manifest_path, "a", encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_failure(failed_path: str, match_id: str, reason: str) -> None:
    """Append one failure entry to the failed log."""
    # 【修改点 3】添加 encoding='utf-8'，防止写入错误日志时包含特殊字符导致崩溃
    with open(failed_path, "a", encoding='utf-8') as f:
        f.write(json.dumps({"match_id": match_id, "reason": reason}) + "\n")