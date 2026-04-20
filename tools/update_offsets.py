"""Refresh cs2-dumper offset JSON files used by the memory reader."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = REPO_ROOT / "src" / "memory_reader"
BASE_URL = "https://raw.githubusercontent.com/a2x/cs2-dumper/main/output"
FILES = ("offsets.json", "client_dll.json")


def _download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _offset_build_number(data: bytes) -> int | None:
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception:
        return None

    build_number = parsed.get("engine2.dll", {}).get("dwBuildNumber")
    if build_number is None:
        return None
    try:
        return int(build_number)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh memory-reader offsets from cs2-dumper")
    parser.add_argument("--force", action="store_true", help="Rewrite files even when content is unchanged")
    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    old_offsets = (TARGET_DIR / "offsets.json").read_bytes() if (TARGET_DIR / "offsets.json").exists() else b""
    new_offsets = old_offsets

    for file_name in FILES:
        target_path = TARGET_DIR / file_name
        url = f"{BASE_URL}/{file_name}"
        payload = _download_bytes(url)
        if file_name == "offsets.json":
            new_offsets = payload

        if not args.force and target_path.exists() and target_path.read_bytes() == payload:
            print(f"{file_name}: unchanged")
            continue

        _atomic_write(target_path, payload)
        print(f"{file_name}: updated ({len(payload)} bytes)")

    old_build = _offset_build_number(old_offsets)
    new_build = _offset_build_number(new_offsets)
    if old_build is not None and new_build is not None:
        print(f"offsets.json engine2.dll.dwBuildNumber: {old_build} -> {new_build}")
    else:
        old_size = len(old_offsets)
        new_size = len(new_offsets)
        print(f"offsets.json size: {old_size} -> {new_size} bytes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
