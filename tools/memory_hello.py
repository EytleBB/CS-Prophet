"""Stage 0/1: attach to CS2 and read static symbols.

Prints build number, local player controller address, and entity list base.
If these look sane (nonzero, consistent across reads), Stage 2 is safe.

Usage:
    python tools/memory_hello.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pymem

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFSETS_PATH = REPO_ROOT / "src" / "memory_reader" / "offsets.json"


def main() -> int:
    with OFFSETS_PATH.open() as f:
        off = json.load(f)

    try:
        pm = pymem.Pymem("cs2.exe")
    except pymem.exception.ProcessNotFound:
        print("ERROR: cs2.exe not running. Start CS2 first.")
        return 1

    client = pymem.process.module_from_name(pm.process_handle, "client.dll")
    engine = pymem.process.module_from_name(pm.process_handle, "engine2.dll")

    if client is None or engine is None:
        print("ERROR: could not locate client.dll / engine2.dll modules.")
        return 1

    print(f"cs2.exe PID       = {pm.process_id}")
    print(f"client.dll base   = 0x{client.lpBaseOfDll:X}")
    print(f"engine2.dll base  = 0x{engine.lpBaseOfDll:X}")

    build_number = pm.read_int(engine.lpBaseOfDll + off["engine2.dll"]["dwBuildNumber"])
    print(f"buildNumber       = {build_number}")

    local_ctrl = pm.read_longlong(client.lpBaseOfDll + off["client.dll"]["dwLocalPlayerController"])
    print(f"localPlayerCtrl   = 0x{local_ctrl:X}")

    entity_list = pm.read_longlong(client.lpBaseOfDll + off["client.dll"]["dwEntityList"])
    print(f"entityList ptr    = 0x{entity_list:X}")

    # Re-read after a second; values should stay the same (or controller may change
    # if you switched spectator target). This sanity-checks we're reading real memory.
    time.sleep(1.0)
    local_ctrl2 = pm.read_longlong(client.lpBaseOfDll + off["client.dll"]["dwLocalPlayerController"])
    print(f"localPlayerCtrl#2 = 0x{local_ctrl2:X}  (same_pointer={local_ctrl == local_ctrl2})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
