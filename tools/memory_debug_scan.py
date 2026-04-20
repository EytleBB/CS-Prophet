"""Debug: scan entity indices and print class names to see what's there."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pymem
import pymem.process

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFSETS_PATH = REPO_ROOT / "src" / "memory_reader" / "offsets.json"


def main() -> int:
    with OFFSETS_PATH.open() as f:
        off = json.load(f)
    pm = pymem.Pymem("cs2.exe")
    client = pymem.process.module_from_name(pm.process_handle, "client.dll")
    client_base = client.lpBaseOfDll

    ent_list_base = pm.read_longlong(client_base + off["client.dll"]["dwEntityList"])
    print(f"entity list base = 0x{ent_list_base:X}")

    highest = pm.read_int(ent_list_base + off["client.dll"]["dwGameEntitySystem_highestEntityIndex"])
    print(f"highest entity index = {highest}")

    counts: Counter[str] = Counter()
    samples: dict[str, int] = {}
    scan_to = 8192
    for i in range(1, scan_to):
        block = pm.read_longlong(ent_list_base + 0x8 * (i >> 9) + 0x10)
        if block == 0:
            continue
        ent = pm.read_longlong(block + 120 * (i & 0x1FF))
        if ent == 0:
            continue
        try:
            identity = pm.read_longlong(ent + 16)
            if identity == 0: continue
            name_ptr = pm.read_longlong(identity + 32)
            if name_ptr == 0: continue
            name = pm.read_string(name_ptr, 64)
        except Exception:
            continue
        counts[name] += 1
        samples.setdefault(name, i)

    print(f"\n{'class':<40}  count   first_idx")
    for name, c in counts.most_common():
        print(f"{name:<40}  {c:>5}   {samples[name]:>5}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
