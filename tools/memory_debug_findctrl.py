"""Brute-force search for localPlayerCtrl pointer inside entity list structure."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pymem
import pymem.process

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with (REPO_ROOT / "src" / "memory_reader" / "offsets.json").open() as f:
        off = json.load(f)
    pm = pymem.Pymem("cs2.exe")
    client = pymem.process.module_from_name(pm.process_handle, "client.dll")
    cb = client.lpBaseOfDll

    local_ctrl = pm.read_longlong(cb + off["client.dll"]["dwLocalPlayerController"])
    ent_list = pm.read_longlong(cb + off["client.dll"]["dwEntityList"])
    print(f"localPlayerCtrl = 0x{local_ctrl:X}")
    print(f"ent_list base   = 0x{ent_list:X}")

    # Scan block pointers: ent_list + 0x10, +0x18, +0x20, ...
    # Each points to a 512-entity block.
    for block_idx in range(64):
        block = pm.read_longlong(ent_list + 0x10 + 8 * block_idx)
        if block == 0:
            continue
        # Read block as large buffer, search for local_ctrl pointer
        try:
            buf = pm.read_bytes(block, 120 * 512)
        except Exception:
            continue
        for offset in range(0, len(buf) - 8, 8):
            val = struct.unpack_from("<Q", buf, offset)[0]
            if val == local_ctrl:
                # i ranged as block_idx * 512 + (offset / stride)
                # We don't know stride yet — try both 120 and 8 bytes
                for stride in (8, 16, 32, 64, 120):
                    if offset % stride == 0:
                        idx_guess = block_idx * 512 + offset // stride
                        print(f"  FOUND at block {block_idx}, byte_offset=0x{offset:X}, "
                              f"stride={stride}: entity_index_guess={idx_guess}")
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
