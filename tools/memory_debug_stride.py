"""Confirm entity list stride by walking m_pNextByClass and searching for each controller."""

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

    # Walk m_pNextByClass to collect all controller entity pointers.
    # But identity->entity back-mapping is needed. We'll brute-force search each identity in block 0.
    local_identity = pm.read_longlong(local_ctrl + 16)
    identities = [local_identity]
    cur = local_identity
    for _ in range(20):
        nxt = pm.read_longlong(cur + 104)  # m_pNextByClass
        if nxt == 0 or nxt in identities:
            break
        identities.append(nxt)
        cur = nxt
    print(f"identities via m_pNextByClass: {len(identities)}")

    # Read block 0
    block0 = pm.read_longlong(ent_list + 0x10)
    buf = pm.read_bytes(block0, 512 * 120)

    # For each identity, infer its owning entity by searching: which 8-byte-aligned slot in block0
    # contains a pointer X such that read_longlong(X + 16) == identity?
    def find_entity_for_identity(identity: int):
        hits = []
        for offset in range(0, len(buf) - 8, 8):
            val = struct.unpack_from("<Q", buf, offset)[0]
            if val < 0x1000 or val > 0x7FFFFFFFFFFF:
                continue
            # Deref val+16, see if equals identity
            try:
                got = pm.read_longlong(val + 16)
            except Exception:
                continue
            if got == identity:
                hits.append((offset, val))
        return hits

    for ident in identities:
        hits = find_entity_for_identity(ident)
        print(f"  identity=0x{ident:X}: hits={[(hex(o), hex(v)) for o, v in hits]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
