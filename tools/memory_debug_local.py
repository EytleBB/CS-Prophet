"""Debug: read local player controller directly via dwLocalPlayerController."""

from __future__ import annotations

import json
from pathlib import Path

import pymem
import pymem.process

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_class_name(pm, entity: int) -> str:
    try:
        identity = pm.read_longlong(entity + 16)
        if identity == 0:
            return ""
        name_ptr = pm.read_longlong(identity + 32)
        if name_ptr == 0:
            return ""
        return pm.read_string(name_ptr, 64)
    except Exception as e:
        return f"<err {e}>"


def main() -> int:
    with (REPO_ROOT / "src" / "memory_reader" / "offsets.json").open() as f:
        off = json.load(f)
    pm = pymem.Pymem("cs2.exe")
    client = pymem.process.module_from_name(pm.process_handle, "client.dll")
    cb = client.lpBaseOfDll

    local_ctrl = pm.read_longlong(cb + off["client.dll"]["dwLocalPlayerController"])
    print(f"localPlayerCtrl  = 0x{local_ctrl:X}")
    print(f"  class          = {read_class_name(pm, local_ctrl)}")

    local_pawn = pm.read_longlong(cb + off["client.dll"]["dwLocalPlayerPawn"])
    print(f"localPlayerPawn  = 0x{local_pawn:X}")
    print(f"  class          = {read_class_name(pm, local_pawn)}")

    # Get entity index of local controller via CEntityIdentity.m_nameStringableIndex?
    # Actually simpler: CEntityIdentity.m_pEntity->m_EHandle or similar. Skip for now.
    # Instead use the identity linked-list via m_pNextByClass to walk all controllers.
    print("\nWalking CEntityIdentity linked list from localPlayerCtrl via m_pNextByClass:")
    identity = pm.read_longlong(local_ctrl + 16)
    print(f"  identity       = 0x{identity:X}")
    if identity:
        seen = {identity}
        cur = identity
        for step in range(16):
            cur = pm.read_longlong(cur + 104)  # CEntityIdentity.m_pNextByClass
            if cur == 0 or cur in seen:
                break
            seen.add(cur)
            # Walk back up to entity: identity has parent pointer? No — identity.m_pEntity doesn't exist.
            # But we know entity points to identity via offset 16, so entity = identity - some_offset is wrong.
            # Alternate: CEntityIdentity usually has a pointer back to the entity. Let me check offsets.
            # Print what we have at this identity.
            name_ptr = pm.read_longlong(cur + 32)
            cls_name = pm.read_string(name_ptr, 64) if name_ptr else ""
            print(f"  step={step} identity=0x{cur:X} class={cls_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
