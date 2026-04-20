"""Diagnose C_Team score reads: dump raw m_szTeamname / m_iScore for handles 1..256.

Tries three string-storage layouts at the name offset:
  - inline ASCII
  - CUtlString style: field holds a pointer to a char*
  - string_t: field -> ptr -> ptr -> char*

Prints whichever yields printable text, plus raw hex. Also prints any entity
whose class_name contains "team" even if the name read fails.

Usage:
    python tools/memory_dump_teams.py
    python tools/memory_dump_teams.py --scan 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.memory_reader import (  # noqa: E402
    CS2MemoryReader,
    _clean_inline_string,
    _safe_read_string,
    entity_class_name,
    resolve_handle,
)
from src.inference.memory_reader import _optional_field  # noqa: E402


def _probe_name(pm, field_addr: int) -> dict[str, str]:
    """Try multiple storage layouts. Return what each yielded."""
    results: dict[str, str] = {}

    # Raw hex of the first 32 bytes at the field itself.
    try:
        raw = pm.read_bytes(field_addr, 32)
        results["raw_hex"] = raw.hex()
        results["inline"] = _clean_inline_string(raw.decode("latin-1", errors="replace"))
    except Exception as exc:
        results["raw_hex"] = f"<err: {exc}>"
        results["inline"] = ""

    # CUtlString: field holds a direct pointer to a C string.
    try:
        p1 = pm.read_longlong(field_addr)
        if 0x10000 < p1 < 0x7FFFFFFFFFFF:
            text = _clean_inline_string(pm.read_string(p1, 64))
            results["ptr1"] = text
            results["ptr1_addr"] = hex(p1)
        else:
            results["ptr1"] = ""
            results["ptr1_addr"] = hex(p1)
    except Exception as exc:
        results["ptr1"] = f"<err: {exc}>"

    # string_t: field -> ptr -> ptr -> char*.
    try:
        p1 = pm.read_longlong(field_addr)
        if 0x10000 < p1 < 0x7FFFFFFFFFFF:
            p2 = pm.read_longlong(p1)
            if 0x10000 < p2 < 0x7FFFFFFFFFFF:
                text = _clean_inline_string(pm.read_string(p2, 64))
                results["ptr2"] = text
                results["ptr2_addr"] = hex(p2)
    except Exception:
        pass

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump raw C_Team name/score across entity handles.")
    ap.add_argument("--scan", type=int, default=256, help="Max handle to scan (default 256)")
    ap.add_argument("--out", default="data/team_dump.txt", help="Write report here (utf-8)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("w", encoding="utf-8")

    def echo(line: str = "") -> None:
        out_file.write(line + "\n")

    try:
        reader = CS2MemoryReader.attach()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    pm = reader.pm
    classes = reader.classes

    score_offset = _optional_field(classes, "C_Team", "m_iScore")
    name_offset = _optional_field(classes, "C_Team", "m_szTeamname")
    num_offset = _optional_field(classes, "C_Team", "m_iTeamNum")

    echo(f"C_Team.m_iScore       offset: {score_offset}")
    echo(f"C_Team.m_szTeamname   offset: {name_offset}")
    echo(f"C_Team.m_iTeamNum     offset: {num_offset}")
    echo()

    if score_offset is None or name_offset is None:
        echo("client_dll.json is missing C_Team fields — nothing to probe.")
        out_file.close()
        return 2

    ent_list_base = reader._read_entity_list_base()
    if not ent_list_base:
        echo("entity list base resolved to 0 — CS2 not running?")
        out_file.close()
        return 3
    echo(f"entity_list_base = {hex(ent_list_base)}")
    echo()

    # Class histogram across the full scan so we can spot the real team class.
    class_hist: dict[str, int] = {}
    # Entities whose score value is in the valid game range (0..30 inclusive).
    plausible: list[tuple[int, str, int, dict[str, str]]] = []
    # Entities whose class name mentions team.
    team_like: list[tuple[int, str, int, dict[str, str]]] = []

    for handle in range(1, args.scan + 1):
        entity_ptr = resolve_handle(pm, ent_list_base, handle)
        if entity_ptr == 0:
            continue

        cls = entity_class_name(pm, entity_ptr, classes)
        if not cls:
            continue

        class_hist[cls] = class_hist.get(cls, 0) + 1

        try:
            score = int(pm.read_int(entity_ptr + score_offset))
        except Exception:
            score = -999

        # Only probe name for potentially interesting rows; dumping every entity
        # is spammy.
        interesting = ("team" in cls.lower()) or (0 <= score <= 30)
        if not interesting:
            continue

        probe = _probe_name(pm, entity_ptr + name_offset)
        try:
            team_num = pm.read_int(entity_ptr + num_offset) if num_offset is not None else None
        except Exception:
            team_num = None

        row = (handle, cls, score, probe)
        if 0 <= score <= 30:
            plausible.append(row)
        if "team" in cls.lower():
            team_like.append(row)

    def _emit_row(handle, cls, score, probe):
        echo(f"handle={handle:<4} class={cls:<40} score={score}")
        echo(f"    inline  : {probe.get('inline', '')!r}")
        echo(f"    ptr1    : {probe.get('ptr1', '')!r}  (addr={probe.get('ptr1_addr', '')})")
        if "ptr2" in probe:
            echo(f"    ptr2    : {probe.get('ptr2', '')!r}  (addr={probe.get('ptr2_addr', '')})")
        echo(f"    raw_hex : {probe.get('raw_hex', '')}")
        echo()

    echo("=" * 70)
    echo("Section A: entities whose score is in the plausible 0..30 range")
    echo("=" * 70)
    if not plausible:
        echo("(none)")
    for row in plausible:
        _emit_row(*row)

    echo("=" * 70)
    echo("Section B: entities whose class name contains 'team'")
    echo("=" * 70)
    for row in team_like:
        _emit_row(*row)

    echo("=" * 70)
    echo(f"Section C: class-name histogram across {args.scan} handles")
    echo("=" * 70)
    for cls, count in sorted(class_hist.items(), key=lambda kv: (-kv[1], kv[0])):
        echo(f"  {count:>4}  {cls}")

    out_file.close()
    print(f"wrote report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
