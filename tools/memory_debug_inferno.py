"""Inspect live inferno/molotov-related entities from CS2 memory.

Usage:
    python tools/memory_debug_inferno.py
    python tools/memory_debug_inferno.py --watch 1.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.memory_reader import CS2MemoryReader  # noqa: E402


def _print_snapshot(reader: CS2MemoryReader) -> None:
    projectiles = reader.read_projectiles()
    candidates = reader.debug_inferno_candidates()

    print("Section 1 - read_projectiles()")
    molotovs = list(projectiles.get("molotovs", []))
    if molotovs:
        print("molotovs:", " ".join(f"({x:.1f},{y:.1f},{remain:.2f})" for x, y, remain in molotovs))
    else:
        print("molotovs: -")

    print()
    print("Section 2 - inferno candidates")
    if not candidates:
        print("no inferno/molotov/incendiary-related entities found")
        return

    for item in candidates:
        print(
            f"idx={int(item.get('entity_index', 0)):>4} "
            f"class={str(item.get('class_name', '')):<28} "
            f"accepted={1 if item.get('accepted') else 0} "
            f"seed={1 if item.get('synthetic_seedable') else 0} "
            f"burn={1 if item.get('is_burning') else 0} "
            f"post={1 if item.get('in_post_effect') else 0} "
            f"explode={1 if item.get('explode_effect_began') else 0} "
            f"live={1 if item.get('is_live') else 0} "
            f"inc={1 if item.get('is_inc_grenade') else 0} "
            f"bounce={int(item.get('bounces', 0)):>2} "
            f"count={int(item.get('fire_count', 0)):>2} "
            f"type={int(item.get('inferno_type', -1)):>2} "
            f"tick={int(item.get('tick_begin', 0)):>6} "
            f"life={float(item.get('fire_lifetime', 0.0)):.2f} "
            f"pos=({float(item.get('x', 0.0)):.1f},{float(item.get('y', 0.0)):.1f}) "
            f"src={str(item.get('position_source', '-'))}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug inferno/molotov-related entities from live CS2 memory.")
    ap.add_argument("--watch", type=float, default=0.0, help="Repeat every N seconds (0=once)")
    args = ap.parse_args()

    try:
        reader = CS2MemoryReader.attach()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.watch <= 0:
        _print_snapshot(reader)
        return 0

    try:
        while True:
            print("=" * 120)
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            _print_snapshot(reader)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
