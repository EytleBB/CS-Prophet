"""Stage 3.5/3.6 live eyeball verification for the CS2 memory reader.

Usage:
    python tools/memory_verify.py
    python tools/memory_verify.py --watch 1.0
    python tools/memory_verify.py --map de_dust2
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.state_vector_v2 import FEATURE_NAMES, MAPS  # noqa: E402
from src.inference.memory_reader import CS2MemoryReader  # noqa: E402
from src.inference.memory_state_builder import (  # noqa: E402
    _active_weapon_id,
    _equip_value,
    build_row_from_memory,
)
from src.utils.map_utils import infer_map_from_positions  # noqa: E402


def _flag(value: object) -> str:
    return "1" if bool(value) else "0"


def _format_projectiles(entries: list[tuple[float, float, float]]) -> str:
    if not entries:
        return "-"
    return " ".join(f"({x:.1f},{y:.1f},{remain:.2f})" for x, y, remain in entries)


def _snapshot(reader: CS2MemoryReader, map_override: str) -> None:
    players = reader.read_players()
    map_state = reader.read_map_state()
    bomb = reader.read_bomb()
    projectiles = reader.read_projectiles()

    map_state = dict(map_state)
    map_state["bomb"] = bomb
    map_state["projectiles"] = projectiles

    map_name = map_override or str(map_state.get("map_name", "") or "")
    if not map_name:
        map_name = infer_map_from_positions(
            [(float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0)))
             for p in players if p.get("alive")]
        )
    round_num = int(map_state.get("round_num", 0) or 0)
    row = build_row_from_memory(
        players=players,
        map_state=map_state,
        round_num=round_num,
        map_name=map_name,
    )

    print("Section 1 - players")
    if not players:
        print("no players found")
    else:
        for player in players:
            weapons = ",".join(str(item) for item in player.get("weapons", [])) or "-"
            print(
                f"{player.get('team', ''):<2} "
                f"name={str(player.get('name', ''))[:20]:<20} "
                f"alive={_flag(player.get('alive'))} "
                f"hp={int(player.get('hp', 0) or 0):>3} "
                f"armor={int(player.get('armor', 0) or 0):>3} "
                f"helmet={_flag(player.get('helmet'))} "
                f"money={int(player.get('money', 0) or 0):>5} "
                f"x={float(player.get('x', 0.0) or 0.0):>8.1f} "
                f"y={float(player.get('y', 0.0) or 0.0):>8.1f} "
                f"z={float(player.get('z', 0.0) or 0.0):>7.1f} "
                f"yaw={float(player.get('yaw', 0.0) or 0.0):>6.1f} "
                f"smk={_flag(player.get('has_smoke'))} "
                f"fl={_flag(player.get('has_flash'))} "
                f"he={_flag(player.get('has_he'))} "
                f"mol={_flag(player.get('has_molotov'))} "
                f"c4={_flag(player.get('has_c4'))} "
                f"weapon_id={int(_active_weapon_id(player)):>2} "
                f"equip_value={int(_equip_value(player)):>4} "
                f"weapon={str(player.get('active_weapon_class') or '-')} "
                f"weapons={weapons}"
            )

    print()
    print("Section 2 - bomb")
    print(
        f"planted={_flag(bomb.get('planted'))} "
        f"dropped={_flag(bomb.get('dropped'))} "
        f"x={float(bomb.get('x', 0.0) or 0.0):.1f} "
        f"y={float(bomb.get('y', 0.0) or 0.0):.1f} "
        f"site={str(bomb.get('site', '') or '-')}"
    )

    print()
    print("Section 3 - projectiles")
    smokes = list(projectiles.get("smokes", []))
    molotovs = list(projectiles.get("molotovs", []))
    print(f"smokes: {len(smokes)} entries: {_format_projectiles(smokes)}")
    print(f"molotovs: {len(molotovs)} entries: {_format_projectiles(molotovs)}")

    print()
    print("Section 4 - row dict sanity")
    if row is None:
        print(
            f"row build returned None (players={len(players)}, map_name={map_name or '-'}, "
            f"known_map={map_name in MAPS})"
        )
        return

    assert len(row) == 218
    player_nonzero = sum(
        1
        for name, value in row.items()
        if (
            any(name.startswith(f"t{idx}_") for idx in range(5))
            or any(name.startswith(f"ct{idx}_") for idx in range(5))
        )
        and value != 0.0
    )
    global_nonzero = sum(1 for name in ("ct_score", "t_score", "round_num", "time_in_round") if row[name] != 0.0)
    bomb_nonzero = sum(1 for name in ("bomb_dropped", "bomb_x", "bomb_y") if row[name] != 0.0)
    util_nonzero = sum(
        1
        for name, value in row.items()
        if (name.startswith("smoke") or name.startswith("molotov")) and value != 0.0
    )
    map_nonzero = sum(1 for name, value in row.items() if name.startswith("map_") and value != 0.0)
    bad_values = [name for name, value in row.items() if not math.isfinite(float(value))]

    print(f"len(row)={len(row)} expected={len(FEATURE_NAMES)}")
    print(
        f"nonzero_counts players={player_nonzero} global={global_nonzero} "
        f"bomb={bomb_nonzero} util={util_nonzero} map={map_nonzero}"
    )
    print(f"nan_or_inf={len(bad_values)}")
    if bad_values:
        print(f"bad_keys={','.join(bad_values[:12])}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Eyeball-verify Stage 3.5/3.6 CS2 memory reads.")
    ap.add_argument("--watch", type=float, default=0.0, help="Repeat every N seconds (0=once)")
    ap.add_argument("--map", default="", help="Override map name passed into build_row_from_memory")
    args = ap.parse_args()

    try:
        reader = CS2MemoryReader.attach()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.watch <= 0:
        _snapshot(reader, args.map)
        return 0

    try:
        while True:
            print("=" * 120)
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            _snapshot(reader, args.map)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
