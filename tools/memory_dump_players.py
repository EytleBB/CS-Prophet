"""Stage 2: iterate entity list, dump all player controllers + pawns.

Reads name, steamid, team, pos, yaw, hp, armor, money, alive/scoped/defusing flags
for every valid player entity. Use this to eyeball-verify memory reads match CS2 HUD.

Usage:
    python tools/memory_dump_players.py                  # single snapshot
    python tools/memory_dump_players.py --watch 0.5      # repeat every 0.5s
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pymem
import pymem.process

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.memory_reader import (  # noqa: E402
    TEAM_NAMES,
    entity_class_name,
    load_offsets,
    read_player_name,
    read_vec3,
    resolve_handle,
)


def dump_once(pm: pymem.Pymem, client_base: int, off: dict, classes: dict) -> None:
    f_ctrl = classes["CCSPlayerController"]["fields"]
    f_pawn = classes["C_CSPlayerPawn"]["fields"]
    f_base = classes["C_BaseEntity"]["fields"]
    f_bpc  = classes["CBasePlayerController"]["fields"]
    f_node = classes["CGameSceneNode"]["fields"]
    f_money = classes["CCSPlayerController_InGameMoneyServices"]["fields"]

    ent_list_base = pm.read_longlong(client_base + off["client.dll"]["dwEntityList"])

    rows = []
    for i in range(1, 65):
        ctrl = resolve_handle(pm, ent_list_base, i)
        if ctrl == 0:
            continue

        # Filter by class name — indices 1..64 also hold world entities.
        cls_name = entity_class_name(pm, ctrl)
        if cls_name != "cs_player_controller":
            continue

        try:
            pawn_handle = pm.read_uint(ctrl + f_ctrl["m_hPlayerPawn"])
        except Exception:
            continue
        if pawn_handle == 0xFFFFFFFF:
            continue

        try:
            name = read_player_name(pm, ctrl, f_bpc["m_iszPlayerName"])
            # Fallback to sanitized name if primary didn't land
            if not name and "m_sSanitizedPlayerName" in f_ctrl:
                name = read_player_name(pm, ctrl, f_ctrl["m_sSanitizedPlayerName"])
            steam_id = pm.read_ulonglong(ctrl + f_bpc["m_steamID"])
            team = pm.read_uchar(ctrl + f_base["m_iTeamNum"])
            is_alive = bool(pm.read_uchar(ctrl + f_ctrl["m_bPawnIsAlive"]))
            hp_ctrl = pm.read_int(ctrl + f_ctrl["m_iPawnHealth"])
            armor_ctrl = pm.read_int(ctrl + f_ctrl["m_iPawnArmor"])
            helmet = bool(pm.read_uchar(ctrl + f_ctrl["m_bPawnHasHelmet"]))
            defuser = bool(pm.read_uchar(ctrl + f_ctrl["m_bPawnHasDefuser"]))
            money_svc = pm.read_longlong(ctrl + f_ctrl["m_pInGameMoneyServices"])
            money = pm.read_int(money_svc + f_money["m_iAccount"]) if money_svc else -1
        except Exception as exc:
            print(f"  idx={i} controller read failed: {exc}")
            continue

        pawn = resolve_handle(pm, ent_list_base, pawn_handle)
        pos = (0.0, 0.0, 0.0)
        yaw = 0.0
        scoped = defusing = False
        if pawn:
            try:
                scene_node = pm.read_longlong(pawn + f_base["m_pGameSceneNode"])
                if scene_node:
                    pos = read_vec3(pm, scene_node + f_node["m_vecAbsOrigin"])
                yaw = pm.read_float(pawn + f_pawn["m_angEyeAngles"] + 4)  # QAngle: [pitch, yaw, roll]
                scoped = bool(pm.read_uchar(pawn + f_pawn["m_bIsScoped"]))
                defusing = bool(pm.read_uchar(pawn + f_pawn["m_bIsDefusing"]))
            except Exception as exc:
                print(f"  idx={i} pawn read failed: {exc}")

        rows.append({
            "idx": i,
            "name": name,
            "steamid": steam_id,
            "team": TEAM_NAMES.get(team, f"T{team}"),
            "alive": is_alive,
            "hp": hp_ctrl,
            "armor": armor_ctrl,
            "money": money,
            "helmet": helmet,
            "defuser": defuser,
            "pos": pos,
            "yaw": yaw,
            "scoped": scoped,
            "defusing": defusing,
        })

    if not rows:
        print("no player controllers found")
        return

    print(f"{'idx':>3} {'team':<4} {'name':<22} {'alive':<5} {'hp':>4} {'arm':>4} {'money':>6}  "
          f"{'x':>9} {'y':>9} {'z':>8} {'yaw':>7}  flags")
    for r in rows:
        flags = []
        if r["helmet"]: flags.append("helm")
        if r["defuser"]: flags.append("kit")
        if r["scoped"]: flags.append("scp")
        if r["defusing"]: flags.append("defuse")
        print(f"{r['idx']:>3} {r['team']:<4} {r['name'][:22]:<22} {str(r['alive']):<5} "
              f"{r['hp']:>4} {r['armor']:>4} {r['money']:>6}  "
              f"{r['pos'][0]:>9.1f} {r['pos'][1]:>9.1f} {r['pos'][2]:>8.1f} {r['yaw']:>7.1f}  {','.join(flags)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=0.0, help="Repeat every N seconds (0=once)")
    args = ap.parse_args()

    off, classes = load_offsets()

    try:
        pm = pymem.Pymem("cs2.exe")
    except pymem.exception.ProcessNotFound:
        print("ERROR: cs2.exe not running.")
        return 1

    client = pymem.process.module_from_name(pm.process_handle, "client.dll")
    if client is None:
        print("ERROR: client.dll not located.")
        return 1

    print(f"cs2.exe PID={pm.process_id}  client.dll=0x{client.lpBaseOfDll:X}")
    print()

    if args.watch <= 0:
        dump_once(pm, client.lpBaseOfDll, off, classes)
        return 0

    try:
        while True:
            print("=" * 110)
            print(time.strftime("%H:%M:%S"))
            dump_once(pm, client.lpBaseOfDll, off, classes)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
