"""Legacy preview helper for the old 348-dim exploratory schema.

This script is kept for historical debugging only. The active v2 pipeline now
uses ``src/features/feature_builder_v2.py`` + ``src/features/state_vector_v2.py``
with the 218-dim realtime-aligned schema.
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path, resolve_path_input

# ── Weapon classification ────────────────────────────────────────────────
_PRIMARY = {
    "AK-47", "M4A4", "M4A1-S", "FAMAS", "Galil AR", "SG 553", "AUG",
    "AWP", "SSG 08", "SCAR-20", "G3SG1",
    "MP9", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "MAC-10", "MP7",
    "Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev",
}
_SECONDARY = {
    "Glock-18", "USP-S", "P2000", "P250", "Five-SeveN", "Tec-9",
    "CZ75-Auto", "Desert Eagle", "R8 Revolver", "Dual Berettas",
}
_WEAPON_CAT = {}
for _w in {"AK-47", "M4A4", "M4A1-S", "FAMAS", "Galil AR", "SG 553", "AUG"}:
    _WEAPON_CAT[_w] = "rifle"
for _w in {"AWP", "SSG 08", "SCAR-20", "G3SG1"}:
    _WEAPON_CAT[_w] = "sniper"
for _w in {"MP9", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "MAC-10", "MP7"}:
    _WEAPON_CAT[_w] = "smg"
for _w in {"Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev"}:
    _WEAPON_CAT[_w] = "heavy"
for _w in _SECONDARY:
    _WEAPON_CAT[_w] = "pistol"

WEAPON_CATS = ["pistol", "rifle", "sniper", "smg", "heavy", "none"]

MAPS = ["de_mirage", "de_inferno", "de_dust2", "de_nuke", "de_ancient", "de_overpass", "de_anubis"]

SMOKE_SLOTS = 5
MOLOTOV_SLOTS = 3
SMOKE_DURATION_TICKS = 18 * 64   # 1152
MOLOTOV_DURATION_TICKS = 7 * 64  # 448


def best_weapon(inventory) -> str:
    if not isinstance(inventory, (list, np.ndarray)):
        return "none"
    primary = None
    secondary = None
    for item in inventory:
        s = str(item)
        if s in _PRIMARY:
            primary = s
        elif s in _SECONDARY:
            secondary = s
    return primary or secondary or "none"


def weapon_onehot(weapon_name: str) -> list[float]:
    cat = _WEAPON_CAT.get(weapon_name, "none")
    return [1.0 if c == cat else 0.0 for c in WEAPON_CATS]


def map_onehot(map_name: str) -> list[float]:
    return [1.0 if m == map_name else 0.0 for m in MAPS]


def has_c4(inventory) -> bool:
    if not isinstance(inventory, (list, np.ndarray)):
        return False
    return "C4 Explosive" in [str(x) for x in inventory]


def _safe_float(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _safe_int(value, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(value)


def _lookup_player_xy_at_tick(
    tick_df: pd.DataFrame,
    round_num: int,
    player_name: str,
    target_tick: int,
) -> tuple[float, float] | None:
    """Best-effort player position at a specific event tick for older pkls."""
    if not player_name:
        return None

    rows = tick_df[(tick_df["round_num"] == round_num) & (tick_df["name"] == player_name)]
    if rows.empty:
        return None

    exact = rows[rows["tick"] == target_tick]
    if not exact.empty:
        row = exact.iloc[0]
        return _safe_float(row.get("X", 0.0)), _safe_float(row.get("Y", 0.0))

    earlier = rows[rows["tick"] <= target_tick].sort_values("tick")
    if not earlier.empty:
        row = earlier.iloc[-1]
        return _safe_float(row.get("X", 0.0)), _safe_float(row.get("Y", 0.0))

    nearest_idx = (rows["tick"] - target_tick).abs().idxmin()
    row = rows.loc[nearest_idx]
    return _safe_float(row.get("X", 0.0)), _safe_float(row.get("Y", 0.0))


def _event_xy(row: pd.Series) -> tuple[float, float] | None:
    """Read event coordinates from either normalized or demoparser2-prefixed cols."""
    for x_col, y_col in (("X", "Y"), ("user_X", "user_Y")):
        if x_col in row.index and y_col in row.index and pd.notna(row.get(x_col)) and pd.notna(row.get(y_col)):
            return _safe_float(row[x_col]), _safe_float(row[y_col])
    return None


def get_bomb_state(
    tick_df: pd.DataFrame,
    events: dict,
    round_info: pd.DataFrame,
    current_tick: int,
    current_round: int,
) -> tuple[bool, float, float]:
    """Return (bomb_dropped, bomb_x, bomb_y) at current_tick.

    Logic: replay bomb_dropped/bomb_pickup events up to current_tick.
    If the last event is a drop, bomb is on the ground at the exact drop point.
    """
    drop_df = events.get("bomb_dropped", pd.DataFrame())
    pick_df = events.get("bomb_pickup", pd.DataFrame())
    ri = round_info[round_info["round_num"] == current_round]
    round_start = int(ri["freeze_tick"].iloc[0]) if not ri.empty else 0
    round_end = int(ri["end_tick"].iloc[0]) if not ri.empty and pd.notna(ri["end_tick"].iloc[0]) else current_tick

    # Collect all drop/pickup events in this round up to current_tick
    timeline = []  # (tick, order, type, row)
    if not drop_df.empty and "tick" in drop_df.columns:
        rnd = drop_df[(drop_df["tick"] >= round_start) & (drop_df["tick"] <= round_end)]
        for _, row in rnd.iterrows():
            t = int(row["tick"])
            if t <= current_tick:
                timeline.append((t, 0, "drop", row))
    if not pick_df.empty and "tick" in pick_df.columns:
        rnd = pick_df[(pick_df["tick"] >= round_start) & (pick_df["tick"] <= round_end)]
        for _, row in rnd.iterrows():
            t = int(row["tick"])
            if t <= current_tick:
                timeline.append((t, 1, "pick", row))

    if not timeline:
        return False, 0.0, 0.0

    timeline.sort(key=lambda x: (x[0], x[1]))
    last_event = timeline[-1]

    if last_event[2] == "pick":
        return False, 0.0, 0.0

    # Bomb is dropped — find dropper's position at drop tick
    drop_tick = last_event[0]
    drop_row = last_event[3]

    event_pos = _event_xy(drop_row)
    if event_pos is not None:
        return True, event_pos[0], event_pos[1]

    dropper = str(drop_row.get("user_name", ""))
    pos = _lookup_player_xy_at_tick(tick_df, current_round, dropper, drop_tick)
    if pos is not None:
        return True, pos[0], pos[1]

    return True, 0.0, 0.0


def get_active_utils(events: dict, current_tick: int, current_round: int):
    """Return active smoke and molotov positions at current_tick."""
    smokes = []  # list of (x, y, remaining_normalized)
    molotovs = []

    smoke_df = events.get("smokegrenade_detonate", pd.DataFrame())
    if not smoke_df.empty and "round_num" in smoke_df.columns:
        rnd = smoke_df[smoke_df["round_num"] == current_round]
        for _, row in rnd.iterrows():
            det_tick = int(row["tick"])
            elapsed = current_tick - det_tick
            if 0 <= elapsed < SMOKE_DURATION_TICKS:
                remaining = 1.0 - elapsed / SMOKE_DURATION_TICKS
                smokes.append((float(row["x"]), float(row["y"]), remaining))

    fire_df = events.get("inferno_startburn", pd.DataFrame())
    if not fire_df.empty and "round_num" in fire_df.columns:
        rnd = fire_df[fire_df["round_num"] == current_round]
        for _, row in rnd.iterrows():
            det_tick = int(row["tick"])
            elapsed = current_tick - det_tick
            if 0 <= elapsed < MOLOTOV_DURATION_TICKS:
                remaining = 1.0 - elapsed / MOLOTOV_DURATION_TICKS
                molotovs.append((float(row["x"]), float(row["y"]), remaining))

    return smokes, molotovs


def build_feature_vector(tick_df: pd.DataFrame, tick_slice: pd.DataFrame, events: dict,
                         tick: int, round_num: int, map_name: str,
                         round_info: pd.DataFrame) -> dict:
    """Build the 348-dim feature dict for one tick."""
    t_rows = tick_slice[tick_slice["team_name"] == "TERRORIST"].sort_values("name")
    ct_rows = tick_slice[tick_slice["team_name"] == "CT"].sort_values("name")

    feat = {}

    # Per-player features (10 players × 31 dims = 310)
    for side_prefix, side_rows in [("t", t_rows), ("ct", ct_rows)]:
        for i in range(5):
            p_prefix = f"{side_prefix}{i}"
            if i < len(side_rows):
                p = side_rows.iloc[i]
                alive = bool(p.get("is_alive", False))

                feat[f"{p_prefix}_name"] = str(p.get("name", ""))
                feat[f"{p_prefix}_x"] = _safe_float(p.get("X", 0.0))
                feat[f"{p_prefix}_y"] = _safe_float(p.get("Y", 0.0))
                feat[f"{p_prefix}_z"] = _safe_float(p.get("Z", 0.0))
                feat[f"{p_prefix}_vx"] = _safe_float(p.get("velocity_X", 0.0)) if alive else 0.0
                feat[f"{p_prefix}_vy"] = _safe_float(p.get("velocity_Y", 0.0)) if alive else 0.0
                feat[f"{p_prefix}_vz"] = _safe_float(p.get("velocity_Z", 0.0)) if alive else 0.0
                feat[f"{p_prefix}_yaw"] = _safe_float(p.get("yaw", 0.0)) if alive else 0.0
                feat[f"{p_prefix}_is_walking"] = bool(p.get("is_walking", False)) if alive else False
                feat[f"{p_prefix}_ducking"] = bool(p.get("ducking", False)) if alive else False
                feat[f"{p_prefix}_in_bomb_zone"] = bool(p.get("in_bomb_zone", False)) if alive else False
                feat[f"{p_prefix}_hp"] = _safe_int(p.get("health", 0))
                feat[f"{p_prefix}_armor"] = _safe_int(p.get("armor_value", 0))
                feat[f"{p_prefix}_helmet"] = bool(p.get("has_helmet", False))
                feat[f"{p_prefix}_alive"] = alive
                feat[f"{p_prefix}_spotted"] = bool(p.get("spotted", False)) if alive else False
                feat[f"{p_prefix}_is_scoped"] = bool(p.get("is_scoped", False)) if alive else False
                feat[f"{p_prefix}_flash_dur"] = _safe_float(p.get("flash_duration", 0.0))
                inv = p.get("inventory", [])
                if not isinstance(inv, list):
                    inv = []
                bw = best_weapon(inv)
                feat[f"{p_prefix}_best_weapon"] = bw
                feat[f"{p_prefix}_weapon_cat"] = weapon_onehot(bw)
                feat[f"{p_prefix}_has_smoke"] = "Smoke Grenade" in [str(x) for x in inv]
                feat[f"{p_prefix}_has_flash"] = "Flashbang" in [str(x) for x in inv]
                feat[f"{p_prefix}_has_he"] = any(x in [str(v) for v in inv] for x in ["HE Grenade", "High Explosive Grenade"])
                feat[f"{p_prefix}_has_molotov"] = any(x in [str(v) for v in inv] for x in ["Molotov", "Incendiary Grenade"])
                feat[f"{p_prefix}_has_c4"] = has_c4(inv)
                feat[f"{p_prefix}_balance"] = _safe_int(p.get("balance", 0))
                feat[f"{p_prefix}_equip_value"] = _safe_int(p.get("current_equip_value", 0))
                feat[f"{p_prefix}_score"] = _safe_int(p.get("score", 0))
            else:
                feat[f"{p_prefix}_name"] = "(empty)"
                for k in ["x","y","z","vx","vy","vz","yaw"]:
                    feat[f"{p_prefix}_{k}"] = 0.0
                for k in ["is_walking","ducking","in_bomb_zone","helmet","alive","spotted","is_scoped","has_smoke","has_flash","has_he","has_molotov","has_c4"]:
                    feat[f"{p_prefix}_{k}"] = False
                feat[f"{p_prefix}_hp"] = 0
                feat[f"{p_prefix}_armor"] = 0
                feat[f"{p_prefix}_flash_dur"] = 0.0
                feat[f"{p_prefix}_best_weapon"] = "none"
                feat[f"{p_prefix}_weapon_cat"] = weapon_onehot("none")
                feat[f"{p_prefix}_balance"] = 0
                feat[f"{p_prefix}_equip_value"] = 0
                feat[f"{p_prefix}_score"] = 0

    # Global features (4 dims: ct_score, t_score, round_num, time_in_round)
    ri = round_info[round_info["round_num"] == round_num]
    ct_score = _safe_int(ri["ct_score"].iloc[0]) if not ri.empty and "ct_score" in ri.columns else 0
    t_score = _safe_int(ri["t_score"].iloc[0]) if not ri.empty and "t_score" in ri.columns else 0
    freeze_tick = _safe_int(ri["freeze_tick"].iloc[0], tick) if not ri.empty else tick
    time_in_round = (tick - freeze_tick) / 64.0  # seconds since freeze_end
    feat["ct_score"] = ct_score
    feat["t_score"] = t_score
    feat["round_num"] = round_num
    feat["time_in_round"] = round(time_in_round, 2)

    # Bomb state (3 dims: bomb_dropped, bomb_x, bomb_y)
    bomb_dropped, bomb_x, bomb_y = get_bomb_state(tick_df, events, round_info, tick, round_num)
    feat["bomb_dropped"] = bomb_dropped
    feat["bomb_x"] = bomb_x
    feat["bomb_y"] = bomb_y

    # Active utility slots (smoke 5×3=15, molotov 3×3=9 = 24 dims)
    smokes, molotovs = get_active_utils(events, tick, round_num)
    for s in range(SMOKE_SLOTS):
        if s < len(smokes):
            feat[f"smoke{s}_x"] = smokes[s][0]
            feat[f"smoke{s}_y"] = smokes[s][1]
            feat[f"smoke{s}_remain"] = smokes[s][2]
        else:
            feat[f"smoke{s}_x"] = 0.0
            feat[f"smoke{s}_y"] = 0.0
            feat[f"smoke{s}_remain"] = 0.0

    for m in range(MOLOTOV_SLOTS):
        if m < len(molotovs):
            feat[f"molotov{m}_x"] = molotovs[m][0]
            feat[f"molotov{m}_y"] = molotovs[m][1]
            feat[f"molotov{m}_remain"] = molotovs[m][2]
        else:
            feat[f"molotov{m}_x"] = 0.0
            feat[f"molotov{m}_y"] = 0.0
            feat[f"molotov{m}_remain"] = 0.0

    # Map one-hot (7 dims)
    feat["map_onehot"] = map_onehot(map_name)

    return feat


def print_feature_vector(feat: dict, tick: int, round_num: int):
    print(f"\n{'='*80}")
    print(f"  TICK {tick}  |  ROUND {round_num}  |  {feat['time_in_round']}s into round  |  "
          f"Score: CT {feat['ct_score']} - {feat['t_score']} T")
    print(f"{'='*80}")

    for side, label in [("t", "TERRORIST"), ("ct", "CT")]:
        print(f"\n  [{label}]")
        for i in range(5):
            p = f"{side}{i}"
            name = feat[f"{p}_name"]
            if name == "(empty)":
                print(f"    {p}: (empty slot)")
                continue
            alive = feat[f"{p}_alive"]
            status = "ALIVE" if alive else "DEAD"
            bw = feat[f"{p}_best_weapon"]
            cat = [WEAPON_CATS[j] for j, v in enumerate(feat[f"{p}_weapon_cat"]) if v == 1.0]
            cat_str = cat[0] if cat else "none"

            print(f"    {p}: {name} [{status}]")
            print(f"        pos=({feat[f'{p}_x']:.1f}, {feat[f'{p}_y']:.1f}, {feat[f'{p}_z']:.1f})  "
                  f"vel=({feat[f'{p}_vx']:.1f}, {feat[f'{p}_vy']:.1f}, {feat[f'{p}_vz']:.1f})  "
                  f"yaw={feat[f'{p}_yaw']:.1f}")
            print(f"        hp={feat[f'{p}_hp']}  armor={feat[f'{p}_armor']}  helmet={feat[f'{p}_helmet']}  "
                  f"spotted={feat[f'{p}_spotted']}  scoped={feat[f'{p}_is_scoped']}  "
                  f"flash={feat[f'{p}_flash_dur']:.1f}")
            print(f"        walk={feat[f'{p}_is_walking']}  duck={feat[f'{p}_ducking']}  "
                  f"bomb_zone={feat[f'{p}_in_bomb_zone']}")
            print(f"        weapon={bw} ({cat_str})  "
                  f"smoke={feat[f'{p}_has_smoke']}  flash={feat[f'{p}_has_flash']}  "
                  f"he={feat[f'{p}_has_he']}  molotov={feat[f'{p}_has_molotov']}  "
                  f"c4={feat[f'{p}_has_c4']}")
            print(f"        balance={feat[f'{p}_balance']}  equip={feat[f'{p}_equip_value']}  "
                  f"score={feat[f'{p}_score']}")

    # Active utilities
    print(f"\n  [ACTIVE UTILITIES]")
    any_active = False
    for s in range(SMOKE_SLOTS):
        if feat[f"smoke{s}_remain"] > 0:
            print(f"    smoke{s}: ({feat[f'smoke{s}_x']:.1f}, {feat[f'smoke{s}_y']:.1f})  "
                  f"remaining={feat[f'smoke{s}_remain']:.2f}")
            any_active = True
    for m in range(MOLOTOV_SLOTS):
        if feat[f"molotov{m}_remain"] > 0:
            print(f"    molotov{m}: ({feat[f'molotov{m}_x']:.1f}, {feat[f'molotov{m}_y']:.1f})  "
                  f"remaining={feat[f'molotov{m}_remain']:.2f}")
            any_active = True
    if not any_active:
        print("    (none)")

    # Bomb state
    print(f"\n  [BOMB]")
    if feat["bomb_dropped"]:
        print(f"    DROPPED at ({feat['bomb_x']:.1f}, {feat['bomb_y']:.1f})")
    else:
        print(f"    Carried by a player")

    # Map
    map_vec = feat["map_onehot"]
    active_map = [MAPS[i] for i, v in enumerate(map_vec) if v == 1.0]
    print(f"\n  [GLOBAL]  map={active_map[0] if active_map else '?'}  "
          f"round={feat['round_num']}  time={feat['time_in_round']}s  "
          f"ct_score={feat['ct_score']}  t_score={feat['t_score']}")

    # Count dims
    dim = 0
    for side in ["t", "ct"]:
        for i in range(5):
            dim += 25 + 6  # 25 scalars/bools + 6 weapon one-hot
    dim += 4   # global (ct_score, t_score, round_num, time_in_round)
    dim += 3   # bomb state (bomb_dropped, bomb_x, bomb_y)
    dim += 24  # utility slots
    dim += 7   # map one-hot
    print(f"\n  Total feature dimensions: {dim}")


def main():
    parser = argparse.ArgumentParser(
        description="Legacy preview for the old 348-dim exploratory feature schema.",
    )
    parser.add_argument(
        "pkl_path",
        nargs="?",
        default=str(data_path("viz", "2389983_de_dust2_full.pkl", prefer_existing=True)),
        help="Path to an extracted *_full.pkl file",
    )
    args = parser.parse_args()

    pkl_path = resolve_path_input(args.pkl_path)
    print(f"Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    tick_df = data["tick_df"]
    events = data["events"]
    round_info = data["round_info"]
    header = data["header"]
    map_name = header.get("map_name", "unknown")

    # Get all unique (round_num, tick) pairs
    valid = tick_df[tick_df["round_num"].notna()][["round_num", "tick"]].drop_duplicates()
    if len(valid) < 20:
        samples = valid
    else:
        samples = valid.sample(20, random_state=42).sort_values(["round_num", "tick"])

    print(f"Map: {map_name}, Rounds: {len(round_info)}, Sampling {len(samples)} ticks\n")

    for _, row in samples.iterrows():
        rn = int(row["round_num"])
        t = int(row["tick"])
        tick_slice = tick_df[(tick_df["tick"] == t) & (tick_df["round_num"] == rn)]
        feat = build_feature_vector(tick_df, tick_slice, events, t, rn, map_name, round_info)
        print_feature_vector(feat, t, rn)


if __name__ == "__main__":
    main()
