"""Build canonical v2 feature rows from extracted demo data."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.label_extractor import extract_bomb_site
from src.utils.map_utils import classify_zone
from src.features.state_vector_v2 import (
    FEATURE_NAMES,
    MAPS,
    MOLOTOV_SLOTS,
    SMOKE_SLOTS,
    WEAPON_ID_MAP,
)

# Map from demoparser2 weapon display name -> canonical weapon ID in WEAPONS tuple.
WEAPON_NAME_TO_ID: dict[str, str] = {
    # Pistols
    "Glock-18": "glock_18", "USP-S": "usp_s", "P2000": "p2000",
    "P250": "p250", "Five-SeveN": "five_seven", "Tec-9": "tec_9",
    "CZ75-Auto": "cz75_auto", "Desert Eagle": "desert_eagle",
    "R8 Revolver": "r8_revolver", "Dual Berettas": "dual_berettas",
    # Rifles
    "AK-47": "ak_47", "M4A4": "m4a4", "M4A1-S": "m4a1_s",
    "FAMAS": "famas", "Galil AR": "galil_ar", "SG 553": "sg_553", "AUG": "aug",
    # Snipers
    "AWP": "awp", "SSG 08": "ssg_08", "SCAR-20": "scar_20", "G3SG1": "g3sg1",
    # SMGs
    "MP9": "mp9", "MP5-SD": "mp5_sd", "UMP-45": "ump_45", "P90": "p90",
    "PP-Bizon": "pp_bizon", "MAC-10": "mac_10", "MP7": "mp7",
    # Heavy
    "Nova": "nova", "XM1014": "xm1014", "MAG-7": "mag_7",
    "Sawed-Off": "sawed_off", "M249": "m249", "Negev": "negev",
}

PRIMARY_WEAPONS: frozenset[str] = frozenset(
    k for k, v in WEAPON_NAME_TO_ID.items()
    if v not in {"glock_18", "usp_s", "p2000", "p250", "five_seven", "tec_9",
                 "cz75_auto", "desert_eagle", "r8_revolver", "dual_berettas"}
)
SECONDARY_WEAPONS: frozenset[str] = frozenset(
    k for k, v in WEAPON_NAME_TO_ID.items() if k not in PRIMARY_WEAPONS
)

SMOKE_DURATION_TICKS: int = 18 * 64
MOLOTOV_DURATION_TICKS: int = 7 * 64


def _safe_float(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _safe_int(value, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(value)


def _safe_bool(value) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _inventory_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    return []


def best_weapon(inventory) -> str:
    """Pick best weapon from inventory using primary > secondary priority."""
    items = _inventory_list(inventory)
    primary = None
    secondary = None
    for item in items:
        if item in PRIMARY_WEAPONS:
            primary = item
        elif item in SECONDARY_WEAPONS:
            secondary = item
    return primary or secondary or "none"


def weapon_id(best_weapon_name: str) -> int:
    """Return integer weapon ID for embedding lookup."""
    canonical = WEAPON_NAME_TO_ID.get(best_weapon_name, "no_weapon")
    return WEAPON_ID_MAP.get(canonical, WEAPON_ID_MAP["no_weapon"])


def has_c4(inventory) -> bool:
    return "C4 Explosive" in _inventory_list(inventory)


def map_onehot(map_name: str) -> tuple[float, ...]:
    return tuple(1.0 if map_name == candidate else 0.0 for candidate in MAPS)


def build_round_label_map(events: dict, map_name: str) -> dict[int, str]:
    """Return round_num -> {'A','B','other'} for bomb-planted rounds."""
    planted_df = events.get("bomb_planted", pd.DataFrame())
    if planted_df.empty or "round_num" not in planted_df.columns:
        return {}

    labels = extract_bomb_site(planted_df, map_name=map_name)
    out: dict[int, str] = {}
    for (_, row), label in zip(planted_df.iterrows(), labels, strict=False):
        if pd.isna(row.get("round_num")):
            continue
        out[int(row["round_num"])] = str(label)
    return out


def _event_xy(row: pd.Series) -> tuple[float, float] | None:
    for x_col, y_col in (("X", "Y"), ("user_X", "user_Y")):
        if x_col in row.index and y_col in row.index and pd.notna(row.get(x_col)) and pd.notna(row.get(y_col)):
            return _safe_float(row[x_col]), _safe_float(row[y_col])
    return None


def _lookup_player_xy_at_tick(
    tick_df: pd.DataFrame,
    round_num: int,
    player_name: str,
    target_tick: int,
) -> tuple[float, float] | None:
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


def get_bomb_state(
    tick_df: pd.DataFrame,
    events: dict,
    round_info: pd.DataFrame,
    current_tick: int,
    current_round: int,
) -> tuple[bool, float, float]:
    """Return (bomb_dropped, bomb_x, bomb_y) at a specific tick."""
    drop_df = events.get("bomb_dropped", pd.DataFrame())
    pick_df = events.get("bomb_pickup", pd.DataFrame())
    ri = round_info[round_info["round_num"] == current_round]
    round_start = _safe_int(ri["freeze_tick"].iloc[0]) if not ri.empty else 0
    round_end = _safe_int(ri["end_tick"].iloc[0], current_tick) if not ri.empty and pd.notna(ri["end_tick"].iloc[0]) else current_tick

    timeline: list[tuple[int, int, str, pd.Series]] = []
    if not drop_df.empty and "tick" in drop_df.columns:
        rnd = drop_df[(drop_df["tick"] >= round_start) & (drop_df["tick"] <= round_end)]
        for _, row in rnd.iterrows():
            tick = _safe_int(row["tick"])
            if tick <= current_tick:
                timeline.append((tick, 0, "drop", row))

    if not pick_df.empty and "tick" in pick_df.columns:
        rnd = pick_df[(pick_df["tick"] >= round_start) & (pick_df["tick"] <= round_end)]
        for _, row in rnd.iterrows():
            tick = _safe_int(row["tick"])
            if tick <= current_tick:
                timeline.append((tick, 1, "pick", row))

    if not timeline:
        return False, 0.0, 0.0

    timeline.sort(key=lambda item: (item[0], item[1]))
    last_tick, _, event_type, event_row = timeline[-1]
    if event_type == "pick":
        return False, 0.0, 0.0

    event_pos = _event_xy(event_row)
    if event_pos is not None:
        return True, event_pos[0], event_pos[1]

    player_name = str(event_row.get("user_name", ""))
    pos = _lookup_player_xy_at_tick(tick_df, current_round, player_name, last_tick)
    if pos is not None:
        return True, pos[0], pos[1]

    return True, 0.0, 0.0


def get_active_utils(
    events: dict,
    current_tick: int,
    current_round: int,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Return active smokes and molotovs with remaining-time ratio."""
    smokes: list[tuple[float, float, float]] = []
    molotovs: list[tuple[float, float, float]] = []

    smoke_df = events.get("smokegrenade_detonate", pd.DataFrame())
    if not smoke_df.empty and "round_num" in smoke_df.columns:
        rnd = smoke_df[smoke_df["round_num"] == current_round]
        for _, row in rnd.iterrows():
            det_tick = _safe_int(row["tick"])
            elapsed = current_tick - det_tick
            if 0 <= elapsed < SMOKE_DURATION_TICKS:
                remain = 1.0 - elapsed / SMOKE_DURATION_TICKS
                smokes.append((_safe_float(row["x"]), _safe_float(row["y"]), remain))

    inferno_df = events.get("inferno_startburn", pd.DataFrame())
    if not inferno_df.empty and "round_num" in inferno_df.columns:
        rnd = inferno_df[inferno_df["round_num"] == current_round]
        for _, row in rnd.iterrows():
            det_tick = _safe_int(row["tick"])
            elapsed = current_tick - det_tick
            if 0 <= elapsed < MOLOTOV_DURATION_TICKS:
                remain = 1.0 - elapsed / MOLOTOV_DURATION_TICKS
                molotovs.append((_safe_float(row["x"]), _safe_float(row["y"]), remain))

    return smokes, molotovs


def build_feature_row_v2(
    tick_df: pd.DataFrame,
    tick_slice: pd.DataFrame,
    events: dict,
    tick: int,
    round_num: int,
    map_name: str,
    round_info: pd.DataFrame,
) -> dict[str, float]:
    """Build one canonical raw feature row matching state_vector_v2.FEATURE_NAMES."""
    row = {name: 0.0 for name in FEATURE_NAMES}

    t_rows = tick_slice[tick_slice["team_name"] == "TERRORIST"].sort_values("name").reset_index(drop=True)
    ct_rows = tick_slice[tick_slice["team_name"] == "CT"].sort_values("name").reset_index(drop=True)

    for side_name, side_rows in (("t", t_rows), ("ct", ct_rows)):
        for idx in range(5):
            prefix = f"{side_name}{idx}"
            if idx >= len(side_rows):
                continue

            player = side_rows.iloc[idx]
            alive = _safe_bool(player.get("is_alive", False))
            inventory = _inventory_list(player.get("inventory", []))
            best = best_weapon(inventory)
            wid = weapon_id(best)

            row[f"{prefix}_x"] = _safe_float(player.get("X", 0.0))
            row[f"{prefix}_y"] = _safe_float(player.get("Y", 0.0))
            row[f"{prefix}_z"] = _safe_float(player.get("Z", 0.0))
            row[f"{prefix}_yaw"] = _safe_float(player.get("yaw", 0.0)) if alive else 0.0
            row[f"{prefix}_in_bomb_zone"] = float(
                alive
                and classify_zone(
                    row[f"{prefix}_x"],
                    row[f"{prefix}_y"],
                    map_name,
                    row[f"{prefix}_z"],
                )
                in {"A", "B"}
            )
            row[f"{prefix}_hp"] = float(_safe_int(player.get("health", 0)))
            row[f"{prefix}_armor"] = float(_safe_int(player.get("armor_value", 0)))
            row[f"{prefix}_helmet"] = float(_safe_bool(player.get("has_helmet", False)))
            row[f"{prefix}_alive"] = float(alive)
            row[f"{prefix}_has_smoke"] = float("Smoke Grenade" in inventory)
            row[f"{prefix}_has_flash"] = float("Flashbang" in inventory)
            row[f"{prefix}_has_he"] = float(any(item in inventory for item in ("HE Grenade", "High Explosive Grenade")))
            row[f"{prefix}_has_molotov"] = float(any(item in inventory for item in ("Molotov", "Incendiary Grenade")))
            row[f"{prefix}_has_c4"] = float(has_c4(inventory))
            row[f"{prefix}_balance"] = float(_safe_int(player.get("balance", 0)))
            row[f"{prefix}_equip_value"] = float(_safe_int(player.get("current_equip_value", 0)))
            row[f"{prefix}_score"] = float(_safe_int(player.get("score", 0)))
            row[f"{prefix}_weapon_id"] = float(wid)

    ri = round_info[round_info["round_num"] == round_num]
    freeze_tick = _safe_int(ri["freeze_tick"].iloc[0], tick) if not ri.empty else tick
    row["ct_score"] = float(_safe_int(ri["ct_score"].iloc[0])) if not ri.empty else 0.0
    row["t_score"] = float(_safe_int(ri["t_score"].iloc[0])) if not ri.empty else 0.0
    row["round_num"] = float(round_num)
    row["time_in_round"] = (tick - freeze_tick) / 64.0

    bomb_dropped, bomb_x, bomb_y = get_bomb_state(tick_df, events, round_info, tick, round_num)
    row["bomb_dropped"] = float(bomb_dropped)
    row["bomb_x"] = bomb_x
    row["bomb_y"] = bomb_y

    smokes, molotovs = get_active_utils(events, tick, round_num)
    for slot in range(SMOKE_SLOTS):
        if slot < len(smokes):
            row[f"smoke{slot}_x"] = smokes[slot][0]
            row[f"smoke{slot}_y"] = smokes[slot][1]
            row[f"smoke{slot}_remain"] = smokes[slot][2]
    for slot in range(MOLOTOV_SLOTS):
        if slot < len(molotovs):
            row[f"molotov{slot}_x"] = molotovs[slot][0]
            row[f"molotov{slot}_y"] = molotovs[slot][1]
            row[f"molotov{slot}_remain"] = molotovs[slot][2]

    for idx, candidate in enumerate(MAPS):
        row[f"map_{candidate}"] = map_onehot(map_name)[idx]

    return row


def build_round_feature_frame(
    data: dict,
    round_num: int,
) -> pd.DataFrame:
    """Build a canonical raw feature frame for one round from an extracted full-pkl payload."""
    tick_df = data["tick_df"]
    events = data["events"]
    round_info = data["round_info"]
    header = data["header"]
    map_name = header.get("map_name", "unknown")

    valid = tick_df[tick_df["round_num"] == round_num][["round_num", "tick"]].drop_duplicates().sort_values("tick")
    rows = []
    for _, item in valid.iterrows():
        tick = _safe_int(item["tick"])
        tick_slice = tick_df[(tick_df["round_num"] == round_num) & (tick_df["tick"] == tick)]
        rows.append(build_feature_row_v2(tick_df, tick_slice, events, tick, round_num, map_name, round_info))
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def iter_round_ticks(tick_df: pd.DataFrame) -> Iterator[tuple[int, int]]:
    """Yield (round_num, tick) in sorted order for unique ticks in tick_df."""
    unique_ticks = tick_df[tick_df["round_num"].notna()][["round_num", "tick"]].drop_duplicates()
    unique_ticks = unique_ticks.sort_values(["round_num", "tick"])
    for _, row in unique_ticks.iterrows():
        yield _safe_int(row["round_num"]), _safe_int(row["tick"])
