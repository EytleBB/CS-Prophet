"""Demo parser — reads CS2 .dem files and writes per-round state sequences (parquet).

Self-contained version for portable deployment. No project-root imports.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .label_extractor import extract_bomb_site, get_plant_ticks
from .map_utils import normalize_coords

try:
    from demoparser2 import DemoParser
except ImportError:
    DemoParser = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

TICK_RATE: int = 64
TARGET_RATE: int = 8
DOWNSAMPLE: int = TICK_RATE // TARGET_RATE
POST_START_SECS: int = 90
MAX_STEPS: int = POST_START_SECS * TARGET_RATE

PLAYER_PROPS: list[str] = [
    "X", "Y", "Z",
    "health", "armor_value", "has_helmet", "is_alive",
    "team_name", "name", "weapon_name", "inventory",
    "flash_duration", "equipment_value_this_round",
    "is_scoped", "is_defusing",
    "ct_losing_streak", "t_losing_streak",
]

_TEAM_T = "TERRORIST"
_TEAM_CT = "CT"

_WEAPON_CAT_MAP: dict[str, str] = {}
_WEAPON_CATEGORIES: dict[str, set[str]] = {
    "pistol":  {"Glock-18", "USP-S", "P2000", "P250", "Five-SeveN", "Tec-9",
                "CZ75-Auto", "Desert Eagle", "R8 Revolver", "Dual Berettas"},
    "rifle":   {"AK-47", "M4A4", "M4A1-S", "FAMAS", "Galil AR", "SG 553",
                "AUG", "SCAR-20", "G3SG1"},
    "sniper":  {"AWP", "SSG 08"},
    "smg":     {"MP9", "MP5-SD", "UMP-45", "P90", "PP-Bizon", "MAC-10", "MP7"},
    "heavy":   {"Nova", "XM1014", "MAG-7", "Sawed-Off", "M249", "Negev"},
    "grenade": {"HE Grenade", "Flashbang", "Smoke Grenade", "Molotov",
                "Incendiary Grenade", "Decoy Grenade"},
}
for _cat, _names in _WEAPON_CATEGORIES.items():
    for _n in _names:
        _WEAPON_CAT_MAP[_n] = _cat


def parse_demo(
    dem_path: Path | str,
    output_dir: Path | str,
    player_roles: Optional[dict[str, str]] = None,
) -> Optional[Path]:
    dem_path = Path(dem_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Parsing demo: %s", dem_path.name)

    if DemoParser is None:
        raise ImportError("demoparser2 is not installed -- run: pip install demoparser2")

    parser = DemoParser(str(dem_path))

    try:
        plant_df = parser.parse_event("bomb_planted", player=["X", "Y", "Z"])
    except Exception:
        logger.exception("parse_event('bomb_planted') failed in %s", dem_path.name)
        return None

    if plant_df is None or plant_df.empty:
        logger.warning("No bomb_planted events in %s -- skipping", dem_path.name)
        return None

    map_name = _get_map_name(parser)
    sites = extract_bomb_site(plant_df, map_name=map_name)
    plant_ticks = get_plant_ticks(plant_df)

    try:
        round_end_df = parser.parse_event("round_end")
    except Exception:
        round_end_df = None

    freeze_end_ticks: list[int] = []
    for event_name in ("round_freeze_end", "round_start"):
        try:
            ev_df = parser.parse_event(event_name)
        except Exception:
            ev_df = None
        if ev_df is not None and not ev_df.empty and "tick" in ev_df.columns:
            freeze_end_ticks = sorted(ev_df["tick"].astype(int).tolist())
            break

    sequences: list[pd.DataFrame] = []
    for round_num, (plant_tick, bomb_site) in enumerate(
        zip(plant_ticks, sites), start=1
    ):
        if bomb_site not in ("A", "B"):
            continue

        action_tick = 0
        for t in freeze_end_ticks:
            if t <= int(plant_tick):
                action_tick = t
            else:
                break

        if action_tick == 0 and freeze_end_ticks:
            continue

        ct_score, t_score = _scores_before_tick(round_end_df, int(plant_tick))
        seq = _extract_sequence(
            parser, round_num, action_tick, int(plant_tick), bomb_site, map_name,
            player_roles, ct_score, t_score,
        )
        if seq is not None:
            sequences.append(seq)

    if not sequences:
        logger.warning("No valid sequences extracted from %s", dem_path.name)
        return None

    result = pd.concat(sequences, ignore_index=True)
    result.insert(0, "demo_name", dem_path.stem)

    out_path = output_dir / f"{dem_path.stem}.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Saved %d rows -> %s", len(result), out_path)
    return out_path


def _get_map_name(parser) -> str:
    try:
        return parser.parse_header().get("map_name", "")
    except Exception:
        return ""


def _scores_before_tick(round_end_df, tick: int) -> tuple[int, int]:
    if round_end_df is None or round_end_df.empty or "winner" not in round_end_df.columns:
        return 0, 0
    prior = round_end_df[round_end_df["tick"] < tick]
    return int((prior["winner"] == "CT").sum()), int((prior["winner"] == "T").sum())


def _get_weapon_cat(weapon_name) -> str:
    if weapon_name is None:
        return "other"
    try:
        if isinstance(weapon_name, float) and np.isnan(weapon_name):
            return "other"
    except (TypeError, ValueError):
        pass
    return _WEAPON_CAT_MAP.get(str(weapon_name), "other")


def _parse_nade_inventory(inventory) -> tuple[bool, bool, bool, bool]:
    if not isinstance(inventory, (list, np.ndarray)):
        return False, False, False, False
    inv = set(inventory)
    return (
        "Smoke Grenade" in inv,
        "Flashbang" in inv,
        "HE Grenade" in inv,
        "Molotov" in inv or "Incendiary Grenade" in inv,
    )


def _extract_sequence(
    parser, round_num: int, round_start_tick: int, plant_tick: int,
    bomb_site: str, map_name: str,
    player_roles: Optional[dict[str, str]] = None,
    ct_score: int = 0, t_score: int = 0,
) -> Optional[pd.DataFrame]:
    end_tick = round_start_tick + POST_START_SECS * TICK_RATE
    end_tick = min(end_tick, plant_tick)
    wanted_ticks = list(range(round_start_tick, end_tick + 1, DOWNSAMPLE))

    try:
        tick_df = parser.parse_ticks(PLAYER_PROPS, ticks=wanted_ticks)
    except Exception:
        logger.exception("parse_ticks failed for round %d", round_num)
        return None

    if tick_df is None or tick_df.empty:
        return None

    rows: list[dict] = []
    for step, tick in enumerate(wanted_ticks):
        tick_slice = tick_df[tick_df["tick"] == tick]
        if tick_slice.empty:
            continue
        rows.append(
            _build_state_row(
                tick_slice, step, tick, round_num, bomb_site,
                map_name, player_roles, ct_score, t_score,
            )
        )

    return pd.DataFrame(rows) if rows else None


def _build_state_row(
    tick_slice: pd.DataFrame, step: int, tick: int, round_num: int,
    bomb_site: str, map_name: str,
    player_roles: Optional[dict[str, str]] = None,
    ct_score: int = 0, t_score: int = 0,
) -> dict:
    t_rows = tick_slice[tick_slice["team_name"] == _TEAM_T].sort_values("name")
    ct_rows = tick_slice[tick_slice["team_name"] == _TEAM_CT].sort_values("name")

    first = tick_slice.iloc[0] if not tick_slice.empty else pd.Series(dtype=object)
    ct_streak = int(first.get("ct_losing_streak", 0)) if not tick_slice.empty else 0
    t_streak  = int(first.get("t_losing_streak",  0)) if not tick_slice.empty else 0

    row: dict = {
        "round_num": round_num, "step": step, "tick": tick,
        "bomb_site": bomb_site,
        "ct_score": ct_score, "t_score": t_score,
        "ct_losing_streak": ct_streak, "t_losing_streak": t_streak,
    }

    for side_prefix, side_rows in (("t", t_rows), ("ct", ct_rows)):
        for i in range(5):
            prefix = f"{side_prefix}{i}"
            if i < len(side_rows):
                p = side_rows.iloc[i]
                x_n, y_n, z_n = normalize_coords(
                    float(p.get("X", 0.0)), float(p.get("Y", 0.0)),
                    float(p.get("Z", 0.0)), map_name,
                )
                inv = p.get("inventory", [])
                if not isinstance(inv, list):
                    inv = []
                has_smoke, has_flash, has_he, has_molotov = _parse_nade_inventory(inv)
                name = str(p.get("name", ""))

                row[f"{prefix}_x"]              = x_n
                row[f"{prefix}_y"]              = y_n
                row[f"{prefix}_z"]              = z_n
                row[f"{prefix}_hp"]             = int(p.get("health", 0))
                row[f"{prefix}_armor"]          = int(p.get("armor_value", 0))
                row[f"{prefix}_helmet"]         = bool(p.get("has_helmet", False))
                row[f"{prefix}_alive"]          = bool(p.get("is_alive", False))
                row[f"{prefix}_role"]           = player_roles.get(name, "") if player_roles else ""
                row[f"{prefix}_weapon"]         = _get_weapon_cat(p.get("weapon_name"))
                row[f"{prefix}_has_smoke"]      = has_smoke
                row[f"{prefix}_has_flash"]      = has_flash
                row[f"{prefix}_has_he"]         = has_he
                row[f"{prefix}_has_molotov"]    = has_molotov
                row[f"{prefix}_flash_duration"] = float(p.get("flash_duration", 0.0))
                row[f"{prefix}_equip_value"]    = int(p.get("equipment_value_this_round", 0))
                row[f"{prefix}_is_scoped"]      = bool(p.get("is_scoped", False))
                row[f"{prefix}_is_defusing"]    = bool(p.get("is_defusing", False))
            else:
                row[f"{prefix}_x"]              = 0.0
                row[f"{prefix}_y"]              = 0.0
                row[f"{prefix}_z"]              = 0.0
                row[f"{prefix}_hp"]             = 0
                row[f"{prefix}_armor"]          = 0
                row[f"{prefix}_helmet"]         = False
                row[f"{prefix}_alive"]          = False
                row[f"{prefix}_role"]           = ""
                row[f"{prefix}_weapon"]         = "other"
                row[f"{prefix}_has_smoke"]      = False
                row[f"{prefix}_has_flash"]      = False
                row[f"{prefix}_has_he"]         = False
                row[f"{prefix}_has_molotov"]    = False
                row[f"{prefix}_flash_duration"] = 0.0
                row[f"{prefix}_equip_value"]    = 0
                row[f"{prefix}_is_scoped"]      = False
                row[f"{prefix}_is_defusing"]    = False

    return row
