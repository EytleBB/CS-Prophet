"""Convert live memory-reader payloads into raw v2 feature rows."""

from __future__ import annotations

from collections.abc import Mapping

from src.features.state_vector_v2 import (
    FEATURE_NAMES,
    MAPS,
    MOLOTOV_SLOTS,
    SMOKE_SLOTS,
    WEAPON_ID_MAP,
)
from src.inference.gsi_state_builder import _GSI_TO_CANONICAL_WEAPON
from src.utils.map_utils import classify_zone

_NO_WEAPON_ID = float(WEAPON_ID_MAP["no_weapon"])
_WEAPON_PRICES: dict[str, float] = {
    "ak_47": 2700.0,
    "m4a4": 3100.0,
    "m4a1_s": 2900.0,
    "awp": 4750.0,
    "aug": 3300.0,
    "sg_553": 3000.0,
    "famas": 2050.0,
    "galil_ar": 1800.0,
    "ssg_08": 1700.0,
    "scar_20": 5000.0,
    "g3sg1": 5000.0,
    "desert_eagle": 700.0,
    "glock_18": 200.0,
    "usp_s": 200.0,
    "p2000": 200.0,
    "p250": 300.0,
    "five_seven": 500.0,
    "tec_9": 500.0,
    "cz75_auto": 500.0,
    "r8_revolver": 600.0,
    "dual_berettas": 300.0,
    "mp9": 1250.0,
    "mp7": 1500.0,
    "mp5_sd": 1500.0,
    "ump_45": 1200.0,
    "p90": 2350.0,
    "pp_bizon": 1400.0,
    "mac_10": 1050.0,
    "nova": 1050.0,
    "xm1014": 2000.0,
    "mag_7": 1300.0,
    "sawed_off": 1100.0,
    "m249": 2250.0,
    "negev": 1700.0,
    "weapon_smokegrenade": 300.0,
    "weapon_flashbang": 200.0,
    "weapon_hegrenade": 300.0,
    "weapon_molotov": 400.0,
    "weapon_incgrenade": 600.0,
    "weapon_decoy": 50.0,
}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool01(value: object) -> float:
    return float(bool(value))


def _player_name(player: Mapping[str, object]) -> str:
    return str(player.get("name", ""))


def _weapon_list(player: Mapping[str, object]) -> list[str]:
    weapons = player.get("weapons", [])
    if isinstance(weapons, list):
        return [str(item) for item in weapons if item]
    return []


def _active_weapon_id(player: Mapping[str, object]) -> float:
    active_weapon = str(player.get("active_weapon_class", "") or "")
    if not active_weapon:
        return _NO_WEAPON_ID
    if active_weapon in WEAPON_ID_MAP:
        return float(WEAPON_ID_MAP[active_weapon])
    canonical = _GSI_TO_CANONICAL_WEAPON.get(active_weapon)
    if not canonical:
        return _NO_WEAPON_ID
    return float(WEAPON_ID_MAP.get(canonical, WEAPON_ID_MAP["no_weapon"]))


def _weapon_price(class_name: str) -> float:
    if class_name in _WEAPON_PRICES:
        return _WEAPON_PRICES[class_name]
    canonical = _GSI_TO_CANONICAL_WEAPON.get(class_name)
    if canonical is None:
        return 0.0
    return _WEAPON_PRICES.get(canonical, 0.0)


def _equip_value(player: Mapping[str, object]) -> float:
    return sum(_weapon_price(class_name) for class_name in _weapon_list(player))


def _inventory_flag(
    player: Mapping[str, object],
    *,
    weapons: set[str],
    key: str,
    class_names: tuple[str, ...],
) -> float:
    return float(bool(player.get(key, False)) or any(name in weapons for name in class_names))


def _projectile_rows(projectiles: object) -> list[tuple[float, float, float]]:
    if not isinstance(projectiles, list):
        return []

    parsed: list[tuple[float, float, float]] = []
    for item in projectiles:
        if isinstance(item, Mapping):
            x = _safe_float(item.get("x", 0.0))
            y = _safe_float(item.get("y", 0.0))
            remain = _safe_float(item.get("remain", 0.0))
        elif isinstance(item, tuple | list) and len(item) >= 3:
            x = _safe_float(item[0])
            y = _safe_float(item[1])
            remain = _safe_float(item[2])
        else:
            continue
        if remain <= 0.0:
            continue
        parsed.append((x, y, remain))

    parsed.sort(key=lambda item: (-item[2], item[0], item[1]))
    return parsed


def build_row_from_memory(
    players: list[dict],
    map_state: dict,
    round_num: int,
    map_name: str,
    score_baseline: dict[int, int] | None = None,
) -> dict[str, float] | None:
    """Build a raw, unnormalized v2 feature row from live process memory."""
    del score_baseline

    if not players or not map_name or map_name not in MAPS:
        return None

    row = {name: 0.0 for name in FEATURE_NAMES}
    bomb = map_state.get("bomb", {})
    bomb = bomb if isinstance(bomb, Mapping) else {}
    projectiles = map_state.get("projectiles", {})
    projectiles = projectiles if isinstance(projectiles, Mapping) else {}

    t_players = [player for player in players if str(player.get("team", "")) == "T"]
    ct_players = [player for player in players if str(player.get("team", "")) == "CT"]
    t_players.sort(key=_player_name)
    ct_players.sort(key=_player_name)

    for side_name, side_players in (("t", t_players), ("ct", ct_players)):
        for idx in range(5):
            if idx >= len(side_players):
                continue

            prefix = f"{side_name}{idx}"
            player = side_players[idx]
            weapons = set(_weapon_list(player))
            x = _safe_float(player.get("x", 0.0))
            y = _safe_float(player.get("y", 0.0))
            z = _safe_float(player.get("z", 0.0))
            hp = max(0.0, _safe_float(player.get("hp", 0.0)))
            alive = float(bool(player.get("alive", False)) and hp > 0.0)

            row[f"{prefix}_x"] = x
            row[f"{prefix}_y"] = y
            row[f"{prefix}_z"] = z
            row[f"{prefix}_yaw"] = _safe_float(player.get("yaw", 0.0)) if alive else 0.0
            row[f"{prefix}_in_bomb_zone"] = float(
                alive and classify_zone(x, y, map_name, z) in {"A", "B"}
            )
            row[f"{prefix}_hp"] = hp
            row[f"{prefix}_armor"] = max(0.0, _safe_float(player.get("armor", 0.0)))
            row[f"{prefix}_helmet"] = _safe_bool01(player.get("helmet", False))
            row[f"{prefix}_alive"] = alive
            row[f"{prefix}_has_smoke"] = _inventory_flag(
                player,
                weapons=weapons,
                key="has_smoke",
                class_names=("weapon_smokegrenade",),
            )
            row[f"{prefix}_has_flash"] = _inventory_flag(
                player,
                weapons=weapons,
                key="has_flash",
                class_names=("weapon_flashbang",),
            )
            row[f"{prefix}_has_he"] = _inventory_flag(
                player,
                weapons=weapons,
                key="has_he",
                class_names=("weapon_hegrenade",),
            )
            row[f"{prefix}_has_molotov"] = _inventory_flag(
                player,
                weapons=weapons,
                key="has_molotov",
                class_names=("weapon_molotov", "weapon_incgrenade"),
            )
            row[f"{prefix}_has_c4"] = _inventory_flag(
                player,
                weapons=weapons,
                key="has_c4",
                class_names=("weapon_c4",),
            )
            row[f"{prefix}_balance"] = max(0.0, _safe_float(player.get("money", 0.0)))
            row[f"{prefix}_equip_value"] = _equip_value(player)
            row[f"{prefix}_score"] = 0.0
            row[f"{prefix}_weapon_id"] = _active_weapon_id(player)

    row["ct_score"] = max(0.0, _safe_float(map_state.get("ct_score", 0.0)))
    row["t_score"] = max(0.0, _safe_float(map_state.get("t_score", 0.0)))
    row["round_num"] = float(round_num)
    row["time_in_round"] = max(0.0, _safe_float(map_state.get("time_in_round", 0.0)))
    row["bomb_dropped"] = _safe_bool01(bomb.get("dropped", False))
    if bool(bomb.get("dropped", False)) or bool(bomb.get("planted", False)):
        row["bomb_x"] = _safe_float(bomb.get("x", 0.0))
        row["bomb_y"] = _safe_float(bomb.get("y", 0.0))

    smokes = _projectile_rows(projectiles.get("smokes", []))
    for slot, (x, y, remain) in enumerate(smokes[:SMOKE_SLOTS]):
        row[f"smoke{slot}_x"] = x
        row[f"smoke{slot}_y"] = y
        row[f"smoke{slot}_remain"] = remain

    molotovs = _projectile_rows(projectiles.get("molotovs", []))
    for slot, (x, y, remain) in enumerate(molotovs[:MOLOTOV_SLOTS]):
        row[f"molotov{slot}_x"] = x
        row[f"molotov{slot}_y"] = y
        row[f"molotov{slot}_remain"] = remain

    for candidate in MAPS:
        row[f"map_{candidate}"] = float(candidate == map_name)

    return row
