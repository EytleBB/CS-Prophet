"""Convert live CS2 GSI payloads into raw v2 feature rows."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from src.features.state_vector_v2 import (
    FEATURE_NAMES,
    MAPS,
    MOLOTOV_SLOTS,
    SMOKE_SLOTS,
    WEAPON_ID_MAP,
)
from src.utils.map_utils import classify_zone

_LIVE_ROUND_SECONDS: float = 115.0
_SMOKE_DURATION_SECONDS: float = 18.0
_MOLOTOV_DURATION_SECONDS: float = 7.0

_GSI_TO_CANONICAL_WEAPON: dict[str, str] = {
    "weapon_glock": "glock_18",
    "weapon_usp_silencer": "usp_s",
    "weapon_hkp2000": "p2000",
    "weapon_p2000": "p2000",
    "weapon_p250": "p250",
    "weapon_fiveseven": "five_seven",
    "weapon_tec9": "tec_9",
    "weapon_cz75a": "cz75_auto",
    "weapon_deagle": "desert_eagle",
    "weapon_revolver": "r8_revolver",
    "weapon_elite": "dual_berettas",
    "weapon_ak47": "ak_47",
    "weapon_m4a1": "m4a4",
    "weapon_m4a1_silencer": "m4a1_s",
    "weapon_m4a1_silencer_off": "m4a1_s",
    "weapon_famas": "famas",
    "weapon_galilar": "galil_ar",
    "weapon_sg556": "sg_553",
    "weapon_aug": "aug",
    "weapon_awp": "awp",
    "weapon_ssg08": "ssg_08",
    "weapon_scar20": "scar_20",
    "weapon_g3sg1": "g3sg1",
    "weapon_mp9": "mp9",
    "weapon_mp5sd": "mp5_sd",
    "weapon_ump45": "ump_45",
    "weapon_p90": "p90",
    "weapon_bizon": "pp_bizon",
    "weapon_mac10": "mac_10",
    "weapon_mp7": "mp7",
    "weapon_nova": "nova",
    "weapon_xm1014": "xm1014",
    "weapon_mag7": "mag_7",
    "weapon_sawedoff": "sawed_off",
    "weapon_m249": "m249",
    "weapon_negev": "negev",
}

_PRIMARY_WEAPONS: frozenset[str] = frozenset(
    name
    for name, canonical in _GSI_TO_CANONICAL_WEAPON.items()
    if canonical
    not in {
        "glock_18",
        "usp_s",
        "p2000",
        "p250",
        "five_seven",
        "tec_9",
        "cz75_auto",
        "desert_eagle",
        "r8_revolver",
        "dual_berettas",
    }
)
_SECONDARY_WEAPONS: frozenset[str] = frozenset(
    name for name in _GSI_TO_CANONICAL_WEAPON if name not in _PRIMARY_WEAPONS
)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_vec3(text: str | None) -> tuple[float, float, float]:
    if not text:
        return 0.0, 0.0, 0.0
    try:
        x_str, y_str, z_str = (part.strip() for part in str(text).split(","))
        return float(x_str), float(y_str), float(z_str)
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0


def _parse_yaw(forward_text: str | None) -> float:
    fx, fy, _ = _parse_vec3(forward_text)
    if fx == 0.0 and fy == 0.0:
        return 0.0
    return math.degrees(math.atan2(fy, fx))


def _weapon_entries(weapons: object) -> list[dict]:
    if isinstance(weapons, dict):
        return [item for item in weapons.values() if isinstance(item, dict)]
    return []


def _weapon_names(weapons: object) -> list[str]:
    names = []
    for item in _weapon_entries(weapons):
        name = item.get("name")
        if name:
            names.append(str(name))
    return names


def _best_weapon_id(weapons: object) -> int:
    primary = None
    secondary = None
    for name in _weapon_names(weapons):
        if name in _PRIMARY_WEAPONS:
            primary = name
        elif name in _SECONDARY_WEAPONS:
            secondary = name
    picked = primary or secondary
    if not picked:
        return WEAPON_ID_MAP["no_weapon"]
    canonical = _GSI_TO_CANONICAL_WEAPON.get(picked, "no_weapon")
    return WEAPON_ID_MAP.get(canonical, WEAPON_ID_MAP["no_weapon"])


def _has_grenade(weapons: object, grenade_name: str) -> float:
    return float(grenade_name in set(_weapon_names(weapons)))


def _has_molotov(weapons: object) -> float:
    names = set(_weapon_names(weapons))
    return float("weapon_molotov" in names or "weapon_incgrenade" in names)


def _has_c4(weapons: object) -> float:
    return float("weapon_c4" in set(_weapon_names(weapons)))


def _phase_time_elapsed(gsi: Mapping[str, object]) -> float:
    phase = str(gsi.get("round", {}).get("phase", "")).lower()
    countdowns = gsi.get("phase_countdowns", {})
    if phase != "live" or not isinstance(countdowns, Mapping):
        return 0.0
    ends_in = _safe_float(countdowns.get("phase_ends_in", 0.0))
    return max(0.0, _LIVE_ROUND_SECONDS - ends_in)


def _bomb_state(gsi: Mapping[str, object]) -> tuple[float, float, float]:
    bomb = gsi.get("bomb", {})
    if not isinstance(bomb, Mapping):
        return 0.0, 0.0, 0.0
    state = str(bomb.get("state", "")).lower()
    if state != "dropped":
        return 0.0, 0.0, 0.0
    x, y, _ = _parse_vec3(str(bomb.get("position", "")))
    return 1.0, x, y


def _parse_flame_positions(flames: object) -> list[tuple[float, float]]:
    if not isinstance(flames, Mapping):
        return []
    points: list[tuple[float, float]] = []
    for value in flames.values():
        x, y, _ = _parse_vec3(str(value))
        points.append((x, y))
    return points


def _centroid(points: Iterable[tuple[float, float]]) -> tuple[float, float]:
    pts = list(points)
    if not pts:
        return 0.0, 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _active_smokes(gsi: Mapping[str, object]) -> list[tuple[float, float, float]]:
    grenades = gsi.get("grenades", {})
    if not isinstance(grenades, Mapping):
        return []
    out: list[tuple[float, float, float]] = []
    for item in grenades.values():
        if not isinstance(item, Mapping) or str(item.get("type", "")).lower() != "smoke":
            continue
        effect_time = _safe_float(item.get("effecttime", 0.0))
        if effect_time <= 0.0:
            continue
        x, y, _ = _parse_vec3(str(item.get("position", "")))
        remain = max(0.0, 1.0 - effect_time / _SMOKE_DURATION_SECONDS)
        out.append((x, y, remain))
    out.sort(key=lambda rec: (-rec[2], rec[0], rec[1]))
    return out[:SMOKE_SLOTS]


def _active_molotovs(gsi: Mapping[str, object]) -> list[tuple[float, float, float]]:
    grenades = gsi.get("grenades", {})
    if not isinstance(grenades, Mapping):
        return []
    out: list[tuple[float, float, float]] = []
    for item in grenades.values():
        if not isinstance(item, Mapping) or str(item.get("type", "")).lower() != "inferno":
            continue
        x, y = _centroid(_parse_flame_positions(item.get("flames", {})))
        remain = max(0.0, 1.0 - _safe_float(item.get("lifetime", 0.0)) / _MOLOTOV_DURATION_SECONDS)
        out.append((x, y, remain))
    out.sort(key=lambda rec: (-rec[2], rec[0], rec[1]))
    return out[:MOLOTOV_SLOTS]


def build_row_from_gsi(
    gsi: dict,
    step: int,
    round_num: int,
    map_name: str,
    score_baseline: dict[str, int] | None = None,
) -> dict[str, float] | None:
    """Convert one merged GSI payload to a raw v2 feature row."""
    del step

    allplayers = gsi.get("allplayers", {})
    if not isinstance(allplayers, Mapping) or not allplayers:
        return None

    row = {name: 0.0 for name in FEATURE_NAMES}
    baseline = score_baseline or {}

    t_players: list[Mapping[str, object]] = []
    ct_players: list[Mapping[str, object]] = []
    for pdata in allplayers.values():
        if not isinstance(pdata, Mapping):
            continue
        team = str(pdata.get("team", ""))
        if team == "T":
            t_players.append(pdata)
        elif team == "CT":
            ct_players.append(pdata)

    t_players.sort(key=lambda player: str(player.get("name", "")))
    ct_players.sort(key=lambda player: str(player.get("name", "")))

    for side_name, players in (("t", t_players), ("ct", ct_players)):
        for idx in range(5):
            if idx >= len(players):
                continue

            prefix = f"{side_name}{idx}"
            player = players[idx]
            player_name = str(player.get("name", ""))
            state = player.get("state", {})
            state = state if isinstance(state, Mapping) else {}
            stats = player.get("match_stats", {})
            stats = stats if isinstance(stats, Mapping) else {}
            weapons = player.get("weapons", {})
            weapons = weapons if isinstance(weapons, Mapping) else {}

            x, y, z = _parse_vec3(str(player.get("position", "")))
            hp = _safe_int(state.get("health", 0))
            alive = float(hp > 0)

            row[f"{prefix}_x"] = x
            row[f"{prefix}_y"] = y
            row[f"{prefix}_z"] = z
            row[f"{prefix}_yaw"] = _parse_yaw(player.get("forward")) if alive else 0.0
            row[f"{prefix}_in_bomb_zone"] = float(
                alive and classify_zone(x, y, map_name, z) in {"A", "B"}
            )
            row[f"{prefix}_hp"] = float(hp)
            row[f"{prefix}_armor"] = float(_safe_int(state.get("armor", 0)))
            row[f"{prefix}_helmet"] = float(bool(state.get("helmet", False)))
            row[f"{prefix}_alive"] = alive
            row[f"{prefix}_has_smoke"] = _has_grenade(weapons, "weapon_smokegrenade")
            row[f"{prefix}_has_flash"] = _has_grenade(weapons, "weapon_flashbang")
            row[f"{prefix}_has_he"] = _has_grenade(weapons, "weapon_hegrenade")
            row[f"{prefix}_has_molotov"] = _has_molotov(weapons)
            row[f"{prefix}_has_c4"] = _has_c4(weapons)
            row[f"{prefix}_balance"] = float(_safe_int(state.get("money", 0)))
            row[f"{prefix}_equip_value"] = float(_safe_int(state.get("equip_value", 0)))
            row[f"{prefix}_score"] = float(
                max(0, _safe_int(stats.get("score", 0)) - baseline.get(player_name, 0))
            )
            row[f"{prefix}_weapon_id"] = float(_best_weapon_id(weapons))

    map_info = gsi.get("map", {})
    map_info = map_info if isinstance(map_info, Mapping) else {}
    team_ct = map_info.get("team_ct", {})
    team_t = map_info.get("team_t", {})
    team_ct = team_ct if isinstance(team_ct, Mapping) else {}
    team_t = team_t if isinstance(team_t, Mapping) else {}
    row["ct_score"] = float(_safe_int(team_ct.get("score", 0)))
    row["t_score"] = float(_safe_int(team_t.get("score", 0)))
    row["round_num"] = float(round_num)
    row["time_in_round"] = _phase_time_elapsed(gsi)

    bomb_dropped, bomb_x, bomb_y = _bomb_state(gsi)
    row["bomb_dropped"] = bomb_dropped
    row["bomb_x"] = bomb_x
    row["bomb_y"] = bomb_y

    smokes = _active_smokes(gsi)
    for slot, (x, y, remain) in enumerate(smokes):
        row[f"smoke{slot}_x"] = x
        row[f"smoke{slot}_y"] = y
        row[f"smoke{slot}_remain"] = remain

    molotovs = _active_molotovs(gsi)
    for slot, (x, y, remain) in enumerate(molotovs):
        row[f"molotov{slot}_x"] = x
        row[f"molotov{slot}_y"] = y
        row[f"molotov{slot}_remain"] = remain

    for candidate in MAPS:
        row[f"map_{candidate}"] = float(candidate == map_name)

    return row
