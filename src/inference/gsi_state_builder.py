"""Convert CS2 GSI JSON payload to a row dict compatible with build_state_vector()."""

from __future__ import annotations

from src.utils.map_utils import classify_zone, normalize_coords

# GSI weapon name (weapon_xxx) → our training category
_WEAPON_CATS: dict[str, str] = {
    "weapon_glock": "pistol", "weapon_usp_silencer": "pistol",
    "weapon_p2000": "pistol", "weapon_p250": "pistol",
    "weapon_fiveseven": "pistol", "weapon_tec9": "pistol",
    "weapon_cz75a": "pistol", "weapon_deagle": "pistol",
    "weapon_revolver": "pistol", "weapon_elite": "pistol",
    "weapon_ak47": "rifle", "weapon_m4a1": "rifle",
    "weapon_m4a1_silencer": "rifle", "weapon_famas": "rifle",
    "weapon_galilar": "rifle", "weapon_sg556": "rifle",
    "weapon_aug": "rifle", "weapon_scar20": "rifle", "weapon_g3sg1": "rifle",
    "weapon_awp": "sniper", "weapon_ssg08": "sniper",
    "weapon_mp9": "smg", "weapon_mp5sd": "smg", "weapon_ump45": "smg",
    "weapon_p90": "smg", "weapon_bizon": "smg", "weapon_mac10": "smg", "weapon_mp7": "smg",
    "weapon_nova": "heavy", "weapon_xm1014": "heavy", "weapon_mag7": "heavy",
    "weapon_sawedoff": "heavy", "weapon_m249": "heavy", "weapon_negev": "heavy",
    "weapon_hegrenade": "grenade", "weapon_flashbang": "grenade",
    "weapon_smokegrenade": "grenade", "weapon_molotov": "grenade",
    "weapon_incgrenade": "grenade", "weapon_decoy": "grenade",
}


def _parse_pos(pos_str: str) -> tuple[float, float, float]:
    try:
        x, y, z = (float(v.strip()) for v in pos_str.split(","))
        return x, y, z
    except Exception:
        return 0.0, 0.0, 0.0


def _active_weapon(weapons: dict) -> str:
    for w in weapons.values():
        if isinstance(w, dict) and w.get("state") == "active":
            return w.get("name", "")
    return ""


def _grenade_inventory(weapons: dict) -> tuple[bool, bool, bool, bool]:
    names = {w.get("name", "") for w in weapons.values() if isinstance(w, dict)}
    return (
        "weapon_smokegrenade" in names,
        "weapon_flashbang" in names,
        "weapon_hegrenade" in names,
        "weapon_molotov" in names or "weapon_incgrenade" in names,
    )


def build_row_from_gsi(
    gsi: dict,
    step: int,
    round_num: int,
    map_name: str,
) -> dict | None:
    """Convert one GSI payload to a row dict for build_state_vector().

    Coordinates are normalized to [0, 1] to match parquet training data.
    Returns None if allplayers data is missing.
    """
    allplayers = gsi.get("allplayers", {})
    if not allplayers:
        return None

    map_info = gsi.get("map", {})
    ct_score  = int(map_info.get("team_ct", {}).get("score", 0))
    t_score   = int(map_info.get("team_t",  {}).get("score", 0))
    ct_streak = int(map_info.get("team_ct", {}).get("consecutive_round_losses", 0))
    t_streak  = int(map_info.get("team_t",  {}).get("consecutive_round_losses", 0))

    t_players: list[dict]  = []
    ct_players: list[dict] = []
    for pdata in allplayers.values():
        team = pdata.get("team", "")
        if team == "T":
            t_players.append(pdata)
        elif team == "CT":
            ct_players.append(pdata)

    # Sort by name — same ordering as demoparser2 training data
    t_players.sort(key=lambda p: p.get("name", ""))
    ct_players.sort(key=lambda p: p.get("name", ""))

    # Compute map_zone from mean T raw position (before normalization)
    t_raw = [_parse_pos(p.get("position", "0,0,0")) for p in t_players
             if int(p.get("state", {}).get("health", 0)) > 0]
    if t_raw:
        mx = sum(c[0] for c in t_raw) / len(t_raw)
        my = sum(c[1] for c in t_raw) / len(t_raw)
        mz = sum(c[2] for c in t_raw) / len(t_raw)
        map_zone = classify_zone(mx, my, map_name, z=mz)
    else:
        map_zone = "other"

    row: dict = {
        "step":             step,
        "tick":             step,
        "round_num":        round_num,
        "bomb_site":        "other",  # unknown at inference time
        "map_zone":         map_zone,
        "ct_score":         ct_score,
        "t_score":          t_score,
        "ct_losing_streak": ct_streak,
        "t_losing_streak":  t_streak,
    }

    for side, players in (("t", t_players), ("ct", ct_players)):
        for i in range(5):
            prefix = f"{side}{i}"
            if i < len(players):
                p      = players[i]
                state  = p.get("state", {})
                weps   = p.get("weapons", {})
                x, y, z = _parse_pos(p.get("position", "0,0,0"))
                xn, yn, zn = normalize_coords(x, y, z, map_name)
                has_smoke, has_flash, has_he, has_molotov = _grenade_inventory(weps)
                flashed = int(state.get("flashed", 0))

                row[f"{prefix}_x"]             = xn
                row[f"{prefix}_y"]             = yn
                row[f"{prefix}_z"]             = zn
                row[f"{prefix}_hp"]            = int(state.get("health", 0))
                row[f"{prefix}_armor"]         = int(state.get("armor", 0))
                row[f"{prefix}_helmet"]        = bool(state.get("helmet", False))
                row[f"{prefix}_alive"]         = int(state.get("health", 0)) > 0
                row[f"{prefix}_role"]          = ""
                row[f"{prefix}_weapon"]        = _WEAPON_CATS.get(_active_weapon(weps), "other")
                row[f"{prefix}_has_smoke"]     = has_smoke
                row[f"{prefix}_has_flash"]     = has_flash
                row[f"{prefix}_has_he"]        = has_he
                row[f"{prefix}_has_molotov"]   = has_molotov
                row[f"{prefix}_flash_duration"] = (flashed / 255.0) * 3.0
                row[f"{prefix}_equip_value"]   = int(state.get("equip_value", 0))
                row[f"{prefix}_is_scoped"]     = False  # not exposed in GSI
                row[f"{prefix}_is_defusing"]   = False  # not exposed in GSI
            else:
                row[f"{prefix}_x"]             = 0.0
                row[f"{prefix}_y"]             = 0.0
                row[f"{prefix}_z"]             = 0.0
                row[f"{prefix}_hp"]            = 0
                row[f"{prefix}_armor"]         = 0
                row[f"{prefix}_helmet"]        = False
                row[f"{prefix}_alive"]         = False
                row[f"{prefix}_role"]          = ""
                row[f"{prefix}_weapon"]        = "other"
                row[f"{prefix}_has_smoke"]     = False
                row[f"{prefix}_has_flash"]     = False
                row[f"{prefix}_has_he"]        = False
                row[f"{prefix}_has_molotov"]   = False
                row[f"{prefix}_flash_duration"] = 0.0
                row[f"{prefix}_equip_value"]   = 0
                row[f"{prefix}_is_scoped"]     = False
                row[f"{prefix}_is_defusing"]   = False

    return row
