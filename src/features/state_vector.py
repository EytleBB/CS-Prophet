"""Build fixed-size float32 feature vectors from parquet rows."""

from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_DIM: int = 279
# Layout
# ──────────────────────────────────────────────────────────────────
# [0:135]    T players 0–4  ×  27 dims each
# [135:270]  CT players 0–4 ×  27 dims each
# [270:274]  map_zone one-hot  (A=270, B=271, mid=272, other=273)
# [274:279]  global: ct_score/30, t_score/30, round_num/30,
#                    ct_losing_streak/5, t_losing_streak/5
#
# Per-player stride = 27:
#   [0:7]   x, y, z, hp/100, armor/100, helmet, alive
#   [7:12]  role one-hot (IGL, AWPer, Entry fragger, Support, Lurker)
#   [12:19] weapon_category one-hot (pistol, rifle, sniper, smg, heavy, grenade, other)
#   [19]    has_smoke
#   [20]    has_flash
#   [21]    has_he
#   [22]    has_molotov
#   [23]    flash_duration / 3.0
#   [24]    equip_value / 20000.0
#   [25]    is_scoped
#   [26]    is_defusing
# ──────────────────────────────────────────────────────────────────

_PLAYER_STRIDE: int = 27
_CT_BASE: int = 135
_ZONE_BASE: int = 270
_GLOBAL_BASE: int = 274

ZONE_IDX: dict[str, int] = {"A": 0, "B": 1, "mid": 2, "other": 3}

PLAYER_FIELDS: tuple[str, ...] = ("x", "y", "z", "hp", "armor", "helmet", "alive")
NORMALISE: frozenset[str] = frozenset({"hp", "armor"})

ROLE_IDX: dict[str, int] = {
    "IGL": 0,
    "AWPer": 1,
    "Entry fragger": 2,
    "Support": 3,
    "Lurker": 4,
}

WEAPON_CAT_IDX: dict[str, int] = {
    "pistol":  0,
    "rifle":   1,
    "sniper":  2,
    "smg":     3,
    "heavy":   4,
    "grenade": 5,
    "other":   6,
}

_FLASH_MAX:  float = 3.0
_EQUIP_MAX:  float = 20000.0
_SCORE_MAX:  float = 30.0
_STREAK_MAX: float = 5.0


def build_state_vector(row: pd.Series) -> np.ndarray:
    """Convert one parquet row to a float32 feature vector of shape (FEATURE_DIM,).

    Args:
        row: One row from a parsed demo parquet (pd.Series with column-name index).

    Returns:
        np.ndarray of shape (279,) and dtype float32.
    """
    # Convert to dict once — avoids pandas 3.x Series.__getitem__ cache bug
    d = row.to_dict()
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    for side, base in (("t", 0), ("ct", _CT_BASE)):
        for i in range(5):
            pb = base + i * _PLAYER_STRIDE

            # Base fields
            for j, field in enumerate(PLAYER_FIELDS):
                col = f"{side}{i}_{field}"
                val = float(d.get(col, 0.0))
                if field in NORMALISE:
                    val /= 100.0
                vec[pb + j] = val

            # Role one-hot [7:12]
            role = str(d.get(f"{side}{i}_role", ""))
            if role in ROLE_IDX:
                vec[pb + 7 + ROLE_IDX[role]] = 1.0

            # Weapon category one-hot [12:19]
            weapon_cat = str(d.get(f"{side}{i}_weapon", "other"))
            w_idx = WEAPON_CAT_IDX.get(weapon_cat, 6)
            vec[pb + 12 + w_idx] = 1.0

            # Grenade inventory flags [19:23]
            for k, flag in enumerate(("has_smoke", "has_flash", "has_he", "has_molotov")):
                vec[pb + 19 + k] = float(bool(d.get(f"{side}{i}_{flag}", False)))

            # Flash duration [23]
            vec[pb + 23] = min(1.0, float(d.get(f"{side}{i}_flash_duration", 0.0)) / _FLASH_MAX)

            # Equipment value [24]
            vec[pb + 24] = min(1.0, float(d.get(f"{side}{i}_equip_value", 0.0)) / _EQUIP_MAX)

            # is_scoped [25]
            vec[pb + 25] = float(bool(d.get(f"{side}{i}_is_scoped", False)))

            # is_defusing [26]
            vec[pb + 26] = float(bool(d.get(f"{side}{i}_is_defusing", False)))

    # Map zone one-hot [270:274]
    zone_str = str(d.get("map_zone", "other"))
    zone_idx = ZONE_IDX.get(zone_str, 3)
    vec[_ZONE_BASE + zone_idx] = 1.0

    # Global features [274:279]
    vec[_GLOBAL_BASE + 0] = min(1.0, float(d.get("ct_score", 0)) / _SCORE_MAX)
    vec[_GLOBAL_BASE + 1] = min(1.0, float(d.get("t_score", 0)) / _SCORE_MAX)
    vec[_GLOBAL_BASE + 2] = min(1.0, float(d.get("round_num", 0)) / _SCORE_MAX)
    vec[_GLOBAL_BASE + 3] = min(1.0, float(d.get("ct_losing_streak", 0)) / _STREAK_MAX)
    vec[_GLOBAL_BASE + 4] = min(1.0, float(d.get("t_losing_streak", 0)) / _STREAK_MAX)

    return vec
