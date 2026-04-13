"""Build fixed-size float32 feature vectors from parquet rows."""

from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_DIM: int = 275
# Layout
# ──────────────────────────────────────────────────────────────────
# [0:135]    T players 0–4  ×  27 dims each
# [135:270]  CT players 0–4 ×  27 dims each
# [270:275]  global: ct_score/30, t_score/30, round_num/30,
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
_GLOBAL_BASE: int = 270

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
        np.ndarray of shape (275,) and dtype float32.
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

    # Global features [270:275]
    vec[_GLOBAL_BASE + 0] = min(1.0, float(d.get("ct_score", 0)) / _SCORE_MAX)
    vec[_GLOBAL_BASE + 1] = min(1.0, float(d.get("t_score", 0)) / _SCORE_MAX)
    vec[_GLOBAL_BASE + 2] = min(1.0, float(d.get("round_num", 0)) / _SCORE_MAX)
    vec[_GLOBAL_BASE + 3] = min(1.0, float(d.get("ct_losing_streak", 0)) / _STREAK_MAX)
    vec[_GLOBAL_BASE + 4] = min(1.0, float(d.get("t_losing_streak", 0)) / _STREAK_MAX)

    return vec


def build_state_matrix(df: pd.DataFrame) -> np.ndarray:
    """Vectorised: convert a sorted round DataFrame to (n_rows, FEATURE_DIM) float32."""
    n = len(df)
    mat = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    def _col(name: str, default: float = 0.0) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).values.astype(np.float32)
        return np.full(n, default, dtype=np.float32)

    for side, base in (("t", 0), ("ct", _CT_BASE)):
        for i in range(5):
            pb = base + i * _PLAYER_STRIDE
            for j, field in enumerate(PLAYER_FIELDS):
                vals = _col(f"{side}{i}_{field}")
                if field in NORMALISE:
                    vals = vals / 100.0
                mat[:, pb + j] = vals

            role_col = f"{side}{i}_role"
            if role_col in df.columns:
                for role, ridx in ROLE_IDX.items():
                    mat[:, pb + 7 + ridx] = (df[role_col].values == role).astype(np.float32)

            wep_col = f"{side}{i}_weapon"
            if wep_col in df.columns:
                wvals = df[wep_col].values
                for cat, cidx in WEAPON_CAT_IDX.items():
                    mat[:, pb + 12 + cidx] = (wvals == cat).astype(np.float32)
                known = np.isin(wvals, list(WEAPON_CAT_IDX.keys()))
                mat[:, pb + 12 + 6] = np.where(known, mat[:, pb + 12 + 6], 1.0)

            for k, flag in enumerate(("has_smoke", "has_flash", "has_he", "has_molotov")):
                mat[:, pb + 19 + k] = _col(f"{side}{i}_{flag}")

            mat[:, pb + 23] = np.clip(_col(f"{side}{i}_flash_duration") / _FLASH_MAX, 0, 1)
            mat[:, pb + 24] = np.clip(_col(f"{side}{i}_equip_value") / _EQUIP_MAX, 0, 1)
            mat[:, pb + 25] = _col(f"{side}{i}_is_scoped")
            mat[:, pb + 26] = _col(f"{side}{i}_is_defusing")

    mat[:, _GLOBAL_BASE + 0] = np.clip(_col("ct_score") / _SCORE_MAX, 0, 1)
    mat[:, _GLOBAL_BASE + 1] = np.clip(_col("t_score") / _SCORE_MAX, 0, 1)
    mat[:, _GLOBAL_BASE + 2] = np.clip(_col("round_num") / _SCORE_MAX, 0, 1)
    mat[:, _GLOBAL_BASE + 3] = np.clip(_col("ct_losing_streak") / _STREAK_MAX, 0, 1)
    mat[:, _GLOBAL_BASE + 4] = np.clip(_col("t_losing_streak") / _STREAK_MAX, 0, 1)

    return mat
