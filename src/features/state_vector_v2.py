"""Fixed-order v2 state vector schema aligned with realtime GSI input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils.map_utils import _MAP_BOUNDS, normalize_coords

FEATURE_DIM: int = 218

PLAYER_BASE_FIELDS: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "yaw",
    "in_bomb_zone",
    "hp",
    "armor",
    "helmet",
    "alive",
    "has_smoke",
    "has_flash",
    "has_he",
    "has_molotov",
    "has_c4",
    "balance",
    "equip_value",
    "score",
    "weapon_id",
)

WEAPONS: tuple[str, ...] = (
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
    "ak_47",
    "m4a4",
    "m4a1_s",
    "famas",
    "galil_ar",
    "sg_553",
    "aug",
    "awp",
    "ssg_08",
    "scar_20",
    "g3sg1",
    "mp9",
    "mp5_sd",
    "ump_45",
    "p90",
    "pp_bizon",
    "mac_10",
    "mp7",
    "nova",
    "xm1014",
    "mag_7",
    "sawed_off",
    "m249",
    "negev",
    "no_weapon",
)
NUM_WEAPONS: int = len(WEAPONS)
WEAPON_ID_MAP: dict[str, int] = {w: i for i, w in enumerate(WEAPONS)}

GLOBAL_FIELDS: tuple[str, ...] = ("ct_score", "t_score", "round_num", "time_in_round")
BOMB_FIELDS: tuple[str, ...] = ("bomb_dropped", "bomb_x", "bomb_y")
MAPS: tuple[str, ...] = (
    "de_mirage",
    "de_inferno",
    "de_dust2",
    "de_nuke",
    "de_ancient",
    "de_overpass",
    "de_anubis",
)

SMOKE_SLOTS: int = 5
MOLOTOV_SLOTS: int = 3

PLAYER_STRIDE: int = len(PLAYER_BASE_FIELDS)  # 17 continuous/bool + 1 weapon_id
T_SIDE_DIM: int = 5 * PLAYER_STRIDE
CT_BASE: int = T_SIDE_DIM
GLOBAL_BASE: int = 2 * T_SIDE_DIM
BOMB_BASE: int = GLOBAL_BASE + len(GLOBAL_FIELDS)
SMOKE_BASE: int = BOMB_BASE + len(BOMB_FIELDS)
MOLOTOV_BASE: int = SMOKE_BASE + SMOKE_SLOTS * 3
MAP_BASE: int = MOLOTOV_BASE + MOLOTOV_SLOTS * 3

PLAYER_HP_MAX: float = 100.0
PLAYER_ARMOR_MAX: float = 100.0
PLAYER_BALANCE_MAX: float = 16000.0
PLAYER_EQUIP_MAX: float = 10000.0
PLAYER_SCORE_MAX: float = 50.0
TEAM_SCORE_MAX: float = 30.0
ROUND_NUM_MAX: float = 30.0
TIME_IN_ROUND_MAX: float = 120.0
YAW_MAX: float = 180.0


def _player_feature_names(side: str, player_idx: int) -> tuple[str, ...]:
    prefix = f"{side}{player_idx}"
    return tuple(f"{prefix}_{field}" for field in PLAYER_BASE_FIELDS)


FEATURE_NAMES: tuple[str, ...] = (
    tuple(name for i in range(5) for name in _player_feature_names("t", i))
    + tuple(name for i in range(5) for name in _player_feature_names("ct", i))
    + GLOBAL_FIELDS
    + BOMB_FIELDS
    + tuple(
        f"smoke{slot}_{field}"
        for slot in range(SMOKE_SLOTS)
        for field in ("x", "y", "remain")
    )
    + tuple(
        f"molotov{slot}_{field}"
        for slot in range(MOLOTOV_SLOTS)
        for field in ("x", "y", "remain")
    )
    + tuple(f"map_{map_name}" for map_name in MAPS)
)

FEATURE_IDX: dict[str, int] = {name: idx for idx, name in enumerate(FEATURE_NAMES)}


def _safe_float(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _clip01(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return float(np.clip(value / max_value, 0.0, 1.0))


def _clip_signed(value: float, max_abs: float) -> float:
    if max_abs <= 0:
        return 0.0
    return float(np.clip(value / max_abs, -1.0, 1.0))


def _safe_bool01(value) -> float:
    if pd.isna(value):
        return 0.0
    return float(bool(value))


def _is_sequence(value) -> bool:
    return isinstance(value, Sequence | np.ndarray) and not isinstance(value, (str, bytes))


def _to_mapping(row: pd.Series | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return row


def flatten_feature_dict(feature: Mapping[str, object]) -> dict[str, float]:
    """Flatten a preview-style feature mapping into canonical v2 flat fields."""
    flat: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}

    for side in ("t", "ct"):
        for player_idx in range(5):
            prefix = f"{side}{player_idx}"
            for field in PLAYER_BASE_FIELDS:
                flat[f"{prefix}_{field}"] = _safe_float(feature.get(f"{prefix}_{field}", 0.0))

    for name in GLOBAL_FIELDS + BOMB_FIELDS:
        flat[name] = _safe_float(feature.get(name, 0.0))

    for slot in range(SMOKE_SLOTS):
        for field in ("x", "y", "remain"):
            flat[f"smoke{slot}_{field}"] = _safe_float(feature.get(f"smoke{slot}_{field}", 0.0))

    for slot in range(MOLOTOV_SLOTS):
        for field in ("x", "y", "remain"):
            flat[f"molotov{slot}_{field}"] = _safe_float(feature.get(f"molotov{slot}_{field}", 0.0))

    map_vec = feature.get("map_onehot", [0.0] * len(MAPS))
    for idx, map_name in enumerate(MAPS):
        value = 0.0
        if _is_sequence(map_vec) and idx < len(map_vec):
            value = _safe_float(map_vec[idx], 0.0)
        flat[f"map_{map_name}"] = value

    return flat


def _infer_map_name(data: Mapping[str, object]) -> str | None:
    active_map = None
    active_score = 0.0
    for map_name in MAPS:
        value = _safe_float(data.get(f"map_{map_name}", 0.0))
        if value > active_score:
            active_score = value
            active_map = map_name
    return active_map if active_score > 0.0 else None


def _normalize_xyz(x: float, y: float, z: float, map_name: str | None) -> tuple[float, float, float]:
    if map_name is None or (x == 0.0 and y == 0.0 and z == 0.0):
        return 0.0, 0.0, 0.0
    return normalize_coords(x, y, z, map_name)


def _normalize_xy(x: float, y: float, map_name: str | None) -> tuple[float, float]:
    if map_name is None or (x == 0.0 and y == 0.0):
        return 0.0, 0.0
    xn, yn, _ = normalize_coords(x, y, 0.0, map_name)
    return xn, yn


def normalize_feature_row(row: pd.Series | Mapping[str, object]) -> dict[str, float]:
    """Normalize one canonical raw v2 feature row into model scale."""
    data = _to_mapping(row)
    if "map_onehot" in data:
        data = flatten_feature_dict(data)

    map_name = _infer_map_name(data)
    out = {name: 0.0 for name in FEATURE_NAMES}

    for side in ("t", "ct"):
        for player_idx in range(5):
            prefix = f"{side}{player_idx}"
            raw_x = _safe_float(data.get(f"{prefix}_x", 0.0))
            raw_y = _safe_float(data.get(f"{prefix}_y", 0.0))
            raw_z = _safe_float(data.get(f"{prefix}_z", 0.0))
            norm_x, norm_y, norm_z = _normalize_xyz(raw_x, raw_y, raw_z, map_name)
            out[f"{prefix}_x"] = norm_x
            out[f"{prefix}_y"] = norm_y
            out[f"{prefix}_z"] = norm_z
            out[f"{prefix}_yaw"] = _clip_signed(_safe_float(data.get(f"{prefix}_yaw", 0.0)), YAW_MAX)

            for field in (
                "in_bomb_zone",
                "helmet",
                "alive",
                "has_smoke",
                "has_flash",
                "has_he",
                "has_molotov",
                "has_c4",
            ):
                out[f"{prefix}_{field}"] = _safe_bool01(data.get(f"{prefix}_{field}", False))

            out[f"{prefix}_hp"] = _clip01(_safe_float(data.get(f"{prefix}_hp", 0.0)), PLAYER_HP_MAX)
            out[f"{prefix}_armor"] = _clip01(_safe_float(data.get(f"{prefix}_armor", 0.0)), PLAYER_ARMOR_MAX)
            out[f"{prefix}_balance"] = _clip01(_safe_float(data.get(f"{prefix}_balance", 0.0)), PLAYER_BALANCE_MAX)
            out[f"{prefix}_equip_value"] = _clip01(_safe_float(data.get(f"{prefix}_equip_value", 0.0)), PLAYER_EQUIP_MAX)
            out[f"{prefix}_score"] = _clip01(_safe_float(data.get(f"{prefix}_score", 0.0)), PLAYER_SCORE_MAX)
            out[f"{prefix}_weapon_id"] = _safe_float(
                data.get(f"{prefix}_weapon_id", float(NUM_WEAPONS - 1))
            )

    out["ct_score"] = _clip01(_safe_float(data.get("ct_score", 0.0)), TEAM_SCORE_MAX)
    out["t_score"] = _clip01(_safe_float(data.get("t_score", 0.0)), TEAM_SCORE_MAX)
    out["round_num"] = _clip01(_safe_float(data.get("round_num", 0.0)), ROUND_NUM_MAX)
    out["time_in_round"] = _clip01(
        _safe_float(data.get("time_in_round", 0.0)),
        TIME_IN_ROUND_MAX,
    )

    bomb_dropped = _safe_bool01(data.get("bomb_dropped", False))
    out["bomb_dropped"] = bomb_dropped
    raw_bomb_x = _safe_float(data.get("bomb_x", 0.0))
    raw_bomb_y = _safe_float(data.get("bomb_y", 0.0))
    if bomb_dropped > 0.0 and (raw_bomb_x != 0.0 or raw_bomb_y != 0.0):
        out["bomb_x"], out["bomb_y"] = _normalize_xy(raw_bomb_x, raw_bomb_y, map_name)

    for slot in range(SMOKE_SLOTS):
        remain = _clip01(_safe_float(data.get(f"smoke{slot}_remain", 0.0)), 1.0)
        out[f"smoke{slot}_remain"] = remain
        raw_x = _safe_float(data.get(f"smoke{slot}_x", 0.0))
        raw_y = _safe_float(data.get(f"smoke{slot}_y", 0.0))
        if remain > 0.0 and (raw_x != 0.0 or raw_y != 0.0):
            out[f"smoke{slot}_x"], out[f"smoke{slot}_y"] = _normalize_xy(raw_x, raw_y, map_name)

    for slot in range(MOLOTOV_SLOTS):
        remain = _clip01(_safe_float(data.get(f"molotov{slot}_remain", 0.0)), 1.0)
        out[f"molotov{slot}_remain"] = remain
        raw_x = _safe_float(data.get(f"molotov{slot}_x", 0.0))
        raw_y = _safe_float(data.get(f"molotov{slot}_y", 0.0))
        if remain > 0.0 and (raw_x != 0.0 or raw_y != 0.0):
            out[f"molotov{slot}_x"], out[f"molotov{slot}_y"] = _normalize_xy(raw_x, raw_y, map_name)

    for map_name_candidate in MAPS:
        out[f"map_{map_name_candidate}"] = _safe_bool01(
            _safe_float(data.get(f"map_{map_name_candidate}", 0.0)) > 0.5
        )

    return out


def build_state_vector(row: pd.Series | Mapping[str, object]) -> np.ndarray:
    """Build one v2 218-dim float32 feature vector."""
    data = normalize_feature_row(row)
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    for idx, name in enumerate(FEATURE_NAMES):
        vec[idx] = _safe_float(data.get(name, 0.0))
    return vec


def build_state_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert canonical raw v2 rows to a normalized float32 matrix."""
    if df.empty:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    n_rows = len(df)
    mat = np.zeros((n_rows, FEATURE_DIM), dtype=np.float32)

    map_name: str | None = None
    for m in MAPS:
        col = f"map_{m}"
        if col in df.columns:
            vals = df[col].values
            if np.nanmax(vals) > 0.5:
                map_name = m
                break

    bounds = _MAP_BOUNDS.get(map_name, None) if map_name else None

    def _col(name: str) -> np.ndarray:
        if name in df.columns:
            arr = df[name].values.astype(np.float32)
            np.nan_to_num(arr, copy=False)
            return arr
        return np.zeros(n_rows, dtype=np.float32)

    def _norm_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if bounds is None:
            return (
                np.full(n_rows, 0.5, dtype=np.float32),
                np.full(n_rows, 0.5, dtype=np.float32),
                np.full(n_rows, 0.5, dtype=np.float32),
            )
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        nx = np.clip((x - x_min) / (x_max - x_min or 1.0), 0.0, 1.0)
        ny = np.clip((y - y_min) / (y_max - y_min or 1.0), 0.0, 1.0)
        nz = np.clip((z - z_min) / (z_max - z_min or 1.0), 0.0, 1.0)
        mask = (x == 0.0) & (y == 0.0) & (z == 0.0)
        nx[mask] = 0.0
        ny[mask] = 0.0
        nz[mask] = 0.0
        return nx.astype(np.float32), ny.astype(np.float32), nz.astype(np.float32)

    def _norm_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if bounds is None:
            return (
                np.full(n_rows, 0.5, dtype=np.float32),
                np.full(n_rows, 0.5, dtype=np.float32),
            )
        x_min, x_max, y_min, y_max = bounds[0], bounds[1], bounds[2], bounds[3]
        nx = np.clip((x - x_min) / (x_max - x_min or 1.0), 0.0, 1.0)
        ny = np.clip((y - y_min) / (y_max - y_min or 1.0), 0.0, 1.0)
        mask = (x == 0.0) & (y == 0.0)
        nx[mask] = 0.0
        ny[mask] = 0.0
        return nx.astype(np.float32), ny.astype(np.float32)

    for side in ("t", "ct"):
        for pi in range(5):
            prefix = f"{side}{pi}"
            base = FEATURE_IDX[f"{prefix}_x"]

            nx, ny, nz = _norm_xyz(
                _col(f"{prefix}_x"),
                _col(f"{prefix}_y"),
                _col(f"{prefix}_z"),
            )
            mat[:, base + 0] = nx
            mat[:, base + 1] = ny
            mat[:, base + 2] = nz
            mat[:, base + 3] = np.clip(_col(f"{prefix}_yaw") / YAW_MAX, -1.0, 1.0)
            mat[:, base + 4] = (_col(f"{prefix}_in_bomb_zone") > 0.5).astype(np.float32)
            mat[:, base + 5] = np.clip(_col(f"{prefix}_hp") / PLAYER_HP_MAX, 0.0, 1.0)
            mat[:, base + 6] = np.clip(_col(f"{prefix}_armor") / PLAYER_ARMOR_MAX, 0.0, 1.0)
            mat[:, base + 7] = (_col(f"{prefix}_helmet") > 0.5).astype(np.float32)
            mat[:, base + 8] = (_col(f"{prefix}_alive") > 0.5).astype(np.float32)

            for j, field in enumerate(
                ("has_smoke", "has_flash", "has_he", "has_molotov", "has_c4")
            ):
                mat[:, base + 9 + j] = (_col(f"{prefix}_{field}") > 0.5).astype(np.float32)

            mat[:, base + 14] = np.clip(_col(f"{prefix}_balance") / PLAYER_BALANCE_MAX, 0.0, 1.0)
            mat[:, base + 15] = np.clip(_col(f"{prefix}_equip_value") / PLAYER_EQUIP_MAX, 0.0, 1.0)
            mat[:, base + 16] = np.clip(_col(f"{prefix}_score") / PLAYER_SCORE_MAX, 0.0, 1.0)
            mat[:, base + 17] = _col(f"{prefix}_weapon_id")

    g = FEATURE_IDX["ct_score"]
    mat[:, g + 0] = np.clip(_col("ct_score") / TEAM_SCORE_MAX, 0.0, 1.0)
    mat[:, g + 1] = np.clip(_col("t_score") / TEAM_SCORE_MAX, 0.0, 1.0)
    mat[:, g + 2] = np.clip(_col("round_num") / ROUND_NUM_MAX, 0.0, 1.0)
    mat[:, g + 3] = np.clip(_col("time_in_round") / TIME_IN_ROUND_MAX, 0.0, 1.0)

    b = FEATURE_IDX["bomb_dropped"]
    bomb_dropped = (_col("bomb_dropped") > 0.5).astype(np.float32)
    mat[:, b] = bomb_dropped
    bx, by = _norm_xy(_col("bomb_x"), _col("bomb_y"))
    mat[:, b + 1] = bx * bomb_dropped
    mat[:, b + 2] = by * bomb_dropped

    for slot in range(SMOKE_SLOTS):
        s = FEATURE_IDX[f"smoke{slot}_x"]
        remain = np.clip(_col(f"smoke{slot}_remain"), 0.0, 1.0)
        sx, sy = _norm_xy(_col(f"smoke{slot}_x"), _col(f"smoke{slot}_y"))
        active = (remain > 0.0).astype(np.float32)
        mat[:, s + 0] = sx * active
        mat[:, s + 1] = sy * active
        mat[:, s + 2] = remain

    for slot in range(MOLOTOV_SLOTS):
        m_idx = FEATURE_IDX[f"molotov{slot}_x"]
        remain = np.clip(_col(f"molotov{slot}_remain"), 0.0, 1.0)
        mx, my = _norm_xy(_col(f"molotov{slot}_x"), _col(f"molotov{slot}_y"))
        active = (remain > 0.0).astype(np.float32)
        mat[:, m_idx + 0] = mx * active
        mat[:, m_idx + 1] = my * active
        mat[:, m_idx + 2] = remain

    for m in MAPS:
        mat[:, FEATURE_IDX[f"map_{m}"]] = (_col(f"map_{m}") > 0.5).astype(np.float32)

    return mat
