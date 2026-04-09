"""CS2 map zone utilities — zone classification and coordinate normalisation."""
from __future__ import annotations
from typing import Final, Optional

_ZONE_BOXES: Final[dict[str, dict[str, tuple[float, float, float, float]]]] = {
    "de_mirage": {
        "A":   (-2603.0, -1200.0, -200.0,  882.0),
        "B":   ( -900.0,   200.0, -2603.0, -1400.0),
        "mid": (-1200.0,   400.0, -1400.0,  -200.0),
    },
    "de_inferno": {
        "A":   (1400.0, 2640.0,   -717.0,  900.0),
        "B":   ( -200.0,  900.0,  2400.0, 3514.0),
        "mid": ( 600.0, 1600.0,   600.0, 2400.0),
    },
    "de_dust2": {
        "A":   ( 600.0, 1788.0, 2100.0, 3044.0),
        "B":   (-2186.0, -900.0, 2100.0, 3044.0),
        "mid": ( -900.0,  600.0,  -981.0, 2100.0),
    },
    "de_ancient": {
        "A":   ( 500.0, 1396.0, -400.0,  600.0),
        "B":   (-2263.0, -900.0,  200.0, 1728.0),
        "mid": ( -900.0,  500.0, -400.0, 1000.0),
    },
    "de_overpass": {
        "A":   (-2800.0, -1800.0,  200.0, 1610.0),
        "B":   (-1600.0,  -600.0, -900.0,  200.0),
        "mid": (-1800.0,  -600.0,  200.0,  900.0),
    },
    "de_anubis": {
        "A":   ( 800.0, 1804.0, 1400.0, 2945.0),
        "B":   (-1954.0, -600.0, -400.0,  900.0),
        "mid": ( -600.0,  800.0,  300.0, 1400.0),
    },
}

_NUKE_Z_THRESHOLD: Final[float] = -550.0

_MAP_BOUNDS: Final[dict[str, tuple[float, float, float, float, float, float]]] = {
    "de_mirage":   (-2603.0, 1400.0, -2603.0,  882.0, -400.0,  50.0),
    "de_inferno":  (-1700.0, 2640.0,  -717.0, 3514.0,  -80.0, 350.0),
    "de_dust2":    (-2186.0, 1788.0,  -981.0, 3044.0, -200.0, 200.0),
    "de_nuke":     (-2570.0, 3498.0, -2477.0,  935.0, -800.0,   0.0),
    "de_ancient":  (-2263.0, 1396.0, -2288.0, 1728.0, -200.0, 260.0),
    "de_overpass": (-3960.0,  -106.0, -3450.0, 1610.0,   -20.0, 640.0),
    "de_anubis":   (-1954.0, 1804.0, -1735.0, 2945.0, -210.0, 180.0),
}


def classify_zone(x: float, y: float, map_name: str, z: Optional[float] = None) -> str:
    if map_name == "de_nuke":
        if z is not None:
            return "A" if z > _NUKE_Z_THRESHOLD else "B"
        return "other"

    for zone, (x_min, x_max, y_min, y_max) in _ZONE_BOXES.get(map_name, {}).items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone
    return "other"


def normalize_coords(x: float, y: float, z: float, map_name: str) -> tuple[float, float, float]:
    bounds = _MAP_BOUNDS.get(map_name)
    if not bounds:
        return 0.5, 0.5, 0.5

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0
    z_range = z_max - z_min or 1.0

    return (
        max(0.0, min(1.0, (x - x_min) / x_range)),
        max(0.0, min(1.0, (y - y_min) / y_range)),
        max(0.0, min(1.0, (z - z_min) / z_range)),
    )
