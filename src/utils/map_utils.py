"""CS2 map zone utilities — zone classification and coordinate normalisation."""

from __future__ import annotations
from typing import Final

# Approximate bounding boxes per map zone: (x_min, x_max, y_min, y_max).
# Coordinates are CS2 world units as reported by demoparser2.
_ZONE_BOXES: Final[dict[str, dict[str, tuple[float, float, float, float]]]] = {
    "de_mirage": {
        "A":   ( 630.0,  1530.0, -880.0,   70.0),
        "B":   (-1000.0,  -350.0, -360.0,  490.0),
        "mid": ( -350.0,   630.0, -880.0,  490.0),
    },
    "de_inferno": {
        "A":   (1490.0, 2870.0,  450.0, 1100.0),
        "B":   ( 180.0,  810.0, -1250.0, -550.0),
        "mid": ( 810.0, 1490.0,  -550.0,  450.0),
    },
    "de_dust2": {
        "A":   ( 660.0, 2020.0, 1650.0, 2820.0),
        "B":   (-2220.0, -1100.0,  180.0, 1080.0),
        "mid": (-1100.0,  660.0,  180.0, 1650.0),
    },
    "de_nuke": {
        "A":   (-900.0, -280.0, 1130.0, 1830.0),
        "B":   (-900.0, -280.0,  230.0, 1130.0),
        "mid": (-280.0,  900.0,  230.0, 1830.0),
    },
    "de_ancient": {
        "A":   ( 930.0, 2070.0, -520.0,  370.0),
        "B":   (-1150.0, -180.0, -690.0,  170.0),
        "mid": ( -180.0,  930.0, -690.0,  370.0),
    },
    "de_overpass": {
        "A":   (  -1.0, 1200.0,  700.0, 1800.0),
        "B":   (-1900.0, -700.0, -900.0,  200.0),
        "mid": ( -700.0,   -1.0, -900.0, 1800.0),
    },
    "de_anubis": {
        "A":   (1400.0, 2600.0, -200.0,  700.0),
        "B":   ( -500.0,  600.0, -1000.0, -100.0),
        "mid": (  600.0, 1400.0, -1000.0,  700.0),
    },
}

_Z_MIN: Final[float] = -500.0
_Z_MAX: Final[float] =  500.0


def classify_zone(x: float, y: float, map_name: str) -> str:
    """Return zone label ('A', 'B', 'mid', or 'other') for a world position.

    Args:
        x: World x-coordinate (demoparser2 units).
        y: World y-coordinate.
        map_name: Map name string, e.g. 'de_mirage'.

    Returns:
        Zone name string.
    """
    for zone, (x_min, x_max, y_min, y_max) in _ZONE_BOXES.get(map_name, {}).items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone
    return "other"


def normalize_coords(x: float, y: float, z: float, map_name: str) -> tuple[float, float, float]:
    """Normalise (x, y, z) to [0, 1] using per-map bounding extents.

    x/y extents are derived from the union of all zone bounding boxes.
    z is clamped to [-500, 500] (covers all walkable CS2 geometry).

    Returns (0.5, 0.5, 0.5) for unknown maps.
    """
    zones = _ZONE_BOXES.get(map_name)
    if not zones:
        return 0.5, 0.5, 0.5

    all_x = [v for box in zones.values() for v in (box[0], box[1])]
    all_y = [v for box in zones.values() for v in (box[2], box[3])]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0
    z_range = _Z_MAX - _Z_MIN

    return (
        max(0.0, min(1.0, (x - x_min) / x_range)),
        max(0.0, min(1.0, (y - y_min) / y_range)),
        max(0.0, min(1.0, (z - _Z_MIN) / z_range)),
    )
