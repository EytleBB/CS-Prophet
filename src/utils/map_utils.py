"""CS2 map zone utilities — zone classification and coordinate normalisation."""

from __future__ import annotations
from typing import Final, Optional

# Zone bounding boxes per map: (x_min, x_max, y_min, y_max).
# Calibrated from demoparser2 coordinates across HLTV professional demos.
# For de_nuke, zones are defined by Z (z_min, z_max) instead of X/Y because
# the A (upper) and B (lower) sites share the same X/Y footprint.
_ZONE_BOXES: Final[dict[str, dict[str, tuple[float, float, float, float]]]] = {
    "de_mirage": {
        # A site: van / apps / tunnel area, lower-center
        "A":   ( -900.0,   200.0, -2603.0, -1400.0),
        # B site: palace / ramp / jungle area, upper-left
        "B":   (-2603.0, -1200.0, -200.0,  882.0),
        # mid: connector / catwalk / short
        "mid": (-1200.0,   400.0, -1400.0,  -200.0),
    },
    "de_inferno": {
        # A site: scaffold / CT area
        "A":   (1400.0, 2640.0,   -717.0,  900.0),
        # B site: banana / construction
        "B":   ( -200.0,  900.0,  2400.0, 3514.0),
        # mid: apartments / arch / library
        "mid": ( 600.0, 1600.0,   600.0, 2400.0),
    },
    "de_dust2": {
        # A site: long A / catwalk / ramp
        "A":   ( 600.0, 1788.0, 2100.0, 3044.0),
        # B site: B tunnels / platform
        "B":   (-2186.0, -900.0, 2100.0, 3044.0),
        # mid: mid / catwalk / B doors
        "mid": ( -900.0,  600.0,  -981.0, 2100.0),
    },
    "de_ancient": {
        # A site: cave / donut (negative X, northern area)
        "A":   (-2263.0, -900.0,  200.0, 1728.0),
        # B site: temple / ruins (positive X, center)
        "B":   ( 500.0, 1396.0, -400.0,  600.0),
        # mid: middle path / connector
        "mid": ( -900.0,  500.0, -400.0, 1000.0),
    },
    "de_overpass": {
        # A site: upper bank / short
        "A":   (-2800.0, -1800.0,  200.0, 1610.0),
        # B site: underpass / lower B
        "B":   (-1600.0,  -600.0, -900.0,  200.0),
        # mid: connector / bathrooms / monster
        "mid": (-1800.0,  -600.0,  200.0,  900.0),
    },
    "de_anubis": {
        # A site: upper right / palace area
        "A":   ( 800.0, 1804.0, 1400.0, 2945.0),
        # B site: lower left / canal
        "B":   (-1954.0, -600.0, -400.0,  900.0),
        # mid: bridge / mid connector
        "mid": ( -600.0,  800.0,  300.0, 1400.0),
    },
}

# de_nuke uses Z-based classification (A=upper, B=lower)
_NUKE_Z_THRESHOLD: Final[float] = -550.0   # above → A, below → B

# Full map bounding boxes used for coordinate normalisation (x_min, x_max, y_min, y_max, z_min, z_max)
_MAP_BOUNDS: Final[dict[str, tuple[float, float, float, float, float, float]]] = {
    "de_mirage":   (-2603.0, 1400.0, -2603.0,  882.0, -400.0,  50.0),
    "de_inferno":  (-1700.0, 2640.0,  -717.0, 3514.0,  -80.0, 350.0),
    "de_dust2":    (-2186.0, 1788.0,  -981.0, 3044.0, -200.0, 200.0),
    "de_nuke":     (-2570.0, 3498.0, -2477.0,  935.0, -800.0,   0.0),
    "de_ancient":  (-2263.0, 1396.0, -2288.0, 1728.0, -200.0, 260.0),
    "de_overpass": (-3960.0,  -106.0, -3450.0, 1610.0,   -20.0, 640.0),
    "de_anubis":   (-1954.0, 1804.0, -1735.0, 2945.0, -210.0, 180.0),
}


def infer_map_from_positions(positions: list[tuple[float, float, float]]) -> str:
    """Return the unique map whose bbox contains every non-zero (x,y,z), else ''.

    Used when CS2 process memory doesn't expose the map-name field directly —
    we reverse-lookup via the bounding boxes already defined for normalisation.
    """
    real = [(x, y, z) for x, y, z in positions if not (x == 0.0 and y == 0.0 and z == 0.0)]
    if not real:
        return ""
    hits: list[tuple[str, float]] = []
    for name, (xmin, xmax, ymin, ymax, zmin, zmax) in _MAP_BOUNDS.items():
        if all(xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax for x, y, z in real):
            volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
            hits.append((name, volume))
    if not hits:
        return ""
    hits.sort(key=lambda item: item[1])
    return hits[0][0]


def map_fit_fraction(positions: list[tuple[float, float, float]], map_name: str) -> float:
    """Return the fraction of non-zero positions that fall within a map bbox."""
    bounds = _MAP_BOUNDS.get(map_name)
    if bounds is None:
        return 0.0

    real = [(x, y, z) for x, y, z in positions if not (x == 0.0 and y == 0.0 and z == 0.0)]
    if not real:
        return 0.0

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    fits = sum(
        1
        for x, y, z in real
        if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max
    )
    return fits / len(real)


def positions_fit_map(positions: list[tuple[float, float, float]], map_name: str) -> bool:
    """Return True when every non-zero position lies within the map bounds."""
    return map_fit_fraction(positions, map_name) >= 1.0


def classify_zone(x: float, y: float, map_name: str, z: Optional[float] = None) -> str:
    """Return zone label ('A', 'B', 'mid', or 'other') for a world position.

    For de_nuke, z is used to distinguish the upper (A) and lower (B) sites.

    Args:
        x: World x-coordinate (demoparser2 units).
        y: World y-coordinate.
        map_name: Map name string, e.g. 'de_mirage'.
        z: World z-coordinate (optional; required for de_nuke to give A/B).

    Returns:
        Zone name string.
    """
    if map_name == "de_nuke":
        if z is not None:
            return "A" if z > _NUKE_Z_THRESHOLD else "B"
        return "other"

    for zone, (x_min, x_max, y_min, y_max) in _ZONE_BOXES.get(map_name, {}).items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone
    return "other"


def normalize_coords(x: float, y: float, z: float, map_name: str) -> tuple[float, float, float]:
    """Normalise (x, y, z) to [0, 1] using per-map bounding extents.

    Returns (0.5, 0.5, 0.5) for unknown maps.
    """
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
