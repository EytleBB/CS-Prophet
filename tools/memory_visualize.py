"""Live CS2 memory visualizer for manual parity checks against training rows.

Usage:
    python tools/memory_visualize.py
    python tools/memory_visualize.py --watch 0.5
    python tools/memory_visualize.py --map de_dust2

The left panel renders the current round on top of the radar PNG when one is
available. The right panel shows the exact `t0..t4 / ct0..ct4` slot mapping
used by the canonical raw v2 row builders, so memory reads can be checked
against the same ordering used during training.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle
except ImportError:  # pragma: no cover - environment dependent
    plt = None
    Axes = Any  # type: ignore[assignment]
    Figure = Any  # type: ignore[assignment]
    Circle = Any  # type: ignore[assignment]
    Rectangle = Any  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.state_vector_v2 import FEATURE_IDX, MAPS, MOLOTOV_SLOTS, WEAPONS, build_state_vector  # noqa: E402
from src.inference.memory_reader import CS2MemoryReader  # noqa: E402
from src.inference.memory_state_builder import (  # noqa: E402
    _active_weapon_id,
    _equip_value,
    build_row_from_memory,
)
from src.utils.map_utils import (  # noqa: E402
    _MAP_BOUNDS,
    _ZONE_BOXES,
    classify_zone,
    infer_map_from_positions,
    map_fit_fraction,
)
from src.utils.paths import data_path  # noqa: E402

T_COLOR = "#e74c3c"
CT_COLOR = "#3498db"
BOMB_COLOR = "#f1c40f"
SMOKE_COLOR = "#95a5a6"
MOLOTOV_COLOR = "#e67e22"
DEAD_COLOR = "#7f8c8d"
PANEL_BG = "#121826"
MAP_BG = "#0d1117"
GRID_COLOR = "#2d3748"
TEXT_COLOR = "#f8fafc"
_MAP_SWITCH_CONFIRM_FRAMES = 3
_CACHED_MAP_KEEP_FRACTION = 0.6
_NEW_MAP_SWITCH_FRACTION = 0.8
_MIN_POSITIONS_FOR_COORD_SWITCH = 4

# CS2 radar calibration used by demo_visualize.py as well.
# pos_x = game X at left edge, pos_y = game Y at top edge, scale = game units/pixel
RADAR_META: dict[str, dict[str, float]] = {
    "de_mirage": {"pos_x": -3230.0, "pos_y": 1713.0, "scale": 5.0},
    "de_inferno": {"pos_x": -2087.0, "pos_y": 3870.0, "scale": 4.9},
    "de_dust2": {"pos_x": -2476.0, "pos_y": 3239.0, "scale": 4.4},
    "de_nuke": {"pos_x": -3453.0, "pos_y": 2887.0, "scale": 7.0},
    "de_ancient": {"pos_x": -2953.0, "pos_y": 2164.0, "scale": 5.0},
    "de_overpass": {"pos_x": -4831.0, "pos_y": 1781.0, "scale": 5.2},
    "de_anubis": {"pos_x": -2796.0, "pos_y": 3328.0, "scale": 5.22},
}


@dataclass(slots=True)
class Snapshot:
    captured_at: str
    map_name: str
    map_source: str
    map_status: str
    pending_map: str
    pending_count: int
    players: list[dict[str, Any]]
    map_state: dict[str, Any]
    bomb: dict[str, Any]
    projectiles: dict[str, Any]
    row: dict[str, float] | None
    molotov_reader: list[tuple[float, float, float]]
    molotov_row: list[tuple[float, float, float]]
    molotov_vec: list[tuple[float, float, float]]
    row_status: str
    radar_path: Path | None


@dataclass(slots=True)
class MapResolutionState:
    cached_map: str = ""
    pending_map: str = ""
    pending_count: int = 0


@dataclass(slots=True)
class Projection:
    mode: str
    project_xy: Callable[[float, float], tuple[float, float]]
    yaw_len: float
    label_dx: float
    label_dy: float
    smoke_radius: float
    molotov_radius: float


def _radar_search_dirs() -> list[Path]:
    seen: set[str] = set()
    candidates = [
        data_path("viz", "assets", prefer_existing=True),
        REPO_ROOT.parent / "maps",
        REPO_ROOT / "data" / "viz" / "assets",
    ]
    dirs: list[Path] = []
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        dirs.append(path)
    return dirs


def _available_radar_paths() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for base_dir in _radar_search_dirs():
        if not base_dir.exists():
            continue
        for image_path in sorted(base_dir.glob("*.png")):
            found.setdefault(image_path.stem, image_path)
    return found


def _radar_path(map_name: str) -> Path | None:
    if not map_name:
        return None
    return _available_radar_paths().get(map_name)


def _short_weapon_name(name: object) -> str:
    text = str(name or "")
    if not text:
        return "-"
    if text.startswith("weapon_"):
        text = text[7:]
    replacements = {
        "smokegrenade": "smoke",
        "flashbang": "flash",
        "hegrenade": "he",
        "molotov": "molo",
        "incgrenade": "inc",
        "desert_eagle": "deagle",
        "five_seven": "57",
        "dual_berettas": "dual",
    }
    return replacements.get(text, text)


def _inventory_preview(player: dict[str, Any]) -> str:
    weapons = player.get("weapons", [])
    if not isinstance(weapons, list) or not weapons:
        return "-"
    compact = [_short_weapon_name(item) for item in weapons if item]
    if not compact:
        return "-"
    if len(compact) <= 4:
        return ",".join(compact)
    return ",".join(compact[:4]) + ",..."


def _flag_char(enabled: object, char: str) -> str:
    return char if bool(enabled) else "-"


def _util_flags(player: dict[str, Any]) -> str:
    return "".join(
        (
            _flag_char(player.get("has_smoke"), "S"),
            _flag_char(player.get("has_flash"), "F"),
            _flag_char(player.get("has_he"), "H"),
            _flag_char(player.get("has_molotov"), "M"),
            _flag_char(player.get("has_c4"), "C"),
        )
    )


def _slot_assignments(players: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    t_players = [player for player in players if str(player.get("team", "")) == "T"]
    ct_players = [player for player in players if str(player.get("team", "")) == "CT"]
    t_players.sort(key=lambda player: str(player.get("name", "")))
    ct_players.sort(key=lambda player: str(player.get("name", "")))

    for side_name, side_players in (("t", t_players), ("ct", ct_players)):
        for idx, player in enumerate(side_players[:5]):
            result.append((f"{side_name}{idx}", player))
    return result


def _weapon_name_from_id(weapon_id: float) -> str:
    idx = int(weapon_id)
    if 0 <= idx < len(WEAPONS):
        return WEAPONS[idx]
    return "unknown"


def _short_path(path: Path | None) -> str:
    if path is None:
        return "none"
    return str(path.name)


def _alive_positions(players: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    return [
        (
            float(player.get("x", 0.0) or 0.0),
            float(player.get("y", 0.0) or 0.0),
            float(player.get("z", 0.0) or 0.0),
        )
        for player in players
        if player.get("alive")
    ]


def _player_positions(players: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    return [
        (
            float(player.get("x", 0.0) or 0.0),
            float(player.get("y", 0.0) or 0.0),
            float(player.get("z", 0.0) or 0.0),
        )
        for player in players
    ]


def _map_status_text(
    map_name: str,
    map_source: str,
    state: MapResolutionState | None,
) -> str:
    if map_source == "cli":
        return "manual override"
    if map_source == "gamerules":
        return "memory-reported map"
    if map_source == "infer":
        return "coord inference (no cached map yet)"
    if map_source == "infer-confirmed":
        return "coord inference confirmed map switch"
    if map_source == "cached-pending":
        pending_map = state.pending_map if state is not None else ""
        pending_count = state.pending_count if state is not None else 0
        if pending_map:
            return (
                f"keeping cached map; pending switch to {pending_map} "
                f"({pending_count}/{_MAP_SWITCH_CONFIRM_FRAMES})"
            )
        return "keeping cached map while switch is pending"
    if map_source == "cached":
        if map_name:
            return "stable cached map"
        return "cached fallback"
    return "map unresolved"


def _row_slots(row: dict[str, float] | None, prefix: str, slot_count: int) -> list[tuple[float, float, float]]:
    if row is None:
        return []
    out: list[tuple[float, float, float]] = []
    for slot in range(slot_count):
        remain = float(row.get(f"{prefix}{slot}_remain", 0.0))
        if remain <= 0.0:
            continue
        x = float(row.get(f"{prefix}{slot}_x", 0.0))
        y = float(row.get(f"{prefix}{slot}_y", 0.0))
        out.append((x, y, remain))
    return out


def _vec_slots(row: dict[str, float] | None, prefix: str, slot_count: int) -> list[tuple[float, float, float]]:
    if row is None:
        return []
    vec = build_state_vector(row)
    out: list[tuple[float, float, float]] = []
    for slot in range(slot_count):
        remain = float(vec[FEATURE_IDX[f"{prefix}{slot}_remain"]])
        if remain <= 0.0:
            continue
        x = float(vec[FEATURE_IDX[f"{prefix}{slot}_x"]])
        y = float(vec[FEATURE_IDX[f"{prefix}{slot}_y"]])
        out.append((x, y, remain))
    return out


def _format_triplets(entries: list[tuple[float, float, float]], *, precision: int = 2) -> str:
    if not entries:
        return "-"
    fmt = f"{{:.{precision}f}}"
    return "  ".join(
        f"({fmt.format(x)},{fmt.format(y)},{fmt.format(remain)})"
        for x, y, remain in entries
    )


def _resolve_map_name(
    players: list[dict[str, Any]],
    map_state: dict[str, Any],
    map_override: str,
    state: MapResolutionState | None = None,
) -> tuple[str, str]:
    if map_override:
        if state is not None:
            state.cached_map = map_override
            state.pending_map = ""
            state.pending_count = 0
        return map_override, "cli"

    live_map = str(map_state.get("map_name", "") or "")
    if live_map:
        if state is not None:
            state.cached_map = live_map
            state.pending_map = ""
            state.pending_count = 0
        return live_map, "gamerules"

    player_positions = _player_positions(players)
    nonzero_positions = [pos for pos in player_positions if pos != (0.0, 0.0, 0.0)]
    cached_map = state.cached_map if state is not None else ""
    if cached_map and map_fit_fraction(player_positions, cached_map) >= _CACHED_MAP_KEEP_FRACTION:
        if state is not None:
            state.pending_map = ""
            state.pending_count = 0
        return cached_map, "cached"

    if len(nonzero_positions) < _MIN_POSITIONS_FOR_COORD_SWITCH:
        if state is not None and state.cached_map:
            state.pending_map = ""
            state.pending_count = 0
            return state.cached_map, "cached"
        return "", "unknown"

    inferred = infer_map_from_positions(player_positions)
    if inferred:
        if map_fit_fraction(player_positions, inferred) < _NEW_MAP_SWITCH_FRACTION:
            if state is not None and state.cached_map:
                state.pending_map = ""
                state.pending_count = 0
                return state.cached_map, "cached"
            return "", "unknown"
        if state is None:
            return inferred, "infer"
        if not state.cached_map:
            state.cached_map = inferred
            state.pending_map = ""
            state.pending_count = 0
            return inferred, "infer"
        if inferred == state.cached_map:
            state.pending_map = ""
            state.pending_count = 0
            return state.cached_map, "cached"
        if inferred == state.pending_map:
            state.pending_count += 1
        else:
            state.pending_map = inferred
            state.pending_count = 1
        if state.pending_count >= _MAP_SWITCH_CONFIRM_FRAMES:
            state.cached_map = inferred
            state.pending_map = ""
            state.pending_count = 0
            return inferred, "infer-confirmed"
        return state.cached_map, "cached-pending"

    if state is not None and state.cached_map:
        state.pending_map = ""
        state.pending_count = 0
        return state.cached_map, "cached"

    return "", "unknown"


def _capture_snapshot(
    reader: CS2MemoryReader,
    map_override: str,
    map_state_cache: MapResolutionState | None = None,
) -> Snapshot:
    players = reader.read_players()
    map_state = reader.read_map_state()
    bomb = reader.read_bomb()
    projectiles = reader.read_projectiles()

    merged_map_state = dict(map_state)
    merged_map_state["bomb"] = bomb
    merged_map_state["projectiles"] = projectiles

    map_name, map_source = _resolve_map_name(players, merged_map_state, map_override, map_state_cache)
    round_num = int(merged_map_state.get("round_num", 0) or 0)
    row = build_row_from_memory(
        players=players,
        map_state=merged_map_state,
        round_num=round_num,
        map_name=map_name,
    )

    if row is not None:
        row_status = f"ok ({len(row)} features)"
    elif not players:
        row_status = "missing: no live players found"
    elif not map_name:
        row_status = "missing: could not resolve map name"
    elif map_name not in MAPS:
        row_status = f"missing: unsupported map {map_name}"
    else:
        row_status = "missing: builder returned None"

    map_status = _map_status_text(map_name, map_source, map_state_cache)
    pending_map = map_state_cache.pending_map if map_state_cache is not None else ""
    pending_count = map_state_cache.pending_count if map_state_cache is not None else 0
    molotov_reader = [
        (float(x), float(y), float(remain))
        for x, y, remain in list(projectiles.get("molotovs", []))
        if float(remain) > 0.0
    ]
    molotov_row = _row_slots(row, "molotov", MOLOTOV_SLOTS)
    molotov_vec = _vec_slots(row, "molotov", MOLOTOV_SLOTS)

    return Snapshot(
        captured_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        map_name=map_name,
        map_source=map_source,
        map_status=map_status,
        pending_map=pending_map,
        pending_count=pending_count,
        players=players,
        map_state=merged_map_state,
        bomb=bomb,
        projectiles=projectiles,
        row=row,
        molotov_reader=molotov_reader,
        molotov_row=molotov_row,
        molotov_vec=molotov_vec,
        row_status=row_status,
        radar_path=_radar_path(map_name),
    )


def _prepare_map_axes(ax: Axes, snapshot: Snapshot) -> Projection:
    ax.clear()
    ax.set_facecolor(MAP_BG)

    if snapshot.radar_path is not None and snapshot.map_name in RADAR_META:
        image = plt.imread(snapshot.radar_path)
        height, width = image.shape[:2]
        ax.imshow(image, extent=(0.0, float(width), float(height), 0.0))
        ax.set_xlim(0.0, float(width))
        ax.set_ylim(float(height), 0.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(
            f"{snapshot.map_name} radar",
            loc="left",
            color=TEXT_COLOR,
            fontsize=14,
            fontweight="bold",
        )
        meta = RADAR_META[snapshot.map_name]

        def _project_xy(x: float, y: float) -> tuple[float, float]:
            return (
                (x - float(meta["pos_x"])) / float(meta["scale"]),
                (float(meta["pos_y"]) - y) / float(meta["scale"]),
            )

        return Projection(
            mode="radar",
            project_xy=_project_xy,
            yaw_len=38.0,
            label_dx=20.0,
            label_dy=-16.0,
            smoke_radius=30.0,
            molotov_radius=26.0,
        )

    bounds = _MAP_BOUNDS.get(snapshot.map_name)
    if bounds is None:
        coords: list[tuple[float, float]] = []
        coords.extend(
            (
                float(player.get("x", 0.0) or 0.0),
                float(player.get("y", 0.0) or 0.0),
            )
            for player in snapshot.players
            if player.get("alive")
        )
        if snapshot.bomb.get("planted") or snapshot.bomb.get("dropped"):
            coords.append(
                (
                    float(snapshot.bomb.get("x", 0.0) or 0.0),
                    float(snapshot.bomb.get("y", 0.0) or 0.0),
                )
            )
        if not coords:
            coords = [(-1000.0, -1000.0), (1000.0, 1000.0)]
        xs = [item[0] for item in coords]
        ys = [item[1] for item in coords]
        pad_x = max(250.0, (max(xs) - min(xs)) * 0.2)
        pad_y = max(250.0, (max(ys) - min(ys)) * 0.2)
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(max(ys) + pad_y, min(ys) - pad_y)
    else:
        x_min, x_max, y_min, y_max, _, _ = bounds
        pad_x = max(150.0, (x_max - x_min) * 0.04)
        pad_y = max(150.0, (y_max - y_min) * 0.04)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_max + pad_y, y_min - pad_y)

        for zone_name, rect in _ZONE_BOXES.get(snapshot.map_name, {}).items():
            zx0, zx1, zy0, zy1 = rect
            patch = Rectangle(
                (zx0, zy0),
                zx1 - zx0,
                zy1 - zy0,
                facecolor="#94a3b8",
                edgecolor="#cbd5e1",
                linewidth=1.0,
                alpha=0.08,
            )
            ax.add_patch(patch)
            ax.text(
                (zx0 + zx1) / 2.0,
                (zy0 + zy1) / 2.0,
                zone_name.upper(),
                color="#cbd5e1",
                fontsize=12,
                ha="center",
                va="center",
                alpha=0.75,
                fontweight="bold",
            )

    ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.35)
    ax.set_xlabel("world x", color="#cbd5e1")
    ax.set_ylabel("world y", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.set_aspect("equal")
    ax.set_title(
        f"{snapshot.map_name or 'unknown map'} world view",
        loc="left",
        color=TEXT_COLOR,
        fontsize=14,
        fontweight="bold",
    )
    return Projection(
        mode="world",
        project_xy=lambda x, y: (x, y),
        yaw_len=160.0,
        label_dx=70.0,
        label_dy=-55.0,
        smoke_radius=130.0,
        molotov_radius=105.0,
    )


def _draw_player(ax: Axes, projection: Projection, slot: str, player: dict[str, Any]) -> None:
    x = float(player.get("x", 0.0) or 0.0)
    y = float(player.get("y", 0.0) or 0.0)
    px, py = projection.project_xy(x, y)

    team = str(player.get("team", ""))
    alive = bool(player.get("alive")) and float(player.get("hp", 0.0) or 0.0) > 0.0
    color = T_COLOR if team == "T" else CT_COLOR if team == "CT" else "#ffffff"
    alpha = 0.95 if alive else 0.45
    marker = "o" if alive else "X"

    ax.scatter(
        [px],
        [py],
        s=180 if alive else 120,
        c=color if alive else DEAD_COLOR,
        marker=marker,
        edgecolors="black",
        linewidths=1.2,
        alpha=alpha,
        zorder=5,
    )

    yaw = float(player.get("yaw", 0.0) or 0.0)
    if alive:
        radians = math.radians(yaw)
        dx = math.cos(radians) * projection.yaw_len
        dy = -math.sin(radians) * projection.yaw_len
        ax.annotate(
            "",
            xy=(px + dx, py + dy),
            xytext=(px, py),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.0, alpha=0.9),
            zorder=6,
        )

    hp = int(player.get("hp", 0) or 0)
    label = f"{slot} {str(player.get('name', ''))[:14]} [{hp}]"
    ax.text(
        px + projection.label_dx,
        py + projection.label_dy,
        label,
        color=TEXT_COLOR,
        fontsize=8.5,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", edgecolor=color, alpha=0.88),
        zorder=7,
    )


def _draw_bomb(ax: Axes, projection: Projection, bomb: dict[str, Any]) -> None:
    if not (bomb.get("planted") or bomb.get("dropped")):
        return
    x = float(bomb.get("x", 0.0) or 0.0)
    y = float(bomb.get("y", 0.0) or 0.0)
    px, py = projection.project_xy(x, y)
    planted = bool(bomb.get("planted"))
    label = f"C4 {'P' if planted else 'D'}"
    if bomb.get("site"):
        label += f" {bomb.get('site')}"
    ax.scatter(
        [px],
        [py],
        s=260,
        c=BOMB_COLOR,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        zorder=8,
    )
    ax.text(
        px + 16.0,
        py + 16.0,
        label,
        color=BOMB_COLOR,
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#111827", edgecolor=BOMB_COLOR, alpha=0.9),
        zorder=9,
    )


def _draw_projectiles(ax: Axes, projection: Projection, projectiles: dict[str, Any]) -> None:
    smokes = list(projectiles.get("smokes", []))
    molotovs = list(projectiles.get("molotovs", []))

    for idx, item in enumerate(smokes):
        if len(item) < 3:
            continue
        x, y, remain = float(item[0]), float(item[1]), float(item[2])
        px, py = projection.project_xy(x, y)
        patch = Circle(
            (px, py),
            radius=projection.smoke_radius,
            facecolor=SMOKE_COLOR,
            edgecolor="#d1d5db",
            linewidth=1.3,
            alpha=min(0.18 + remain * 0.32, 0.55),
            zorder=3,
        )
        ax.add_patch(patch)
        ax.scatter(
            [px],
            [py],
            s=60,
            c="#f8fafc",
            marker="o",
            edgecolors="#1f2937",
            linewidths=0.9,
            zorder=4,
        )
        ax.text(px, py, f"S{idx}", color="#111827", fontsize=8, fontweight="bold", ha="center", va="center", zorder=5)

    for idx, item in enumerate(molotovs):
        if len(item) < 3:
            continue
        x, y, remain = float(item[0]), float(item[1]), float(item[2])
        px, py = projection.project_xy(x, y)
        patch = Circle(
            (px, py),
            radius=projection.molotov_radius,
            facecolor=MOLOTOV_COLOR,
            edgecolor="#ef4444",
            linewidth=2.0,
            alpha=min(0.28 + remain * 0.42, 0.82),
            zorder=3.5,
        )
        ax.add_patch(patch)
        ax.scatter(
            [px],
            [py],
            s=170,
            c="#ef4444",
            marker="X",
            edgecolors="#fff7ed",
            linewidths=1.0,
            zorder=4.5,
        )
        ax.text(
            px,
            py,
            f"M{idx}",
            color="#fff7ed",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )


def _panel_player_line(
    slot: str,
    player: dict[str, Any],
    row: dict[str, float] | None,
    map_name: str,
) -> str:
    prefix = slot
    name = str(player.get("name", ""))[:12]
    team = str(player.get("team", ""))
    hp = int((row or {}).get(f"{prefix}_hp", float(player.get("hp", 0) or 0.0)))
    armor = int((row or {}).get(f"{prefix}_armor", float(player.get("armor", 0) or 0.0)))
    balance = int((row or {}).get(f"{prefix}_balance", float(player.get("money", 0) or 0.0)))
    equip = int((row or {}).get(f"{prefix}_equip_value", float(_equip_value(player))))
    weapon_id = int((row or {}).get(f"{prefix}_weapon_id", float(_active_weapon_id(player))))
    alive = int((row or {}).get(f"{prefix}_alive", float(bool(player.get("alive")))))
    yaw = float((row or {}).get(f"{prefix}_yaw", float(player.get("yaw", 0.0) or 0.0)))
    bomb_zone = int((row or {}).get(f"{prefix}_in_bomb_zone", 0.0))
    x = float((row or {}).get(f"{prefix}_x", float(player.get("x", 0.0) or 0.0)))
    y = float((row or {}).get(f"{prefix}_y", float(player.get("y", 0.0) or 0.0)))
    z = float((row or {}).get(f"{prefix}_z", float(player.get("z", 0.0) or 0.0)))
    zone = classify_zone(x, y, map_name, z) if map_name else "other"
    active_name = _short_weapon_name(player.get("active_weapon_class"))
    weapon_name = _short_weapon_name(_weapon_name_from_id(float(weapon_id)))
    return (
        f"{team:<2} {slot:<3} {name:<12} "
        f"A={alive} HP={hp:>3} AR={armor:>3} $={balance:>5} EQ={equip:>4} "
        f"WID={weapon_id:>2}({weapon_name:<7}) ACT={active_name:<7} "
        f"BZ={bomb_zone} ZN={zone:<5} Y={yaw:>6.1f} "
        f"XYZ=({x:>6.0f},{y:>6.0f},{z:>5.0f}) "
        f"F={_util_flags(player)} INV={_inventory_preview(player)}"
    )


def _projectile_line(prefix: str, entries: list[tuple[float, float, float]]) -> str:
    if not entries:
        return f"{prefix}: -"
    formatted = "  ".join(f"({x:.0f},{y:.0f},{remain:.2f})" for x, y, remain in entries[:5])
    return f"{prefix}: {formatted}"


def _render_panel(ax: Axes, snapshot: Snapshot) -> None:
    ax.clear()
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")

    bomb_bits: list[str] = []
    if snapshot.bomb.get("planted"):
        bomb_bits.append("planted")
    if snapshot.bomb.get("dropped"):
        bomb_bits.append("dropped")
    if snapshot.bomb.get("site"):
        bomb_bits.append(f"site={snapshot.bomb.get('site')}")
    bomb_text = ", ".join(bomb_bits) if bomb_bits else "idle"

    score_ct = int(snapshot.map_state.get("ct_score", 0) or 0)
    score_t = int(snapshot.map_state.get("t_score", 0) or 0)
    round_num = int(snapshot.map_state.get("round_num", 0) or 0)
    time_in_round = float(snapshot.map_state.get("time_in_round", 0.0) or 0.0)
    round_phase = str(snapshot.map_state.get("round_phase", "") or "-")
    map_phase = str(snapshot.map_state.get("map_phase", "") or "-")
    radar_text = _short_path(snapshot.radar_path)

    lines = [
        "memory_visualize.py",
        "",
        f"time        : {snapshot.captured_at}",
        f"map         : {snapshot.map_name or '-'}",
        f"map status  : {snapshot.map_status}",
        f"map source  : {snapshot.map_source}",
        f"map switch  : pending {snapshot.pending_map} ({snapshot.pending_count}/{_MAP_SWITCH_CONFIRM_FRAMES})"
        if snapshot.pending_map
        else "map switch  : stable",
        f"phase       : map={map_phase} round={round_phase}",
        f"score/round : CT {score_ct} - {score_t} T   round={round_num}   t={time_in_round:.2f}s",
        f"row status  : {snapshot.row_status}",
        f"radar asset : {radar_text}",
        f"radar set   : found={', '.join(sorted(path.name for path in _available_radar_paths().values())) or 'none'}",
        f"players     : {len(snapshot.players)}   smokes={len(snapshot.projectiles.get('smokes', []))}   "
        f"molotovs={len(snapshot.projectiles.get('molotovs', []))}",
        f"bomb        : {bomb_text}",
        "",
        "slot order matches training/demo rows: alphabetical by name within each side",
        "",
    ]

    assignments = _slot_assignments(snapshot.players)
    if assignments:
        for slot, player in assignments:
            lines.append(_panel_player_line(slot, player, snapshot.row, snapshot.map_name))
    else:
        lines.append("no live T/CT players found")

    lines.extend(
        [
            "",
            f"molotov pipe: reader={len(snapshot.molotov_reader)} row={len(snapshot.molotov_row)} vec={len(snapshot.molotov_vec)}",
            f"molotov raw : {_format_triplets(snapshot.molotov_reader, precision=1)}",
            f"molotov row : {_format_triplets(snapshot.molotov_row, precision=1)}",
            f"molotov vec : {_format_triplets(snapshot.molotov_vec, precision=3)}",
            "",
            _projectile_line("smokes", list(snapshot.projectiles.get("smokes", []))),
            _projectile_line("molotovs", list(snapshot.projectiles.get("molotovs", []))),
        ]
    )

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.3,
        family="monospace",
        color=TEXT_COLOR,
    )


def _create_window() -> tuple[Figure, Axes, Axes]:
    fig = plt.figure(figsize=(20, 11), facecolor=PANEL_BG)
    ax_map = fig.add_axes([0.025, 0.07, 0.64, 0.86], facecolor=MAP_BG)
    ax_panel = fig.add_axes([0.69, 0.05, 0.285, 0.90], facecolor=PANEL_BG)
    return fig, ax_map, ax_panel


def _draw_snapshot(fig: Figure, ax_map: Axes, ax_panel: Axes, snapshot: Snapshot) -> None:
    projection = _prepare_map_axes(ax_map, snapshot)
    _draw_projectiles(ax_map, projection, snapshot.projectiles)
    _draw_bomb(ax_map, projection, snapshot.bomb)

    for slot, player in _slot_assignments(snapshot.players):
        _draw_player(ax_map, projection, slot, player)

    _render_panel(ax_panel, snapshot)
    fig.suptitle(
        "CS2 live memory parity view",
        color=TEXT_COLOR,
        fontsize=18,
        fontweight="bold",
    )


def _single_frame(reader: CS2MemoryReader, map_override: str) -> int:
    fig, ax_map, ax_panel = _create_window()
    snapshot = _capture_snapshot(reader, map_override, MapResolutionState())
    _draw_snapshot(fig, ax_map, ax_panel, snapshot)
    plt.show()
    return 0


def _watch_loop(reader: CS2MemoryReader, map_override: str, watch: float) -> int:
    plt.ion()
    fig, ax_map, ax_panel = _create_window()
    map_state_cache = MapResolutionState(cached_map=map_override)

    running = {"value": True}

    def _on_close(_event: object) -> None:
        running["value"] = False

    fig.canvas.mpl_connect("close_event", _on_close)

    while running["value"]:
        snapshot = _capture_snapshot(reader, map_override, map_state_cache)
        _draw_snapshot(fig, ax_map, ax_panel, snapshot)
        fig.canvas.draw_idle()
        plt.pause(max(0.05, watch))

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Visualize live CS2 memory reads on the radar map and show training-row slot mapping."
    )
    ap.add_argument(
        "--watch",
        type=float,
        default=0.5,
        help="Refresh interval in seconds. Use 0 to render one frame and keep the window open.",
    )
    ap.add_argument(
        "--map",
        default="",
        help="Override the map name used for row building and radar lookup, e.g. de_dust2.",
    )
    args = ap.parse_args()

    if plt is None:
        print("ERROR: matplotlib is not installed. Install it with `pip install matplotlib`.")
        return 1

    try:
        reader = CS2MemoryReader.attach()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.watch <= 0:
        return _single_frame(reader, args.map)
    return _watch_loop(reader, args.map, args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
