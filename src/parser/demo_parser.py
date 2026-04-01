"""Demo parser — reads CS2 .dem files and writes per-round state sequences (parquet).

Output schema (one row per step):
    demo_name, round_num, step, tick, bomb_site, map_zone,
    t{0..4}_{x,y,z,hp,armor,helmet,alive},
    ct{0..4}_{x,y,z,hp,armor,helmet,alive}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.label_extractor import extract_bomb_site, get_plant_ticks
from src.utils.map_utils import classify_zone, normalize_coords

try:
    from demoparser2 import DemoParser
except ImportError:  # pragma: no cover
    DemoParser = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
TICK_RATE: int = 64
TARGET_RATE: int = 8
DOWNSAMPLE: int = TICK_RATE // TARGET_RATE   # 8
PRE_PLANT_SECS: int = 30
MAX_STEPS: int = PRE_PLANT_SECS * TARGET_RATE  # 240

PLAYER_PROPS: list[str] = [
    "X", "Y", "Z",
    "health",
    "armor_value",
    "has_helmet",
    "is_alive",
    "team_name",
    "name",
]

_TEAM_T = "TERRORIST"
_TEAM_CT = "CT"


# ── Public API ─────────────────────────────────────────────────────────────

def parse_demo(dem_path: Path | str, output_dir: Path | str) -> Optional[Path]:
    """Parse one CS2 demo and write a per-round state-sequence parquet.

    Only rounds with a confirmed bomb plant ('A' or 'B') are included.

    Args:
        dem_path: Path to the .dem file.
        output_dir: Directory for the output parquet.

    Returns:
        Path to the written parquet, or None if no valid rounds found.
    """
    dem_path = Path(dem_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Parsing demo: %s", dem_path.name)

    if DemoParser is None:  # pragma: no cover
        raise ImportError(
            "demoparser2 is not installed — run: pip install demoparser2"
        )

    parser = DemoParser(str(dem_path))

    try:
        plant_df = parser.parse_event("bomb_planted", other_props=["site"])
    except Exception:  # pragma: no cover
        logger.exception("parse_event('bomb_planted') failed in %s", dem_path.name)
        return None

    if plant_df is None or plant_df.empty:
        logger.warning("No bomb_planted events in %s — skipping", dem_path.name)
        return None

    sites = extract_bomb_site(plant_df)
    plant_ticks = get_plant_ticks(plant_df)
    map_name = _get_map_name(parser)

    sequences: list[pd.DataFrame] = []
    for round_num, (plant_tick, bomb_site) in enumerate(
        zip(plant_ticks, sites), start=1
    ):
        if bomb_site not in ("A", "B"):  # pragma: no cover
            logger.debug("Round %d: site %r not A/B — skipping", round_num, bomb_site)
            continue
        seq = _extract_sequence(parser, round_num, int(plant_tick), bomb_site, map_name)
        if seq is not None:
            sequences.append(seq)
            logger.debug("Round %d → site %s (%d steps)", round_num, bomb_site, len(seq))

    if not sequences:  # pragma: no cover
        logger.warning("No valid sequences extracted from %s", dem_path.name)
        return None

    result = pd.concat(sequences, ignore_index=True)
    result.insert(0, "demo_name", dem_path.stem)

    out_path = output_dir / f"{dem_path.stem}.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Saved %d rows → %s", len(result), out_path)
    return out_path


def parse_demos_batch(  # pragma: no cover
    dem_dir: Path | str,
    output_dir: Path | str,
    glob: str = "*.dem",
) -> list[Path]:
    """Parse all .dem files in dem_dir, writing one parquet per demo.

    Errors on individual demos are logged and skipped.

    Args:
        dem_dir: Directory containing .dem files.
        output_dir: Output directory for parquet files.
        glob: File glob (default '*.dem').

    Returns:
        List of successfully written parquet paths.
    """
    from tqdm import tqdm  # optional progress bar

    dem_dir = Path(dem_dir)
    dem_files = sorted(dem_dir.glob(glob))
    if not dem_files:
        logger.warning("No files matching '%s' in %s", glob, dem_dir)
        return []

    results: list[Path] = []
    for dem in tqdm(dem_files, desc="Parsing demos"):
        try:
            out = parse_demo(dem, output_dir)
            if out:
                results.append(out)
        except Exception:
            logger.exception("Error parsing %s — skipping", dem.name)
    return results


# ── Private helpers ────────────────────────────────────────────────────────

def _get_map_name(parser) -> str:  # noqa: ANN001
    try:
        return parser.parse_header().get("map_name", "")
    except Exception:  # pragma: no cover
        return ""


def _extract_sequence(
    parser,  # noqa: ANN001
    round_num: int,
    plant_tick: int,
    bomb_site: str,
    map_name: str,
) -> Optional[pd.DataFrame]:
    """Build a downsampled state DataFrame for the 30 s window before a plant.

    Args:
        parser: demoparser2.DemoParser instance.
        round_num: 1-based round index (stored in output).
        plant_tick: Tick of the bomb_planted event.
        bomb_site: 'A' or 'B'.
        map_name: e.g. 'de_mirage'.

    Returns:
        DataFrame (one row per step) or None on failure.
    """
    start_tick = max(0, plant_tick - PRE_PLANT_SECS * TICK_RATE)
    wanted_ticks = list(range(start_tick, plant_tick + 1, DOWNSAMPLE))

    try:
        tick_df = parser.parse_ticks(PLAYER_PROPS, wanted_ticks)
    except Exception:  # pragma: no cover
        logger.exception("parse_ticks failed for round %d", round_num)
        return None

    if tick_df is None or tick_df.empty:
        logger.warning("Empty tick data for round %d", round_num)
        return None

    rows: list[dict] = []
    for step, tick in enumerate(wanted_ticks):
        tick_slice = tick_df[tick_df["tick"] == tick]
        if tick_slice.empty:
            continue
        rows.append(
            _build_state_row(tick_slice, step, tick, round_num, bomb_site, map_name)
        )

    return pd.DataFrame(rows) if rows else None  # pragma: no branch


def _build_state_row(
    tick_slice: pd.DataFrame,
    step: int,
    tick: int,
    round_num: int,
    bomb_site: str,
    map_name: str,
) -> dict:
    """Flatten one tick's player data into a single state dict.

    Players on each side are sorted by name for reproducibility.
    Missing players (< 5 per side) are zero-padded.

    Returns:
        Flat dict with keys round_num, step, tick, bomb_site, map_zone,
        and t{i}/ct{i} suffixed position/status columns.
    """
    t_rows = tick_slice[tick_slice["team_name"] == _TEAM_T].sort_values("name")
    ct_rows = tick_slice[tick_slice["team_name"] == _TEAM_CT].sort_values("name")

    # map_zone tracks where the T side is concentrated
    t_x_mean = float(t_rows["X"].mean()) if not t_rows.empty else 0.0
    t_y_mean = float(t_rows["Y"].mean()) if not t_rows.empty else 0.0

    row: dict = {
        "round_num": round_num,
        "step": step,
        "tick": tick,
        "bomb_site": bomb_site,
        "map_zone": classify_zone(t_x_mean, t_y_mean, map_name),
    }

    for side_prefix, side_rows in (("t", t_rows), ("ct", ct_rows)):
        for i in range(5):
            prefix = f"{side_prefix}{i}"
            if i < len(side_rows):
                p = side_rows.iloc[i]
                x_n, y_n, z_n = normalize_coords(
                    float(p.get("X", 0.0)),
                    float(p.get("Y", 0.0)),
                    float(p.get("Z", 0.0)),
                    map_name,
                )
                row[f"{prefix}_x"] = x_n
                row[f"{prefix}_y"] = y_n
                row[f"{prefix}_z"] = z_n
                row[f"{prefix}_hp"] = int(p.get("health", 0))
                row[f"{prefix}_armor"] = int(p.get("armor_value", 0))
                row[f"{prefix}_helmet"] = bool(p.get("has_helmet", False))
                row[f"{prefix}_alive"] = bool(p.get("is_alive", False))
            else:
                row[f"{prefix}_x"] = 0.0
                row[f"{prefix}_y"] = 0.0
                row[f"{prefix}_z"] = 0.0
                row[f"{prefix}_hp"] = 0
                row[f"{prefix}_armor"] = 0
                row[f"{prefix}_helmet"] = False
                row[f"{prefix}_alive"] = False

    return row
